# CuTe DSL Loop Control: `range`, `range_constexpr`, and Unrolling

**Official docs:** `media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.rst`

## Overview

CuTe DSL has three distinct ways to write `for` loops, each with different compile-time behavior. The key distinction is whether the loop is lowered to an MLIR `scf.ForOp` (a real runtime loop) or executed by the Python interpreter at kernel compilation time (full code duplication).

## `cutlass.range(...)` — Dynamic Runtime Loop

The **default**. The AST preprocessor transforms the loop body into a function decorated with `@loop_selector`, which lowers to an `scf.ForOp` in MLIR IR. The loop variable is a dynamic runtime value.

```python
for k in cutlass.range(k_tile_count):
    # k is a dynamic runtime value
    # loop body appears once in generated code
```

### Preprocessor path

`visit_For` (`ast_preprocessor.py:1155`) → detects `range_kind == "range"` → calls `transform_for_loop()` → wraps body in `@loop_selector` decorator → `loop_selector` (`ast_helpers.py:182`) → `executor.for_execute()` → `_loop_execute_range_dynamic()` (`cutlass_ast_decorators.py:281`) → creates `scf.ForOp`.

### Keyword options

| Keyword | Default | Effect |
|---------|---------|--------|
| `unroll=N` | `-1` (none) | Attaches `#llvm.loop_annotation<unroll = <count = N : i32>>` to the `scf.ForOp`. Hints the LLVM backend to unroll N iterations. |
| `unroll=1` | — | Special case: sets `<disable = true>`, explicitly **disabling** unrolling. |
| `unroll_full=True` | `False` | Attaches `#llvm.loop_annotation<unroll = <full = true>>`. Hints the LLVM backend to **fully unroll** the loop. |
| `prefetch_stages=N` | `None` | Attaches a `cutlass.pipelining` attribute to the `scf.ForOp` (for software pipelining — see below). |

These are **compiler hints** — the loop still exists as a single `scf.ForOp` in IR, but the LLVM backend may unroll it based on the annotation. The loop bounds can be dynamic.

### Unroll attribute generation

From `cutlass_ast_decorators.py:324-328`:
```python
if unroll_full:
    unroll_attr = LoopUnroll(full=True)
elif unroll != -1:
    unroll_attr = LoopUnroll(count=unroll)
```

`LoopUnroll` (`cutlass_ast_decorators.py:33`) generates `#llvm.loop_annotation<unroll = <...>>` MLIR attributes. When `count=1`, it additionally sets `disable = true`.

### Software pipelining (`prefetch_stages`)

The `prefetch_stages=N` keyword asks the compiler to automatically generate a prefetch loop and restructure the main loop for software pipelining. Instead of manually writing a prefetch loop + main loop, you write a single loop and the compiler handles the rest:

```python
# Without prefetch_stages: manual pipelining
for i in range(prefetch_stages):
    cute.copy(atom, gmem[i], buffer[i], ...)
for i in range(bound):
    if i + prefetch_stages < bound:
        cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)
    use(buffer[i % total_stages])

# With prefetch_stages: compiler handles pipelining
for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
    cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
    use(buffer[i % total_stages])
```

This feature is experimental and only supported on SM90+.

### Examples from the codebase

```python
# Hint LLVM to fully unroll (bounds can be dynamic, but LLVM must resolve them)
for k_block in cutlass.range(num_k_block, unroll_full=True)
# Source: examples/python/CuTeDSL/ampere/tensorop_gemm.py:614

# Explicitly disable unrolling (keep as loop)
for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1)
# Source: examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py:1085

# Hint LLVM to unroll by factor of 4
for i in cutlass.range(n, unroll=4)
```

## `cutlass.range_constexpr(...)` — Compile-Time Loop Unrolling

The preprocessor **does not** transform this into an `scf.ForOp`. Instead:

1. Validates all arguments are Python-level constants via `range_value_check()` (`ast_helpers.py:487-521`). If any argument is a dynamic DSL expression, raises an error.
2. Rewrites `range_constexpr(...)` → plain Python `range(...)` in the AST (`ast_preprocessor.py:1163`).
3. Leaves the `for` as a normal Python loop — does **not** call `transform_for_loop()`.

### Preprocessor path

`visit_For` (`ast_preprocessor.py:1155`) → detects `range_kind == "range_constexpr"` → calls `generic_visit(node)` to recurse into the body → rewrites `range_constexpr` to `range` → inserts `range_value_check` call → returns the node as-is (a plain Python `for` loop).

### Behavior

Since it's a plain Python `for` loop, the Python interpreter executes it during kernel compilation. Each iteration **generates its own copy** of the loop body in the IR. The loop variable is a Python `int`, usable for compile-time indexing, type dispatch, etc.

```python
# Each iteration emits separate IR — i is a Python int
for i in cutlass.range_constexpr(cute.size(tAsA.shape[1])):
    cute.copy(gmem_tiled_copy, tAgA[:, :, i], tAsA[:, :, i])
```

### Compile-time warning

`range_value_check` warns if the range has ≥64 iterations (`ast_helpers.py:509-514`):
> This static loop has N iterations, which may be very slow to compile, consider using `cutlass.range(..., unroll_full=True)` instead.

### Note on plain `range()` and non-range iterables

A bare `range()` or a non-range iterable (e.g., iterating over a Python list) hits the `range_kind == None` path in `visit_For` (`ast_preprocessor.py:1158`), which is treated **identically** to `range_constexpr` — the loop is kept as a Python-level loop and not lowered to IR.

## Comparison Table

| | `cutlass.range(...)` | `cutlass.range(..., unroll_full=True)` | `cutlass.range_constexpr(...)` |
|---|---|---|---|
| **Bounds** | Dynamic or static | Dynamic or static | Must be Python constants |
| **IR output** | Single `scf.ForOp` | Single `scf.ForOp` + LLVM unroll hint | No loop — body duplicated N times |
| **Loop variable** | Runtime IR value | Runtime IR value | Python `int` |
| **Unrolling** | By LLVM (if hinted) | By LLVM (requested) | By Python interpreter (guaranteed) |
| **Compile-time indexing / type dispatch** | No | No | Yes |
| **Code size risk** | Low | Depends on LLVM | Proportional to iteration count |

## When to Use Which

- **`range_constexpr`**: When you need the loop variable as a compile-time constant — e.g., indexing into heterogeneous tensor partitions, dispatching on fragment index, or iterating over a fixed number of inputs with different types. Also required when the loop body changes types across iterations (the error message at `cutlass_ast_decorators.py:157-164` suggests switching to `range_constexpr` when type mismatches occur inside dynamic loops).

- **`range(..., unroll_full=True)`**: When the loop body is uniform across iterations but you want it unrolled for performance. The loop variable is still dynamic, so you can't use it for compile-time dispatch, but LLVM can optimize the unrolled code. Preferred over `range_constexpr` for large iteration counts (≥64) since it avoids Python-level code duplication.

- **`range(..., unroll=1)`**: When you explicitly want to prevent unrolling (e.g., a long-running k-tile loop where code size matters more than eliminating loop overhead).

- **`range(...)`** (default): Normal runtime loop with no unroll hints.
