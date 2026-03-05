# CuTe DSL: `Constexpr`, `const_expr()`, and Calling Conventions

**Official docs:** `media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.rst`, `dsl_introduction.rst`, `dsl_control_flow.rst`

## The `Constexpr` type annotation

`Constexpr` (`base_dsl/typing.py:1803`) marks a function parameter as a compile-time constant at JIT entry points. It is a type annotation, not a runtime wrapper.

```python
class Constexpr(Generic[TY]):
    """Value is passed and computed by python interpreter"""
    pass
```

Usage forms:
```python
op: cutlass.Constexpr                              # unparameterized
mnkl: cutlass.Constexpr[tuple[int, int, int, int]]  # parameterized
epilogue_op: cutlass.Constexpr = lambda x: x        # with default
```

## What it does

When the JIT processes function arguments (`dsl.py:732`, `jit_arg_adapters.py:29-54`), `Constexpr`-annotated parameters are:

1. **Excluded from the MLIR function signature** — no IR argument is generated.
2. **Kept as Python values** — passed directly to the function body as closure variables during IR generation.
3. **Baked into compiled code** — different `Constexpr` values produce different compiled kernels (cached separately).

## Auto-constexpr rules

From `is_argument_constexpr()` (`jit_arg_adapters.py:57-75`), these are automatically constexpr **without** the annotation:
- `self` / `cls` (first parameter)
- `Type[X]` arguments (a Python type passed as a value)
- `None` arguments

## Calling conventions

From `dsl_introduction.rst`:

| Caller | Callee | Allowed | Semantics |
|--------|--------|---------|-----------|
| Python function | `@jit` | Yes | DSL runtime (JIT entry point) |
| Python function | `@kernel` | No | Error |
| `@jit` | `@jit` | Yes | Compile-time call, inlined |
| `@jit` | Python function | Yes | Compile-time call, inlined |
| `@jit` | `@kernel` | Yes | Dynamic call via GPU driver |
| `@kernel` | `@jit` | Yes | Compile-time call, inlined |
| `@kernel` | Python function | Yes | Compile-time call, inlined |
| `@kernel` | `@kernel` | No | Error |

**Key insight:** When `@jit` or `@kernel` calls a plain Python function, that call is a **compile-time inlined call** — the Python interpreter executes it during IR generation. Everything in the Python function is already "constexpr" by definition. The `Constexpr` annotation only matters at the **JIT entry boundary** (Python → `@jit`, or Python → `@kernel` via `@jit`), where the DSL must decide whether to lower an argument to MLIR IR or keep it as a Python value.

## Compile-time metaprogramming pattern

From `dsl_control_flow.rst`:

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # main GEMM work
    ...
    if cutlass.const_expr(do_relu):    # compile-time guard
        # ReLU code is emitted only when do_relu is True
        ...

gemm(..., False)   # ReLU is omitted from the generated IR
gemm(..., True)    # ReLU is included
```

## `const_expr()` for compile-time conditionals

`cutlass.const_expr(expr)` (`ast_helpers.py:348`) validates that its argument is a compile-time Python value, then returns it. Used to mark `if` and `while` conditions as compile-time:

```python
if cutlass.const_expr(some_constexpr_var):  # compile-time branch
    ...

while cutlass.const_expr(n < 10):  # compile-time while loop
    n += 1
```

If a dynamic expression is passed, it raises an error suggesting to remove `const_expr()`.
