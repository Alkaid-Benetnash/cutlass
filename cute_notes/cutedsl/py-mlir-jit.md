# Python-MLIR JIT Infrastructure

## Argument Marshalling

CuTe DSL JIT-compiles Python functions into MLIR IR. Python objects (tensors, layouts, copy atoms, etc.) must be **flattened** into lists of MLIR SSA values for `func.func` signatures, then **reconstructed** on the other side so the function body can manipulate them as normal Python objects.

### The Protocol

Every DSL-compatible type implements three methods (the `DynamicExpression` protocol, `base_dsl/typing.py:48`):

| Method | Direction | Purpose |
|--------|-----------|---------|
| `__extract_mlir_values__()` | Python → flat `[ir.Value]` | Serialize object to MLIR values |
| `__get_mlir_types__()` | Python → flat `[ir.Type]` | Get MLIR types (for function signatures) |
| `__new_from_mlir_values__(values)` | flat `[ir.Value]` → Python | Reconstruct object from new MLIR values |

### Top-Level Functions

Two top-level functions in `base_dsl/dsl.py` drive the marshalling recursively:

- **`extract_mlir_values(obj)`** (`dsl.py:167`) — Walks `obj` recursively, collecting all `ir.Value`s into a flat list. Dispatches via `__extract_mlir_values__` for protocol-implementing objects; recurses into tuples, lists, and `SimpleNamespace`; passes through bare `ir.Value`s directly.

- **`new_from_mlir_values(obj, values)`** (`dsl.py:195`) — The inverse. Given the original `obj` (as a structural template) and a flat list of new `ir.Value`s, reconstructs a new Python object with the same structure but new underlying values. Consumes values from the front of the list, using `get_mlir_types()` to determine how many values each sub-object needs.

A companion function `get_mlir_types(obj)` (`base_dsl/typing.py:242`) extracts types without needing values, used to determine MLIR function signatures and to calculate value counts during reconstruction.

### Example: `_Tensor`

A `_Tensor` wraps a single `ir.Value` (a memref or pointer) plus element type metadata (`cute/tensor.py:160-174`):

```python
def __extract_mlir_values__(self):
    return [self.value]          # one ir.Value

def __new_from_mlir_values__(self, values):
    assert len(values) == 1
    return _Tensor(values[0], dtype=self.element_type)
```

The `dtype` is static metadata — it doesn't become an MLIR function argument. Only the memref/pointer value crosses the function boundary.

### Primary Call Site

During JIT compilation, `gen_exec_args` (`dsl.py:630-659`) processes each Python argument to a JIT function:

1. Creates MLIR `func.func` block arguments matching the flattened types of all Python arguments
2. Calls `new_from_mlir_values(arg, blk_args)` to wrap those block arguments back into Python objects
3. The function body then uses these reconstructed objects normally

This is what allows users to write `cute.copy(tma_atom, src_tensor, dst_tensor)` in Python while the DSL maps the underlying MLIR values through function boundaries transparently.

### Supported Container Types

`new_from_mlir_values` handles these cases beyond protocol dispatch:

- **tuple / list** — Recurse per element, preserving container type
- **SimpleNamespace** — Recurse per attribute
- **Bare `ir.Value`** (dynamic expression) — Replace with new value
- **Static values** (Python `int`, `str`, etc.) — Return unchanged, consume no values
- **`set`** — Rejected (non-deterministic ordering)
