# RFC: Make `Constexpr` compatible with Python type checkers

## Problem

`Constexpr` is currently defined as an empty generic class:

```python
# base_dsl/typing.py:1803
class Constexpr(Generic[TY]):
    """Value is passed and computed by python interpreter"""
    pass
```

When used as a type annotation, the type checker sees the parameter as an instance of `Constexpr`, not the inner type:

```python
@cute.kernel
def gemm(..., epilogue_op: cutlass.Constexpr[Callable[[Vec], Vec]]):
    result = epilogue_op(acc_vec)  # Type error: Constexpr has no __call__
```

This affects all `Constexpr[X]` usage — callability, arithmetic, indexing, attribute access — because the checker knows nothing about `X` from `Constexpr[X]`.

## Current JIT detection

Three sites detect `Constexpr` annotations at runtime, all operating on the raw annotation object from `inspect.getfullargspec().annotations`:

### 1. `is_arg_spec_constexpr()` — `jit_arg_adapters.py:29-54`

```python
(isinstance(arg_spec, type) and issubclass(arg_spec, Constexpr))  # bare Constexpr
or (get_origin(arg_spec) is Constexpr)                            # Constexpr[X]
```

### 2. `is_constexpr_field()` — `tree_utils.py:114-122`

```python
field.type is Constexpr            # bare Constexpr
or get_origin(field.type) is Constexpr  # Constexpr[X]
```

### 3. `_validate_arg()` — `cutlass.py:780-783`

Delegates to `is_arg_spec_constexpr()` to skip validation for constexpr args.

## Existing usage in the codebase

- **~40+ sites** use bare `Constexpr` (no type parameter): `op: cutlass.Constexpr`
- **~25 sites** use parameterized form: `Constexpr[int]`, `Constexpr[tuple[...]]`, `Constexpr[bool]`
- Bare `Constexpr` is equally opaque to type checkers under both proposals — the benefit applies only to the parameterized `Constexpr[X]` form.

## Proposed approaches

### Approach A: `Annotated` — unified definition

Replace the `Constexpr` class with a `typing.Annotated` type alias.

**Definition:**

```python
# base_dsl/typing.py

from typing import Annotated, TypeVar

class _ConstexprMeta:
    """Marker for compile-time constant parameters in CuTe DSL JIT."""
    pass

_CONSTEXPR_MARKER = _ConstexprMeta()

T = TypeVar("T")
Constexpr = Annotated[T, _CONSTEXPR_MARKER]
```

**Runtime behavior:**

```python
Constexpr[int]                        # → Annotated[int, _CONSTEXPR_MARKER]
Constexpr[Callable[[int], str]]       # → Annotated[Callable[[int], str], _CONSTEXPR_MARKER]
Constexpr                             # → Annotated[T, _CONSTEXPR_MARKER] (unresolved TypeVar)
get_origin(Constexpr[int])            # → Annotated (not Constexpr)
```

**Type checker behavior:**

Type checkers treat `Annotated[X, metadata]` as `X` for all purposes. So `op: Constexpr[Callable[[int], str]]` allows `op(42)` without errors.

**Required JIT detection changes:**

All three detection sites must be updated. The old `issubclass`/`isinstance`/`get_origin is Constexpr` checks no longer work because `Constexpr` is no longer a class.

New unified detection:

```python
# jit_arg_adapters.py
from typing import Annotated, get_origin, get_args

def is_arg_spec_constexpr(arg_spec, arg_name, arg_index, owning_func):
    # ... existing self/cls checks ...

    # Annotated-based Constexpr detection
    if get_origin(arg_spec) is Annotated:
        return any(isinstance(a, _ConstexprMeta) for a in get_args(arg_spec)[1:])
    return False
```

```python
# tree_utils.py
def is_constexpr_field(field: dataclasses.Field) -> bool:
    if get_origin(field.type) is Annotated:
        return any(isinstance(a, _ConstexprMeta) for a in get_args(field.type)[1:])
    return False
```

`_validate_arg()` delegates to `is_arg_spec_constexpr()`, so no separate change needed.

**Files changed:**

| File | Change |
|------|--------|
| `base_dsl/typing.py` | Replace `class Constexpr` with `Annotated[T, marker]` alias |
| `base_dsl/runtime/jit_arg_adapters.py` | Update `is_arg_spec_constexpr()` detection |
| `cutlass_dsl/tree_utils.py` | Update `is_constexpr_field()` detection |

**Pros:**
- Single definition — runtime and type checker see the same thing
- No divergent code paths to maintain
- Standard Python typing pattern (PEP 593)

**Cons:**
- Breaking change to JIT detection logic (3 sites)
- `Constexpr` is no longer a class — any code doing `isinstance(x, Constexpr)` or `issubclass(x, Constexpr)` at runtime breaks (grep confirms this is limited to the 3 detection sites above)

---

### Approach B: `TYPE_CHECKING` guard — isolated type-checker path

Keep the existing `class Constexpr` at runtime. Provide a separate definition visible only to type checkers.

**Definition:**

```python
# base_dsl/typing.py

from typing import TYPE_CHECKING, TypeVar, Generic

if TYPE_CHECKING:
    from typing import Annotated

    class _ConstexprMeta:
        pass

    _CONSTEXPR_MARKER = _ConstexprMeta()

    T = TypeVar("T")
    Constexpr = Annotated[T, _CONSTEXPR_MARKER]
else:
    TY = TypeVar("TY")

    class Constexpr(Generic[TY]):
        """Value is passed and computed by python interpreter"""
        pass
```

**Runtime behavior:**

Identical to today. `Constexpr` is still a class. `get_origin(Constexpr[int]) is Constexpr` still works. No JIT changes needed.

**Type checker behavior:**

Same as Approach A — type checkers see `Annotated[X, marker]`, treat `Constexpr[X]` as `X`.

**Required JIT detection changes:**

None.

**Files changed:**

| File | Change |
|------|--------|
| `base_dsl/typing.py` | Add `TYPE_CHECKING` guard around `Constexpr` definition |

**Pros:**
- Zero changes to JIT detection logic
- Zero risk of breaking existing runtime behavior
- Minimal diff

**Cons:**
- Two divergent definitions of `Constexpr` to maintain
- If runtime and type-checker definitions drift apart, subtle bugs may arise (e.g., a refactor updates one branch but not the other)
- Slightly unusual pattern — developers reading the code must understand the `TYPE_CHECKING` split

---

## Comparison

| | Approach A (`Annotated`) | Approach B (`TYPE_CHECKING`) |
|---|---|---|
| **Runtime `Constexpr`** | `Annotated[T, marker]` (type alias) | `class Constexpr(Generic[TY])` (unchanged) |
| **Type checker sees** | `Annotated[X, marker]` → treats as `X` | `Annotated[X, marker]` → treats as `X` |
| **JIT detection changes** | 3 sites (`jit_arg_adapters.py`, `tree_utils.py`, `cutlass.py`) | None |
| **Maintenance burden** | One definition | Two definitions (must stay in sync) |
| **Risk** | Moderate — detection logic changes | Low — no runtime changes |
| **Bare `Constexpr` (no param)** | Unresolved `T`, effectively `Any` | Unchanged (class, effectively `Any`) |

## Scope limitation

Both approaches only improve type checking for the parameterized form `Constexpr[X]`. The bare `Constexpr` annotation (the most common usage) remains opaque to type checkers under both proposals. Migrating existing bare `Constexpr` annotations to `Constexpr[ActualType]` is a separate effort.
