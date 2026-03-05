# Tuple Utilities

Helper functions for working with CuTe's `IntTuple` and `Shape` types.

## `product_each`

**Source:** `python/CuTeDSL/cutlass/cute/tuple.py:154`

Collapses each top-level mode of a (possibly hierarchical) tuple to its product:

```python
product_each(((4, 8), (16, 1), 8))  # → (32, 16, 8)
product_each(((2, 3), (4, 5)))      # → (6, 20)
```

Contract: `get(result, i) == product(get(a, i))`. Operates on `IntTuple`/`Shape`, not `Layout` objects directly.

## `find_if`

**Source:** `python/CuTeDSL/cutlass/cute/tuple.py:185`

General-purpose recursive search over tuples. Takes a predicate `pred_fn(value, position)` and returns the position of the first match, or `None`:

```python
stride = (4, 1)
find_if(stride, pred_fn=lambda val, pos: val == 1)  # → 1
```

Recurses into nested tuples, returning nested position tuples for hierarchical matches.

## `is_major` / `leading_dim`

**Source:** `core.py:1459` / `core.py:3624`

### `is_major(mode, stride)`

Returns `True` if the front element of the stride at `mode` is 1:

```python
is_major(0, (4, 1))  # False
is_major(1, (4, 1))  # True
```

### `leading_dim(shape, stride)`

Finds which mode is major (stride 1, shape not 1). Returns the mode index, a nested tuple for hierarchical layouts, or `None`:

```python
leading_dim((4, 8), (8, 1))             # → 1
leading_dim((4, 8), (1, 4))             # → 0
leading_dim(((2,3), 4), ((4,1), 12))    # → (0, 1)
```

Uses [find_if](#find_if) internally.
