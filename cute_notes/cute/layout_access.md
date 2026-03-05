# Layout Access and Mode Manipulation

APIs for extracting, reordering, and extending layout modes.

## `get`

**Source:** `core.py:1663`, `layout.hpp:554`

Extracts a sub-layout by mode index.

```python
cute.get(L, mode=[0])    # extract mode 0
cute.get(L, mode=[1])    # extract mode 1
cute.get(L, mode=[1, 0]) # hierarchical: mode 0 within mode 1
```

## `select`

**Source:** `core.py:1729`, `layout.hpp:518`

Selects and reorders a subset of top-level modes.

```python
cute.select(L, mode=[0, 2])   # keep modes 0 and 2
cute.select(L, mode=[1, 0])   # swap mode 0 and mode 1
```

C++ equivalent: `select<Is...>(layout)`.

## `group_modes`

**Source:** `core.py:1795`

Groups a contiguous range of top-level modes `[begin, end)` into a single hierarchical mode.

```python
cute.group_modes(layout, begin, end)

layout = make_layout((2, 3, 4, 5))
group_modes(layout, 1, 3)  # → (2, (3,4), 5):(1, (2,6), 24)
```

Used extensively to prepare tensors for [tma_partition](../cp.async/tma_partition.md) by collapsing tile dimensions into mode 0.

## `append` / `prepend`

**Source:** `core.py:2126,2071`, `layout.hpp:964,984`

Add a mode to the end or beginning of a layout, shape, or tuple:

```python
layout = make_layout((8, 8))
cute.append(layout, make_layout(1))    # → (8,8,1):(1,8,0)
cute.prepend(layout, make_layout(1))   # → (1,8,8):(0,1,8)
```

With `up_to_rank`, pads by repeating the element:

```python
cute.append(layout, make_layout(1), up_to_rank=5)  # → (8,8,1,1,1):(1,8,0,0,0)
```

## `append_ones` / `prepend_ones`

**Source:** `core.py:2196,2173`

Convenience wrappers that automatically use `make_layout(1)` as the padding element. Work on both `Layout` and `Tensor`:

```python
cute.append_ones(layout, up_to_rank=4)  # → (8,8,1,1):(1,8,0,0)
```
