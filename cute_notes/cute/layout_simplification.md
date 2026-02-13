# Layout Simplification

Function-preserving operations that simplify a layout's structure without changing its coordinate-to-offset mapping.

## `flatten`

**Source:** `layout.hpp:531`, `core.py:2303`

Removes hierarchy, producing a flat layout where every sub-mode becomes a top-level mode. Does not merge or reorder modes.

```python
cute.flatten(L)
# ((1,4),((256,8),4)):((0,8192),((8,1),2048))
# → (1,4,256,8,4):(0,8192,8,1,2048)
```

## `coalesce`

**Source:** `layout.hpp:867`, `core.py:2902`

Flattens, drops shape-1 modes, and merges adjacent modes whose strides are contiguous (`shape[i] * stride[i] == stride[i+1]`). Does **not** reorder modes.

```python
cute.coalesce(L)
# ((1,4),((256,8),4)):((0,8192),((8,1),2048))
# → (4,256,8,4):(8192,8,1,2048)
```

With `target_profile`, coalesces within each mode independently:

```python
cute.coalesce(L, target_profile=(0, 0))
# → (4,(256,8,4)):(8192,(8,1,2048))
# Mode 0 simplified to 4:8192; mode 1 stays (256,8,4):(8,1,2048)
```

**Limitation:** `coalesce` cannot merge modes whose strides are non-monotonic. For example, strides `(8, 1, 2048)` have no adjacent pair where `shape * stride == next_stride`, so no merging occurs. This is by design — `coalesce` preserves the function exactly.

## `filter`

**Source:** `layout.hpp:168` (pycute), `core.py:2385`

Replaces stride-0 modes with size-1, then [coalesces](#coalesce) to remove them. The result is always [injective](../math/functions.md#injective) — no stride-0 modes remain.

```python
cute.filter(L)
# ((1,4),((256,8),4)):((0,8192),((8,1),2048))
# → (4,256,8,4):(8192,8,1,2048)
# (same as coalesce for this layout since the stride-0 mode has size 1)
```

**Injectivity test:** If `size(filter(L)) == size(L)`, the layout is [injective](../math/functions.md#injective). If `size(filter(L)) < size(L)`, the layout is non-injective (stride-0 modes collapsed some coordinates).

## `filter_zeros`

**Source:** `core.py:2350`

Removes stride-0 modes only, without the subsequent [coalesce](#coalesce) step. Less aggressive than [filter](#filter).

```python
cute.filter_zeros(L)
cute.filter_zeros(L, target_profile=stride_profile)  # apply per-mode
```
