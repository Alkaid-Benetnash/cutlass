# Layout Construction

APIs for creating layouts.

## `make_layout`

**Source:** `layout.hpp:376`, `core.py:2554`

Creates a layout from shape and optional stride.

```python
cute.make_layout(shape)                      # default column-major strides
cute.make_layout(shape, stride=stride)       # explicit strides
cute.make_layout((4, 8))                     # (4,8):(1,4)  column-major
cute.make_layout((4, 8), stride=(8, 1))      # (4,8):(8,1)  row-major
```

## `make_layout_like`

**Source:** `layout.hpp:441`, `core.py:2698`

Creates a compact layout with the same shape, where strides follow the ordering induced by the original layout's strides. Preserves stride-0 modes.

```python
cute.make_layout_like(L)
```

Does **not** change the shape or permute modes — it recomputes strides to be compact while preserving the relative stride ordering.

## `make_ordered_layout`

**Source:** `layout.hpp:425`, `core.py:2655`

Creates a compact layout with strides assigned according to an explicit `order` tuple. Smaller order value = faster-varying stride (innermost dimension):

```python
cute.make_ordered_layout((4, 8, 2), order=(2, 0, 1))
# → (4,8,2):(128,1,16)  — mode 1 fastest, then mode 2, then mode 0
```

Neither `make_layout_like` nor `make_ordered_layout` provides a standalone "sort modes by stride" operation. CuTe's `SortByKey` (`layout.hpp:1281`) is used internally by [right_inverse / left_inverse](layout_algebra.md#right_inverse--left_inverse) but is not exposed as a public API.
