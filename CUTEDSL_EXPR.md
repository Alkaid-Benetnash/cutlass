# CuTeDSL Expression Testing Guide

## Running Layout Algebra Expressions

CuTeDSL layout algebra operations execute at **compile/trace time** inside `@cute.jit` functions.
The `print()` statements inside `@cute.jit` run during tracing and show layout results,
even if the final CUDA kernel launch fails (e.g., due to driver issues).

### Minimal Template

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def test():
    L = cute.make_layout((4, 8), stride=(8, 1))
    print(f"L = {L}")
    # Use str(layout) or f-string to see layout representation

if __name__ == "__main__":
    test()
```

The CUDA launch error at the end (`cudaErrorInsufficientDriver`) can be ignored —
all `print()` output from the trace phase has already been emitted.

### Available Layout Algebra APIs

All of these work inside `@cute.jit`:

```python
# Construction
L = cute.make_layout(shape, stride=stride)
L = cute.make_layout(shape)                    # default col-major strides
L = cute.make_ordered_layout(shape, order=order)
L = cute.make_layout_like(L)                   # compact layout, same stride ordering

# Queries
cute.size(L)                    # total domain size
cute.size(L, mode=[0])          # mode-0 size
cute.size(L, mode=[1])          # mode-1 size
cute.cosize(L)                  # codomain upper bound: L(size-1) + 1
cute.rank(L)                    # number of top-level modes
L.shape                         # shape tuple (accessible at trace time)
L.stride                        # stride tuple (accessible at trace time)

# Function-preserving simplification
cute.flatten(L)                              # remove hierarchy, keep all modes
cute.coalesce(L)                             # flatten + merge contiguous adjacent modes
cute.coalesce(L, target_profile=(0, 0))      # coalesce within each mode independently
cute.filter(L)                               # remove stride-0 modes + coalesce
cute.filter_zeros(L)                         # remove stride-0 modes only

# Layout algebra
cute.composition(L, rhs)         # R(c) = L(rhs(c)); rhs can be Layout, Shape, or Tile
cute.complement(L, cotarget)     # layout of elements NOT in L's image
cute.right_inverse(L)            # R such that L(R(i)) == i
cute.left_inverse(L)             # L' such that L(L'(L(i))) == L(i)

# Tiling / division
cute.logical_divide(L, tiler)
cute.zipped_divide(L, tiler)
cute.tiled_divide(L, tiler)
cute.flat_divide(L, tiler)
cute.logical_product(L, tiler)

# Other
cute.max_common_layout(A, B)     # max contiguous prefix where A and B agree
cute.max_common_vector(A, B)     # same, returns count
```

### Accessing Sub-modes and Properties

```python
# Extract sub-layouts
mode0 = cute.get(L, mode=[0])    # extract mode 0 as a sub-layout
mode1 = cute.get(L, mode=[1])    # extract mode 1

# Reorder modes
swapped = cute.select(L, mode=[1, 0])  # swap mode 0 and mode 1
```

Confirmed: `cute.get(L, mode=[0])` returns the sub-layout (e.g., `(1,4):(0,8192)`).
`cute.select(L, mode=[1,0])` reorders modes.

### Property Access at Trace Time

```python
L.shape   # returns Python tuple, e.g. ((1, 4), ((256, 8), 4))
L.stride  # returns Python tuple, e.g. ((0, 8192), ((8, 1), 2048))
```

`L.shape` and `L.stride` return **Python tuples** (not IR values) — usable in
trace-time Python logic (conditionals, loops, tuple manipulation).

### Printing Tips

- `print(f"{L}")` calls `__str__` on the layout, producing e.g. `(4,8):(8,1)`
- `str(L)` works the same way
- For tensors: `print(f"{tensor.type}")` shows the type including layout
- Print statements execute at trace/compile time, not at GPU runtime

### Observed Behaviors (Confirmed by Testing)

Given `L = ((1,4),((256,8),4)):((0,8192),((8,1),2048))`:

| Operation | Result |
|-----------|--------|
| `coalesce(L)` | `(4,256,8,4):(8192,8,1,2048)` |
| `coalesce(L, target_profile=(0,0))` | `(4,(256,8,4)):(8192,(8,1,2048))` |
| `flatten(L)` | `(1,4,256,8,4):(0,8192,8,1,2048)` |
| `filter(L)` | `(4,256,8,4):(8192,8,1,2048)` |
| `composition(L, make_layout((4,8192)))` | `(4,(256,8,4)):(8192,(8,1,2048))` |
| `composition(L, make_layout((4,8192),(8192,1)))` | `(4,(4,256,8)):(2048,(8192,8,1))` |
| `make_layout_like(L)` | `((1,4),((256,8),4)):((0,8192),((8,1),2048))` (unchanged) |
| `right_inverse(mode1)` | `(8,256,4):(256,1,2048)` |
| `left_inverse(mode1)` | `(8,256,4):(256,1,2048)` |
| `composition(mode1, right_inverse(mode1))` | `(8,256,4):(1,8,2048)` (identity) |

Where `mode1 = ((256,8),4):((8,1),2048)`.

### Key Insight: coalesce Cannot Simplify Non-monotonic Strides

`coalesce` merges adjacent modes where `prev_shape * prev_stride == next_stride`.
When strides are not monotonically ordered (e.g., `(8, 1, 2048)`), adjacent pairs
fail this test and cannot be merged. This is by design — `coalesce` preserves
the layout function exactly.

### Bijection Detection via size/cosize

`cosize(L)` is the codomain upper bound: `L(size(L) - 1) + 1`.
For an injective layout with non-negative strides, `size(L) == cosize(L)` means
L is a **bijection** onto `[0, size-1]`.

Confirmed by testing:

| Layout | size | cosize | Bijection? |
|--------|------|--------|-----------|
| `((256,8),4):((8,1),2048)` (mode1) | 8192 | 8192 | Yes |
| `8192:1` (simple) | 8192 | 8192 | Yes |
| `(1,4):(0,8192)` (mode0, has stride-0) | 4 | 24577 | No |
| `4:2` (strided) | 4 | 7 | No |

Two injective layouts with the same `size == cosize` are both bijections onto
the same set `[0, N-1]`, meaning they have identical codomains (images).
CuTe has no dedicated predicate for this; check manually with `size` and `cosize`.
