# Layout Algebra

Operations that compose, invert, or complete layouts.

## `composition`

**Source:** `layout.hpp:1136`, `core.py:3167`

Composes two layouts: `R(c) = A(B(c))`. The result has `B`'s domain shape and `A`'s stride structure baked in.

```python
# Signature
cute.composition(lhs, rhs)   # rhs can be Layout, Shape, or Tile

# Post-condition:
# cute.shape(cute.composition(A, B)) is compatible with cute.shape(B)
# For all c: composition(A, B)(c) == A(B(c))
```

### Common patterns

**Reshape (with_shape):** `composition(L, make_layout(new_shape))` reinterprets L's coordinate domain with a new shape. This is the C++ `L.with_shape(...)` pattern.

```python
cute.composition(L, cute.make_layout((4, 8192)))
# Reshapes L's domain to (4, 8192) with column-major linearization
```

**Per-mode composition:** When `rhs` is a tuple of layouts, composition applies per-mode:

```python
cute.composition(L, (layout_for_mode0, layout_for_mode1))
```

## `complement`

**Source:** `layout.hpp:1163`, `core.py:3222`

Finds a layout that covers the offsets NOT in `A`'s [image](../math/functions.md#domain-and-codomain), up to a given codomain target size.

```python
cute.complement(layout, cotarget)
```

Post-conditions:
- `A` and `complement(A, M)` have disjoint codomains
- `complement(A, M)` is ordered (strides are positive and increasing)
- `cosize(make_layout(A, complement(A, M))) >= size(M)`

Requires [injective](../math/functions.md#injective) input (asserts `"Non-injective Layout detected"` at `layout.hpp:1203`).

Used internally by [logical_product](layout_division.md#logical_product) and [logical_divide](layout_division.md#logical_divide).

## `right_inverse` / `left_inverse`

**Source:** `layout.hpp:1262,1324`, `core.py:3267,3275`

### `right_inverse(L)`

Produces a layout `R` such that `L(R(i)) == i` for all `i < size(R)`. Equivalently, `composition(L, R)` is an identity layout.

```python
R = cute.right_inverse(L)
# Post-condition: composition(L, R) == make_layout(shape(R))  (identity)
```

### `left_inverse(L)`

Produces a layout `L'` such that `L(L'(L(i))) == L(i)` for all `i < size(L)`. When `L` is [injective](../math/functions.md#injective), `L'` is a true left inverse: `L'(L(i)) == i`.

```python
Li = cute.left_inverse(L)
# Post-condition: composition(L, composition(Li, L)) == L
```

### Round-trip example

```python
mode1 = cute.make_layout(((256, 8), 4), stride=((8, 1), 2048))
ri = cute.right_inverse(mode1)   # (8,256,4):(256,1,2048)
cute.composition(mode1, ri)       # (8,256,4):(1,8,2048) — identity
cute.left_inverse(ri)             # (256,8,4):(8,1,2048) — recovers mode1's strides
```

Both require static strides. See [bijective / inverse functions](../math/functions.md#inverse-functions) for the mathematical background.

## `max_common_layout` / `max_common_vector`

**Source:** `layout.hpp:1384,1411`, `core.py:3463,3490`

### `max_common_layout(A, B)`

Returns a layout pointing to the maximum number of contiguous elements where `A` and `B` agree on mapping (both map those indices to `0, 1, 2, ...`).

```python
cute.max_common_layout(A, B)
# Post-condition: For all 0 <= i < size(R), A(R(i)) == i and B(R(i)) == i
```

Internally: `coalesce(composition(A, right_inverse(B)))`, then extract the leading stride-1 component.

### `max_common_vector(A, B)`

Same concept, returns just the count (an integer) rather than the layout.

```python
cute.max_common_vector(A, B)
```
