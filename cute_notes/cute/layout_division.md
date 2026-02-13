# Layout Division and Tiling

Operations that divide layouts into tile and remainder components, used extensively for CTA tiling and thread partitioning.

## `logical_divide`

**Source:** `layout.hpp:1559`, `core.py:3404`

Splits a layout `A` into two modes using a tiler `B`:

```
logical_divide(A, B) = composition(A, (B, complement(B, size(A))))
```

First mode contains elements selected by `B`; second mode contains the "rest" (via [complement](layout_algebra.md#complement)).

```python
cute.logical_divide(layout, tiler)
```

## `zipped_divide`

**Source:** `layout.hpp:1610`, `core.py:3419`

Applies [logical_divide](#logical_divide) per-mode, then reorganizes via `tile_unzip` into `((tile_modes), (rest_modes))`:

```
logical_divide result: ((tile0, rest0), (tile1, rest1))
tile_unzip gathers:    ((tile0, tile1), (rest0, rest1))
```

```python
cute.zipped_divide(layout, tiler)
```

Used internally by [local_tile](../cp.async/local_tile.md) via `inner_partition`.

## `tiled_divide`

**Source:** `layout.hpp:1625`, `core.py:3434`

Like [zipped_divide](#zipped_divide) but keeps the per-mode grouping rather than zipping. Each mode becomes `(tile_i, rest_i)`.

```python
cute.tiled_divide(layout, tiler)
```

## `flat_divide`

**Source:** `layout.hpp:1641`, `core.py:3449`

Like [zipped_divide](#zipped_divide) but flattens the result instead of zipping.

```python
cute.flat_divide(layout, tiler)
```

## `logical_product`

**Source:** `layout.hpp:1653`, `core.py:3291`

### Single-mode form (both args are Layouts)

Produces a rank-2 layout `(block, complement ∘ tiler)`:

```cpp
// layout.hpp:1656
logical_product(block, tiler) = make_layout(block, composition(complement(block, size(block)*cosize(tiler)), tiler));
```

### Per-mode form (tiler is a tuple)

Applies `logical_product` independently to each mode. Increases depth but preserves top-level rank:

```python
block = make_layout((1, (3, 4)))
tiler = (4, 4)
result = logical_product(block, tiler)
# Mode 0: logical_product(1, 4) → (1, 4)
# Mode 1: logical_product((3,4), 4) → ((3,4), 4)
# Result: ((1,4), ((3,4),4))
```

### Difference from [blocked_product](#blocked_product)

Per-mode `logical_product` computes [complement](layout_algebra.md#complement) independently for each mode. `blocked_product` computes a single complement over the entire flattened block, then zips — so tiler strides account for the full block size across all modes.

## `blocked_product`

**Source:** `layout.hpp:1734`

Pads both layouts to the same rank, calls [logical_product](#logical_product) on the whole (not per-mode), then zips block and tiler modes together:

```cpp
// layout.hpp:1737-1741
constexpr int R = max(rank(block), rank(tiler));
auto result = logical_product(append<R>(block), append<R>(tiler));
return zip(get<0>(result), get<1>(result));
```

Result has `rank = max(rank(block), rank(tiler))`, each mode is `(block_i, tiler_i)`.
