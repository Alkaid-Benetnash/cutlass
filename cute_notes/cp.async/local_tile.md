# `local_tile`

**Source:** `include/cute/tensor_impl.hpp:1037` (C++), `python/CuTeDSL/cutlass/cute/core.py:3553` (DSL)

## What it does

Extracts a tile from a tensor. Internally calls `inner_partition` (`tensor_impl.hpp:984`), which:

1. Calls [zipped_divide](../cute/layout_division.md#zipped_divide) on the tensor with the tiler, producing a rank-2 result: `((tile_modes), (rest_modes))`
2. Slices into the rest modes using `coord`, keeping the tile modes intact

## Signature (CuTe DSL)

```python
def local_tile(
    input: Tensor,
    tiler: Tiler,
    coord: Coord,
    proj: XTuple = None,
) -> Tensor
```

## Parameters

**`input`** — The tensor to tile. Shape like `(M, N)` or `(M, N, K)`.

**`tiler`** — Defines the tile shape. Each element divides the corresponding tensor mode. Can be a shape tuple (e.g., `(32, 64)`) or a layout. Must have `rank(tiler) <= rank(input)`. Modes beyond `rank(tiler)` are left untouched.

**`coord`** — Selects which tile from the rest (quotient) modes:
- Concrete coordinate (e.g., `(blockIdx_x, blockIdx_y)`) — fully selects one tile
- Coordinate with underscores/`None` (e.g., `(blockIdx_x, None)`) — keeps unsliced modes as extra dimensions. `None` means "keep all tiles along this mode"

**`proj`** (optional) — Projection tuple that filters which tiler/coord modes apply. Uses `dice(proj, tiler)` and `dice(proj, coord)` to strip modes marked with `None`/`X`. Lets you reuse the same tiler and coord across tensors with different mode subsets.

## Internal mechanism (`inner_partition`)

```cpp
// tensor_impl.hpp:984
auto tensor_tiled = zipped_divide(tensor, tiler);  // ((BLK_A,BLK_B,...),(a,b,...))
// Slice rest modes with coord, keep tile modes:
return tensor_tiled(repeat<R0>(_), append<R1>(coord, _));
```

## Example: GEMM CTA tiling with projection

```python
# Shared tiler for M, N, K
cta_tiler = (32, 64, 4)
cta_coord = (block_x, block_y, None)

# tensor_a is (M, K) — want modes 0 and 2 (M and K)
cta_a = local_tile(tensor_a, cta_tiler, cta_coord, proj=(1, None, 1))  # (32, 4, k)

# tensor_b is (N, K) — want modes 1 and 2 (N and K)
cta_b = local_tile(tensor_b, cta_tiler, cta_coord, proj=(None, 1, 1))  # (64, 4, k)

# tensor_c is (M, N) — want modes 0 and 1 (M and N)
cta_c = local_tile(tensor_c, cta_tiler, cta_coord, proj=(1, 1, None))  # (32, 64)
```

C++ reference: `tensor_impl.hpp:1048-1056`.

## Relationship to other APIs

- The output tensors are typically passed to [tma_partition](tma_partition.md) after [group_modes](../cute/layout_access.md#group_modes) collapses the tile dimensions into mode 0.
- The tiler shape must be consistent with the `cta_tiler` passed to [make_tiled_tma_atom](make_tiled_tma_atom.md).
