# `make_tiled_tma_atom`

**Source:** `include/cute/atom/copy_traits_sm90_tma.hpp:1133,1188` (C++), `python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:44` (DSL)

## What it does

Creates a TMA copy atom and an associated "TMA tensor" for copying tiles between GMEM and SMEM. The atom encodes a TMA descriptor (tensor map) that the hardware TMA unit uses. The TMA tensor repartitions the GMEM tensor so its layout outputs multi-dimensional coordinates (via basis strides) rather than flat offsets, since TMA hardware operates on coordinates.

## Signature (CuTe DSL)

```python
def make_tiled_tma_atom(
    op: Union[CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp,
              CopyBulkTensorTileS2GOp, CopyReduceBulkTensorTileS2GOp],
    gmem_tensor: Tensor,
    smem_layout: Union[Layout, ComposedLayout],
    cta_tiler: Tiler,
    num_multicast: int = 1,
    *,
    internal_type: Optional[Type[Numeric]] = None,
) -> Tuple[CopyAtom, Tensor]
```

## Parameters

**`op`** — The TMA copy operation type. Determines direction and features:
- `CopyBulkTensorTileG2SOp` — Global-to-shared load
- `CopyBulkTensorTileG2SMulticastOp` — Global-to-shared load with cluster multicast
- `CopyBulkTensorTileS2GOp` — Shared-to-global store
- `CopyReduceBulkTensorTileS2GOp` — Shared-to-global store with reduction

**`gmem_tensor`** — The full global memory tensor (e.g., shape `(M, K)`). Used to determine the overall tensor shape/strides for the TMA descriptor.

**`smem_layout`** — The SMEM buffer layout for one tile (e.g., shape `(bM, bK)`), possibly with swizzling. Determines the SMEM access pattern and the maximum contiguous vector the TMA can use. Must satisfy `size(smem_layout) == size(cta_v_map)` (i.e., the SMEM tile covers the same number of elements as the CTA tile).

**`cta_tiler`** — How the CTA tiles the GMEM tensor (e.g., `(bM, bK)`). Internally converted to a `cta_v_map` via [composition](../cute/layout_algebra.md#composition)`(make_identity_layout(gmem_tensor.shape), cta_tiler)` (`helpers.py:103-108`), which maps each element within the CTA tile to its GMEM mode/coordinate.

**`num_multicast`** — The number of CTAs participating in multicast (cluster size along the multicast dimension). Default `1` for non-multicast.

**`internal_type`** (optional) — Override the element type used internally by the TMA unit. Useful when the actual data type isn't directly supported by TMA hardware.

## Returns

A tuple of:

1. **`CopyAtom`** — The TMA copy atom, encoding the TMA descriptor and instruction shape. The atom's `NumValSrc` is `num_bits_per_tma / sizeof_bits<element_type>`, where `num_bits_per_tma = size(tma_gbasis) * sizeof_bits<TmaInternalType>` (`copy_traits_sm90_tma.hpp:1161`).

2. **`Tensor`** (TMA tensor) — The GMEM tensor repartitioned with basis strides. This is used in place of the original GMEM tensor for [local_tile](local_tile.md) and subsequently [tma_partition](tma_partition.md).

## Internal mechanism

1. Computes `cta_v_map = composition(identity_layout(gmem_shape), cta_tiler)` — maps CTA-tile-local indices to GMEM coordinates
2. Inverts the SMEM layout to find the largest contiguous access vector ([right_inverse](../cute/layout_algebra.md#right_inverse--left_inverse)`(get_nonswizzle_portion(slayout))`)
3. Composes the inverted SMEM layout with `cta_v_map` to get `sidx2gmode` — the mapping from SMEM index to GMEM mode
4. Truncates at the first non-unit basis stride (TMA requires contiguous major-mode access)
5. Coalesces modes up to the TMA box extent limit of 256 elements (per TMA dimension)
6. Builds the TMA descriptor (`CUtensorMap`) with the derived shapes/strides
7. Constructs the TMA tensor with basis strides for coordinate output

## Key constraints

- The SMEM layout must select contiguous major GMEM modes (assertion at `copy_traits_sm90_tma.hpp:774`)
- TMA descriptors are limited to rank 5 — excess modes are grouped (`copy_traits_sm90_tma.hpp:834`)
- `size(smem_layout) == size(cta_v_map)` is required (`copy_traits_sm90_tma.hpp:739-740`)

## Example

```python
from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileG2SMulticastOp, make_tiled_tma_atom

# gmem_A shape: (M, K)
# smem_layout_a shape: (bM, bK), possibly with swizzle
# cta_tiler_mk: (bM, bK)
tma_atom_a, tma_tensor_a = make_tiled_tma_atom(
    CopyBulkTensorTileG2SMulticastOp,
    gmem_A,
    smem_layout_a,
    cta_tiler_mk,
    num_multicast=cluster_n,
)
```

The returned `tma_tensor_a` replaces `gmem_A` for subsequent tiling via [local_tile](local_tile.md). The returned `tma_atom_a` is passed to [tma_partition](tma_partition.md) and ultimately to `cute.copy`.
