# `tma_partition`

**Source:** `include/cute/atom/copy_traits_sm90_tma.hpp:1398` (C++), `python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:179` (DSL)

## What it does

Partitions SMEM and GMEM tensors for use with a TMA copy atom. Subdivides the TMA tile (mode 0) into `(TMA_instruction, TMA_iterations)` and applies a multicast offset.

## Signature (CuTe DSL)

```python
def tma_partition(
    atom: CopyAtom,
    cta_coord: Coord,
    cta_layout: Layout,
    smem_tensor: Tensor,
    gmem_tensor: Tensor,
) -> Tuple[Tensor, Tensor]
```

## Parameters

**`atom`** — A TMA copy atom from [make_tiled_tma_atom](make_tiled_tma_atom.md). Provides `NumValSrc` (elements per single TMA instruction).

**`cta_coord`** — The current CTA's coordinate within the cluster along the **multicast dimension**. This is a scalar, not the full cluster coordinate:
- For operand A (multicast across N): `cluster_coord_mnk[1]`
- For operand B (multicast across M): `cluster_coord_mnk[0]`
- For non-multicast (e.g., TMA store): `0`

**`cta_layout`** — Maps CTA coordinate to a logical multicast ID. The slice of the cluster layout along the multicast dimension, recompacted:
- For operand A: `make_layout(slice_(cta_layout_mnk, (0, None, 0)).shape)`
- For operand B: `make_layout(slice_(cta_layout_mnk, (None, 0, 0)).shape)`
- For non-multicast: `make_layout(1)`

**`smem_tensor`** — Shared memory tensor with shape `(TMATile, Rest...)`. Mode 0 must be the TMA tile. Typically prepared with [group_modes](../cute/layout_access.md#group_modes):
```python
# sA shape: (bM, bK, STAGES)
# group_modes(sA, 0, 2) → ((bM, bK), STAGES)
```

**`gmem_tensor`** — Global memory tensor with shape `(TMATile, Rest...)`. Mode 0 must match SMEM mode 0 in [size](../cute/layout_basics.md#size): `size<0>(smem_tensor) == size<0>(gmem_tensor)` (enforced at `copy_traits_sm90_tma.hpp:1404`). Typically prepared with [group_modes](../cute/layout_access.md#group_modes):
```python
# gA_mkl shape: (bM, bK, RestM, RestK, RestL)
# group_modes(gA_mkl, 0, 2) → ((bM, bK), RestM, RestK, RestL)
```

## The TMA tile concept

The "TMA tile" is **not** a parameter to `tma_partition`. It is the **mode-0 shape** already baked into both input tensors. It represents the region of data copied in one logical TMA round, corresponding to the CTA's tile of the tensor.

The TMA tile shape is established upstream by [make_tiled_tma_atom](make_tiled_tma_atom.md) via the intersection of `gmem_tensor`, `smem_layout`, and `cta_tiler`. The returned TMA tensor and the SMEM layout both encode this tile shape.

Preparing tensors for `tma_partition` requires:
1. Using [local_tile](local_tile.md) to tile the GMEM tensor into `(bM, bK, Rest...)`
2. Using [group_modes](../cute/layout_access.md#group_modes) to collapse tile dimensions into mode 0: `((bM, bK), Rest...)`

## Internal mechanism (`copy_traits_sm90_tma.hpp:1406-1428`)

1. **Compute vector layout:** [right_inverse](../cute/layout_algebra.md#right_inverse--left_inverse)`(get_nonswizzle_portion(layout<0>(stensor)))` finds the largest contiguous vector in SMEM, then `tile_to_shape` extends it to cover the full tile
2. **Factor out TMA instruction:** [logical_divide](../cute/layout_division.md#logical_divide)`(layout_v, tma_layout_v)` where `tma_layout_v` has shape `NumValSrc` — splits into `(TMA, TMA_Iter)`
3. **Compose** into both tensors: `tensor.compose(layout_V)` → `((TMA, TMA_Iter), Rest...)`
4. **Apply multicast offset:** `cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout))` offsets within the TMA tile for this CTA's multicast position

## Returns

`(smem_result, gmem_result)` — partitioned tensors with shape `((TMA, TMA_Iter), Rest...)`, ready for `cute.copy`:

```python
cute.copy(tma_atom, tAgA[((None, None), k_tile, 0)],
                    tAsA[((None, None), stage)])
```

## Full pipeline example (Hopper GEMM)

Reference: `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py:655-696`

```python
# 1. make_tiled_tma_atom (host/setup)
tma_atom_a, tma_tensor_a = make_tiled_tma_atom(G2SMulticastOp, gmem_A, smem_layout_a, cta_tiler_mk, cluster_n)

# 2. local_tile (in kernel) — tile the TMA tensor
gA_mkl = local_tile(tma_tensor_a, slice_(tile_shape_mnk, (None, 0, None)), (None, None, None))
# shape: (bM, bK, RestM, RestK, RestL)

# 3. tma_partition — subdivide into TMA instructions
a_cta_layout = make_layout(slice_(cta_layout_mnk, (0, None, 0)).shape)
tAsA, tAgA = tma_partition(
    tma_atom_a,
    cluster_coord_mnk[1],       # multicast along N
    a_cta_layout,
    group_modes(sA, 0, 2),      # ((bM,bK), STAGES)
    group_modes(gA_mkl, 0, 2),  # ((bM,bK), RestM, RestK, RestL)
)
# tAsA: ((TMA, TMA_Iter), STAGES)
# tAgA: ((TMA, TMA_Iter), RestM, RestK, RestL)
```
