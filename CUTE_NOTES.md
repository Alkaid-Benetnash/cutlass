# CUTE_NOTES.md

## File Maintenance Rules

- **One section per API.** Each CuTe/CuTe DSL API gets its own `##` section.
- **Cross-reference** related APIs using `[link text](#section-name)` anchors rather than duplicating explanations.
- **Include source references** (e.g., `file.hpp:123` or `module.py:456`) for key implementation details.
- **Prefer upstream evidence.** Use code and comments from the CUTLASS/CuTe source repository. Only use conversation-generated code when no equivalent exists upstream.
- **Keep examples minimal.** Show the API signature, what it does, parameter requirements, and one or two concrete examples.

---

## `local_tile`

**Source:** `include/cute/tensor_impl.hpp:1037` (C++), `python/CuTeDSL/cutlass/cute/core.py:3553` (DSL)

### What it does

Extracts a tile from a tensor. Internally calls `inner_partition` (`tensor_impl.hpp:984`), which:

1. Calls [`zipped_divide`](#zipped_divide) on the tensor with the tiler, producing a rank-2 result: `((tile_modes), (rest_modes))`
2. Slices into the rest modes using `coord`, keeping the tile modes intact

### Signature (CuTe DSL)

```python
def local_tile(
    input: Tensor,
    tiler: Tiler,
    coord: Coord,
    proj: XTuple = None,
) -> Tensor
```

### Parameters

**`input`** — The tensor to tile. Shape like `(M, N)` or `(M, N, K)`.

**`tiler`** — Defines the tile shape. Each element divides the corresponding tensor mode. Can be a shape tuple (e.g., `(32, 64)`) or a layout. Must have `rank(tiler) <= rank(input)`. Modes beyond `rank(tiler)` are left untouched.

**`coord`** — Selects which tile from the rest (quotient) modes:
- Concrete coordinate (e.g., `(blockIdx_x, blockIdx_y)`) — fully selects one tile
- Coordinate with underscores/`None` (e.g., `(blockIdx_x, None)`) — keeps unsliced modes as extra dimensions. `None` means "keep all tiles along this mode"

**`proj`** (optional) — Projection tuple that filters which tiler/coord modes apply. Uses `dice(proj, tiler)` and `dice(proj, coord)` to strip modes marked with `None`/`X`. Lets you reuse the same tiler and coord across tensors with different mode subsets.

### Internal mechanism (`inner_partition`)

```cpp
// tensor_impl.hpp:984
auto tensor_tiled = zipped_divide(tensor, tiler);  // ((BLK_A,BLK_B,...),(a,b,...))
// Slice rest modes with coord, keep tile modes:
return tensor_tiled(repeat<R0>(_), append<R1>(coord, _));
```

### Example: GEMM CTA tiling with projection

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

### Relationship to other APIs

- The output tensors are typically passed to [`tma_partition`](#tma_partition) after [`group_modes`](#group_modes) collapses the tile dimensions into mode 0.
- The tiler shape must be consistent with the `cta_tiler` passed to [`make_tiled_tma_atom`](#make_tiled_tma_atom).

---

## `make_tiled_tma_atom`

**Source:** `include/cute/atom/copy_traits_sm90_tma.hpp:1133,1188` (C++), `python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:44` (DSL)

### What it does

Creates a TMA copy atom and an associated "TMA tensor" for copying tiles between GMEM and SMEM. The atom encodes a TMA descriptor (tensor map) that the hardware TMA unit uses. The TMA tensor repartitions the GMEM tensor so its layout outputs multi-dimensional coordinates (via basis strides) rather than flat offsets, since TMA hardware operates on coordinates.

### Signature (CuTe DSL)

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

### Parameters

**`op`** — The TMA copy operation type. Determines direction and features:
- `CopyBulkTensorTileG2SOp` — Global-to-shared load
- `CopyBulkTensorTileG2SMulticastOp` — Global-to-shared load with cluster multicast
- `CopyBulkTensorTileS2GOp` — Shared-to-global store
- `CopyReduceBulkTensorTileS2GOp` — Shared-to-global store with reduction

**`gmem_tensor`** — The full global memory tensor (e.g., shape `(M, K)`). Used to determine the overall tensor shape/strides for the TMA descriptor.

**`smem_layout`** — The SMEM buffer layout for one tile (e.g., shape `(bM, bK)`), possibly with swizzling. Determines the SMEM access pattern and the maximum contiguous vector the TMA can use. Must satisfy `size(smem_layout) == size(cta_v_map)` (i.e., the SMEM tile covers the same number of elements as the CTA tile).

**`cta_tiler`** — How the CTA tiles the GMEM tensor (e.g., `(bM, bK)`). Internally converted to a `cta_v_map` via `composition(make_identity_layout(gmem_tensor.shape), cta_tiler)` (`helpers.py:103-108`), which maps each element within the CTA tile to its GMEM mode/coordinate.

**`num_multicast`** — The number of CTAs participating in multicast (cluster size along the multicast dimension). Default `1` for non-multicast.

**`internal_type`** (optional) — Override the element type used internally by the TMA unit. Useful when the actual data type isn't directly supported by TMA hardware.

### Returns

A tuple of:

1. **`CopyAtom`** — The TMA copy atom, encoding the TMA descriptor and instruction shape. The atom's `NumValSrc` is `num_bits_per_tma / sizeof_bits<element_type>`, where `num_bits_per_tma = size(tma_gbasis) * sizeof_bits<TmaInternalType>` (`copy_traits_sm90_tma.hpp:1161`).

2. **`Tensor`** (TMA tensor) — The GMEM tensor repartitioned with basis strides. This is used in place of the original GMEM tensor for [`local_tile`](#local_tile) and subsequently [`tma_partition`](#tma_partition).

### Internal mechanism

1. Computes `cta_v_map = composition(identity_layout(gmem_shape), cta_tiler)` — maps CTA-tile-local indices to GMEM coordinates
2. Inverts the SMEM layout to find the largest contiguous access vector (`right_inverse(get_nonswizzle_portion(slayout))`)
3. Composes the inverted SMEM layout with `cta_v_map` to get `sidx2gmode` — the mapping from SMEM index to GMEM mode
4. Truncates at the first non-unit basis stride (TMA requires contiguous major-mode access)
5. Coalesces modes up to the TMA box extent limit of 256 elements (per TMA dimension)
6. Builds the TMA descriptor (`CUtensorMap`) with the derived shapes/strides
7. Constructs the TMA tensor with basis strides for coordinate output

### Key constraints

- The SMEM layout must select contiguous major GMEM modes (assertion at `copy_traits_sm90_tma.hpp:774`)
- TMA descriptors are limited to rank 5 — excess modes are grouped (`copy_traits_sm90_tma.hpp:834`)
- `size(smem_layout) == size(cta_v_map)` is required (`copy_traits_sm90_tma.hpp:739-740`)

### Example

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

The returned `tma_tensor_a` replaces `gmem_A` for subsequent tiling via [`local_tile`](#local_tile). The returned `tma_atom_a` is passed to [`tma_partition`](#tma_partition) and ultimately to `cute.copy`.

---

## `tma_partition`

**Source:** `include/cute/atom/copy_traits_sm90_tma.hpp:1398` (C++), `python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:179` (DSL)

### What it does

Partitions SMEM and GMEM tensors for use with a TMA copy atom. Subdivides the TMA tile (mode 0) into `(TMA_instruction, TMA_iterations)` and applies a multicast offset.

### Signature (CuTe DSL)

```python
def tma_partition(
    atom: CopyAtom,
    cta_coord: Coord,
    cta_layout: Layout,
    smem_tensor: Tensor,
    gmem_tensor: Tensor,
) -> Tuple[Tensor, Tensor]
```

### Parameters

**`atom`** — A TMA copy atom from [`make_tiled_tma_atom`](#make_tiled_tma_atom). Provides `NumValSrc` (elements per single TMA instruction).

**`cta_coord`** — The current CTA's coordinate within the cluster along the **multicast dimension**. This is a scalar, not the full cluster coordinate:
- For operand A (multicast across N): `cluster_coord_mnk[1]`
- For operand B (multicast across M): `cluster_coord_mnk[0]`
- For non-multicast (e.g., TMA store): `0`

**`cta_layout`** — Maps CTA coordinate to a logical multicast ID. The slice of the cluster layout along the multicast dimension, recompacted:
- For operand A: `make_layout(slice_(cta_layout_mnk, (0, None, 0)).shape)`
- For operand B: `make_layout(slice_(cta_layout_mnk, (None, 0, 0)).shape)`
- For non-multicast: `make_layout(1)`

**`smem_tensor`** — Shared memory tensor with shape `(TMATile, Rest...)`. Mode 0 must be the TMA tile. Typically prepared with [`group_modes`](#group_modes):
```python
# sA shape: (bM, bK, STAGES)
# group_modes(sA, 0, 2) → ((bM, bK), STAGES)
```

**`gmem_tensor`** — Global memory tensor with shape `(TMATile, Rest...)`. Mode 0 must match SMEM mode 0 in size: `size<0>(smem_tensor) == size<0>(gmem_tensor)` (enforced at `copy_traits_sm90_tma.hpp:1404`). Typically prepared with [`group_modes`](#group_modes):
```python
# gA_mkl shape: (bM, bK, RestM, RestK, RestL)
# group_modes(gA_mkl, 0, 2) → ((bM, bK), RestM, RestK, RestL)
```

### The TMA tile concept

The "TMA tile" is **not** a parameter to `tma_partition`. It is the **mode-0 shape** already baked into both input tensors. It represents the region of data copied in one logical TMA round, corresponding to the CTA's tile of the tensor.

The TMA tile shape is established upstream by [`make_tiled_tma_atom`](#make_tiled_tma_atom) via the intersection of `gmem_tensor`, `smem_layout`, and `cta_tiler`. The returned TMA tensor and the SMEM layout both encode this tile shape.

Preparing tensors for `tma_partition` requires:
1. Using [`local_tile`](#local_tile) to tile the GMEM tensor into `(bM, bK, Rest...)`
2. Using [`group_modes`](#group_modes) to collapse tile dimensions into mode 0: `((bM, bK), Rest...)`

### Internal mechanism (`copy_traits_sm90_tma.hpp:1406-1428`)

1. **Compute vector layout:** `right_inverse(get_nonswizzle_portion(layout<0>(stensor)))` finds the largest contiguous vector in SMEM, then `tile_to_shape` extends it to cover the full tile
2. **Factor out TMA instruction:** `logical_divide(layout_v, tma_layout_v)` where `tma_layout_v` has shape `NumValSrc` — splits into `(TMA, TMA_Iter)`
3. **Compose** into both tensors: `tensor.compose(layout_V)` → `((TMA, TMA_Iter), Rest...)`
4. **Apply multicast offset:** `cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout))` offsets within the TMA tile for this CTA's multicast position

### Returns

`(smem_result, gmem_result)` — partitioned tensors with shape `((TMA, TMA_Iter), Rest...)`, ready for `cute.copy`:

```python
cute.copy(tma_atom, tAgA[((None, None), k_tile, 0)],
                    tAsA[((None, None), stage)])
```

### Full pipeline example (Hopper GEMM)

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

---

## `group_modes`

**Source:** `python/CuTeDSL/cutlass/cute/core.py:1795`

Groups a contiguous range of top-level modes `[begin, end)` into a single hierarchical mode.

```python
def group_modes(input, begin: int, end: int = None) -> same_type_as_input
```

```python
layout = make_layout((2, 3, 4, 5))
group_modes(layout, 1, 3)  # → (2, (3,4), 5):(1, (2,6), 24)
```

Used extensively to prepare tensors for [`tma_partition`](#tma_partition) by collapsing tile dimensions into mode 0.

---

## `select`

**Source:** `python/CuTeDSL/cutlass/cute/core.py:1729`, `include/cute/layout.hpp:518`

Selects a subset of top-level modes by index, returning a new layout/tuple with only those modes.

```python
def select(input, mode: List[int]) -> same_type_as_input
```

```python
layout = make_layout((4, 8, 16), stride=(32, 4, 1))
select(layout, mode=[0, 2])  # → (4, 16):(32, 1)
```

C++ equivalent: `select<Is...>(layout)`.

---

## `logical_product`

**Source:** `include/cute/layout.hpp:1653`, `python/CuTeDSL/cutlass/cute/core.py:3291`

### Single-mode form (both args are Layouts)

Produces a rank-2 layout `(block, complement ∘ tiler)`:

```cpp
// layout.hpp:1656
logical_product(block, tiler) = make_layout(block, composition(complement(block, size(block)*cosize(tiler)), tiler));
```

### Per-mode form (tiler is a tuple)

Applies `logical_product` independently to each mode via `transform_layout` (`core.py:3307`, `layout.hpp:1667`). Increases depth but preserves top-level rank:

```python
block = make_layout((1, (3, 4)))
tiler = (4, 4)
result = logical_product(block, tiler)
# Mode 0: logical_product(layout<1>, 4) → (1, 4)
# Mode 1: logical_product(layout<(3,4)>, 4) → ((3,4), 4)
# Result: ((1,4), ((3,4),4))
```

### Difference from [`blocked_product`](#blocked_product)

Per-mode `logical_product` computes `complement` independently for each mode. [`blocked_product`](#blocked_product) computes a single `complement` over the entire flattened block, then zips — so tiler strides account for the full block size across all modes.

---

## `blocked_product`

**Source:** `include/cute/layout.hpp:1734`

Takes two Layout objects, pads both to the same rank, calls `logical_product` on the whole (not per-mode), then zips block and tiler modes together:

```cpp
// layout.hpp:1737-1741
constexpr int R = max(rank(block), rank(tiler));
auto result = logical_product(append<R>(block), append<R>(tiler));
return zip(get<0>(result), get<1>(result));
```

Result has `rank = max(rank(block), rank(tiler))`, each mode is `(block_i, tiler_i)`. See [`logical_product`](#logical_product) for comparison.

---

## `zipped_divide`

**Source:** `include/cute/layout.hpp:1610`, `python/CuTeDSL/cutlass/cute/core.py:3413`

Defined as `tile_unzip(logical_divide(layout, tiler), tiler)`. Divides each mode by the tiler and reorganizes into `((tile_modes), (rest_modes))`.

```
logical_divide result: ((tile0, rest0), (tile1, rest1))
tile_unzip gathers:    ((tile0, tile1), (rest0, rest1))
```

Used internally by [`local_tile`](#local_tile) via `inner_partition`.

---

## `product_each`

**Source:** `python/CuTeDSL/cutlass/cute/tuple.py:154`

Collapses each top-level mode of a (possibly hierarchical) tuple to its product:

```python
product_each(((4, 8), (16, 1), 8))  # → (32, 16, 8)
product_each(((2, 3), (4, 5)))      # → (6, 20)
```

Contract: `get(result, i) == product(get(a, i))`. Operates on `IntTuple`/`Shape`, not `Layout` objects directly.

---

## `append` / `prepend`

**Source:** `python/CuTeDSL/cutlass/cute/core.py:2126` / `core.py:2071`, `include/cute/layout.hpp:964` / `layout.hpp:984`

Add a mode to the end or beginning of a layout, shape, or tuple:

```python
layout = make_layout((8, 8))
append(layout, make_layout(1))   # → (8,8,1):(1,8,0)
prepend(layout, make_layout(1))  # → (1,8,8):(0,1,8)
```

With `up_to_rank`, pads by repeating the element:

```python
append(layout, make_layout(1), up_to_rank=5)  # → (8,8,1,1,1):(1,8,0,0,0)
```

---

## `append_ones` / `prepend_ones`

**Source:** `python/CuTeDSL/cutlass/cute/core.py:2196` / `core.py:2173`

Convenience wrappers around [`append`/`prepend`](#append--prepend) that automatically use `make_layout(1)` as the element. Work on both `Layout` and `Tensor`:

```python
layout = make_layout((8, 8))
append_ones(layout, up_to_rank=4)  # → (8,8,1,1):(1,8,0,0)
```

---

## `is_major` / `leading_dim`

**Source:** `python/CuTeDSL/cutlass/cute/core.py:1459` / `core.py:3624`

**`is_major(mode, stride)`** — Returns `True` if the front element of the stride at `mode` is 1:

```python
is_major(0, (4, 1))  # False
is_major(1, (4, 1))  # True
```

**`leading_dim(shape, stride)`** — Finds which mode is major (stride 1, shape not 1). Returns the mode index, a nested tuple for hierarchical layouts, or `None`:

```python
leading_dim((4, 8), (8, 1))             # → 1
leading_dim((4, 8), (1, 4))             # → 0
leading_dim(((2,3), 4), ((4,1), 12))    # → (0, 1)
```

Uses [`find_if`](#find_if) internally.

---

## `find_if`

**Source:** `python/CuTeDSL/cutlass/cute/tuple.py:185`

General-purpose recursive search over tuples. Takes a predicate `pred_fn(value, position)` and returns the position of the first match, or `None`:

```python
stride = (4, 1)
find_if(stride, pred_fn=lambda val, pos: val == 1)  # → 1
```

Recurses into nested tuples, returning nested position tuples for hierarchical matches.

---

## Layout Domain and Codomain Concepts

CuTe layouts are functions from coordinates (domain) to offsets (codomain). Several concepts describe properties of these functions, using standard mathematical terminology.

### `size` and `cosize`

**Source:** `include/cute/layout.hpp:602,657`, `python/CuTeDSL/cutlass/cute/core.py:2416,2470`

- **`size(L)`** — The number of elements in the layout's **domain** (coordinate space). Equals `product(shape(L))`.
- **`cosize(L)`** — The exclusive upper bound of the layout's **codomain**. Defined as `L(size(L) - 1) + 1` (the largest offset produced, plus one). This is the minimum buffer size needed to hold all offsets the layout can produce.

`cosize` is NOT the number of distinct offsets (the image size). It is the span of the codomain.

**`size` vs `cosize` relationship:**

| Condition | Meaning |
|-----------|---------|
| `cosize == size` | Bijection onto `[0, size-1]` — every offset in range is hit exactly once |
| `cosize > size` | Injective but not surjective — gaps in the offset range (e.g., stride > 1) |
| `cosize < size` | Non-injective — stride-0 modes cause multiple coordinates to map to the same offset |

**Examples (confirmed in CuTeDSL):**

| Layout | size | cosize | Notes |
|--------|------|--------|-------|
| `((256,8),4):((8,1),2048)` | 8192 | 8192 | Bijection onto `[0, 8191]` |
| `8192:1` | 8192 | 8192 | Bijection onto `[0, 8191]` (contiguous) |
| `(1,4):(0,8192)` | 4 | 24577 | Non-injective (stride-0); offsets span `{0, 8192, 16384, 24576}` |
| `4:2` | 4 | 7 | Injective but gaps; offsets = `{0, 2, 4, 6}` |

### Mathematical Properties of Layouts

CuTe uses standard mathematical terms in its source and documentation:

| Term | Definition | CuTe indicator |
|------|-----------|---------------|
| **Injective** (one-to-one) | Distinct inputs → distinct outputs | No stride-0 modes; `filter(L)` has same size as `L` |
| **Surjective** (onto) | Every codomain element is hit | For injective layouts: `size == cosize` |
| **Bijective** (one-to-one and onto) | Injective + surjective | Injective AND `size == cosize` |

These are standard math concepts, not CuTe-specific abstractions. CuTe references them in:
- `complement` (`layout.hpp:1203`): asserts `"Non-injective Layout detected"` — complement requires injectivity
- `left_inverse` (`layout.hpp:1313`): `"left-inverse when layout is injective"` — true left-inverse requires injectivity
- `domain_distribute` (`layout.hpp:1433`): postcondition mentions `"Surjective and Ordered"`

**Practical tool for injectivity:** `filter(L)` removes stride-0 modes and coalesces. If `size(filter(L)) == size(L)`, the layout is injective. If `size(filter(L)) < size(L)`, the layout is non-injective (stride-0 modes collapsed some coordinates).

### Same-Codomain (Same Image)

Two injective layouts with `size(A) == cosize(A)` and `size(B) == cosize(B)` and `size(A) == size(B)` are both bijections onto `[0, N-1]` — they have **identical codomains** (same set of offsets). They differ only in the ordering (they are different permutations of the same set).

CuTe has **no dedicated predicate** for this property. Check manually:

```python
# Both are bijections onto [0, 8191]:
mode1 = cute.make_layout(((256, 8), 4), stride=((8, 1), 2048))
simple = cute.make_layout(8192)
# cute.size(mode1) == cute.cosize(mode1) == 8192  ✓
# cute.size(simple) == cute.cosize(simple) == 8192  ✓
```

### Domain Structure Concepts

CuTe DOES have named concepts for domain (shape) relationships. These are defined in `include/cute/int_tuple.hpp:436-509`:

| Concept | Definition | Example |
|---------|-----------|---------|
| **`congruent(A, B)`** | Same hierarchical nesting profile | `(2,(3,4))` congruent with `(5,(6,7))` — same tree shape |
| **`weakly_congruent(A, B)`** | A's profile fits into B's (partial order) | `4` weakly congruent with `(3,4)` — scalar fits any |
| **`compatible(A, B)`** | Same size at each terminal of A; A's coordinates work for B | `(4, 8)` compatible with `(4, (2,4))` — sizes match at each leaf |

These test **domain structure only** — they say nothing about strides or codomain.

---

## `coalesce`

**Source:** `include/cute/layout.hpp:867`, `python/CuTeDSL/cutlass/cute/core.py:2902`

Simplifies a layout by flattening hierarchy, dropping shape-1 modes, and merging adjacent modes with contiguous strides (`shape[i] * stride[i] == stride[i+1]`). Does **not** reorder modes.

```python
coalesce(layout)  # Simplify but preserve mode order
```

With a `target_profile`, applies coalescing at the terminals matching the profile structure.

---

## `make_layout_like` / `make_ordered_layout`

**Source:** `include/cute/layout.hpp:441` / `layout.hpp:425`, `python/CuTeDSL/cutlass/cute/core.py:2698` / `core.py:2655`

**`make_layout_like(layout)`** — Creates a new compact layout with the same shape, strides following the ordering induced by the original strides. Preserves stride-0 modes. Does not permute modes.

**`make_ordered_layout(shape, order)`** — Creates a compact layout with strides assigned according to `order` (smaller order value = faster-varying stride):

```python
make_ordered_layout((4, 8, 2), order=(2, 0, 1))
# → (4,8,2):(128,1,16)  — mode 1 fastest, then mode 2, then mode 0
```

Neither function permutes modes into stride-sorted order — CuTe does not expose a standalone "sort by stride" operation. See the note on `SortByKey` in `layout.hpp:1281` (used internally by `right_inverse`/`left_inverse` but not exposed).
