# cute_notes/

Notes on CuTe and CuTeDSL concepts, organized by topic.

## Math Foundations

- [functions.md](math/functions.md) — Injective, surjective, bijective, domain, codomain, image

## CuTe Layout Concepts

- [layout_basics.md](cute/layout_basics.md) — Layout as a function, size, cosize, rank, domain structure concepts (congruent, compatible)
- [layout_simplification.md](cute/layout_simplification.md) — coalesce, flatten, filter, filter_zeros
- [layout_algebra.md](cute/layout_algebra.md) — composition, complement, right_inverse, left_inverse
- [layout_construction.md](cute/layout_construction.md) — make_layout, make_layout_like, make_ordered_layout
- [layout_division.md](cute/layout_division.md) — logical_divide, zipped_divide, tiled_divide, flat_divide, logical_product, blocked_product
- [layout_access.md](cute/layout_access.md) — get, select, group_modes, append, prepend
- [tuple_utils.md](cute/tuple_utils.md) — product_each, find_if, is_major, leading_dim

## TMA / cp.async

- [local_tile.md](cp.async/local_tile.md) — Extracting tiles from tensors
- [make_tiled_tma_atom.md](cp.async/make_tiled_tma_atom.md) — Creating TMA copy atoms and TMA tensors
- [tma_partition.md](cp.async/tma_partition.md) — Partitioning tensors for TMA copy

## File Maintenance Rules

- **One section per API.** Each CuTe/CuTe DSL API gets its own section or file.
- **Cross-reference** related APIs using relative links (e.g., `[coalesce](../cute/layout_simplification.md#coalesce)`) rather than duplicating explanations.
- **Include source references** (e.g., `file.hpp:123` or `module.py:456`) for key implementation details.
- **Prefer upstream evidence.** Use code and comments from the CUTLASS/CuTe source repository. Only use conversation-generated code when no equivalent exists upstream.
- **Keep examples minimal.** Show the API signature, what it does, parameter requirements, and one or two concrete examples.
