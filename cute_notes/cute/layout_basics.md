# Layout Basics

A CuTe `Layout` is a function from coordinates to offsets, defined by a pair of `IntTuple`s: `Shape` and `Stride`. The shape defines the coordinate space (domain); the stride defines how coordinates map to memory offsets (codomain).

## Fundamental Queries

**Source:** `include/cute/layout.hpp`, `python/CuTeDSL/cutlass/cute/core.py`

### `size`

The number of elements in the layout's **domain** (coordinate space). Equals `product(shape(L))`.

```python
cute.size(L)              # total domain size
cute.size(L, mode=[0])    # mode-0 size
cute.size(L, mode=[1])    # mode-1 size
```

### `cosize`

**Source:** `layout.hpp:657-666`, `core.py:2470`

The exclusive upper bound of the layout's **codomain**. Defined as `L(size(L) - 1) + 1` — the largest offset produced, plus one.

`cosize` is the minimum buffer size needed to hold all offsets the layout can produce. It is NOT the number of distinct offsets (the [image](../math/functions.md#domain-and-codomain) size).

```python
cute.cosize(L)
```

### `rank`

The number of top-level modes in the layout.

```python
cute.rank(L)
```

### `shape` and `stride`

Direct access to the layout's components. In CuTeDSL, these return **Python tuples** at trace time (not IR values), so they can be used in trace-time Python logic.

```python
L.shape   # e.g., ((1, 4), ((256, 8), 4))
L.stride  # e.g., ((0, 8192), ((8, 1), 2048))
```

## Size vs Cosize Relationship

The relationship between `size` and `cosize` characterizes the layout's function properties. See [mathematical definitions](../math/functions.md) for the underlying concepts.

| Condition | Meaning | Example |
|-----------|---------|---------|
| `cosize == size` | [Bijection](../math/functions.md#bijective) onto `[0, size-1]` | `8192:1` → size=8192, cosize=8192 |
| `cosize > size` | [Injective](../math/functions.md#injective) but not surjective — gaps in offset range | `4:2` → size=4, cosize=7, offsets={0,2,4,6} |
| `cosize < size` | Non-injective — stride-0 modes map multiple coordinates to the same offset | `(4,2):(1,0)` → size=8, cosize=4 |

**Confirmed examples (CuTeDSL):**

| Layout | size | cosize | Notes |
|--------|------|--------|-------|
| `((256,8),4):((8,1),2048)` | 8192 | 8192 | Bijection onto `[0, 8191]` |
| `8192:1` | 8192 | 8192 | Bijection onto `[0, 8191]` (contiguous) |
| `(1,4):(0,8192)` | 4 | 24577 | Non-injective (stride-0); offsets = {0, 8192, 16384, 24576} |
| `4:2` | 4 | 7 | Injective with gaps; offsets = {0, 2, 4, 6} |

### Same-codomain detection

Two [injective](../math/functions.md#injective) layouts with `size(A) == cosize(A)` and `size(B) == cosize(B)` and `size(A) == size(B)` are both [bijections](../math/functions.md#bijective) onto `[0, N-1]` — they have identical images. They differ only in ordering (different permutations of the same set).

CuTe has **no dedicated predicate** for this. Check manually:

```python
mode1 = cute.make_layout(((256, 8), 4), stride=((8, 1), 2048))
simple = cute.make_layout(8192)
# cute.size(mode1) == cute.cosize(mode1) == 8192  ✓
# cute.size(simple) == cute.cosize(simple) == 8192  ✓
# → both are bijections onto [0, 8191]
```

## Domain Structure Concepts

**Source:** `include/cute/int_tuple.hpp:436-509`

CuTe defines named concepts for comparing domain (shape) structures. These test **domain structure only** — they say nothing about strides or codomain.

### `congruent(A, B)`

Two shapes have the same hierarchical nesting profile (same tree structure, ignoring values).

```
(2, (3, 4))  congruent with  (5, (6, 7))   ✓  — same tree shape
(2, (3, 4))  congruent with  (5, 6, 7)     ✗  — different nesting
```

### `weakly_congruent(A, B)`

A's profile fits into B's (partial order: A ≤ B). A scalar is weakly congruent with any shape.

```
4             weakly_congruent with  (3, 4)    ✓  — scalar fits any
(3, 4)        weakly_congruent with  4          ✗  — tuple doesn't fit scalar
```

### `compatible(A, B)`

Same size at each terminal of A's shape; any coordinate of A can also be used as a coordinate of B.

```
(4, 8)        compatible with  (4, (2, 4))   ✓  — sizes match at each leaf
(4, 8)        compatible with  (4, 6)         ✗  — mode-1 sizes differ
```
