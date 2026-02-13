# Mathematical Foundations for Layout Concepts

These are standard mathematical definitions, not CuTe-specific. CuTe uses these terms in its source code and documentation with their standard meanings.

## Domain and Codomain

A **function** `f: A → B` maps elements from set A (the **domain**) to elements in set B (the **codomain**).

- **Domain**: The set of all valid inputs.
- **Codomain**: The set that outputs are drawn from (the declared output space).
- **Image** (or **range**): The set of outputs actually produced — a subset of the codomain.

In CuTe, a layout is a function from coordinates (domain) to memory offsets (codomain). The domain is defined by the shape; the codomain is the set of non-negative integers. The image is the specific set of offsets the layout actually maps to.

## Injective

A function `f` is **injective** (one-to-one) if distinct inputs always produce distinct outputs:

```
For all x, y in domain:  f(x) = f(y)  implies  x = y
```

No two different inputs map to the same output.

**CuTe context:** A layout is injective when no two coordinates map to the same memory offset. A layout with stride-0 modes is non-injective (e.g., `(4,2):(1,0)` maps `(0,0)` and `(0,1)` both to offset 0). See [filter](../cute/layout_simplification.md#filter) for the practical tool to detect/remove non-injective modes.

## Surjective

A function `f: A → B` is **surjective** (onto) if every element of the codomain is hit by at least one input:

```
For all y in codomain:  there exists x in domain such that f(x) = y
```

**CuTe context:** For a layout with non-negative strides, surjectivity onto `[0, cosize-1]` means every offset in that range is produced. A layout like `4:2` (producing `{0, 2, 4, 6}`) is NOT surjective onto `[0, 6]` because offsets 1, 3, 5 are never hit. See [cosize](../cute/layout_basics.md#cosize) for the codomain bound.

## Bijective

A function is **bijective** if it is both [injective](#injective) and [surjective](#surjective) — a perfect one-to-one correspondence between domain and codomain.

**CuTe context:** A layout is a bijection onto `[0, N-1]` when it is injective (no stride-0 modes) and `size == cosize == N`. Two bijective layouts with the same N cover the same set of offsets but in different orders — they are different permutations of `[0, N-1]`. See [size vs cosize](../cute/layout_basics.md#size-vs-cosize-relationship) for the detection method.

## Inverse Functions

For a [bijective](#bijective) function `f: A → B`:

- **Right inverse** `g`: satisfies `f(g(y)) = y` for all y. Given an output, `g` finds the input that produces it.
- **Left inverse** `h`: satisfies `h(f(x)) = x` for all x. After applying `f`, `h` recovers the original input.

For a bijection, the right inverse and left inverse are the same function (the inverse `f⁻¹`).

For non-bijective functions, only quasi-inverses may exist. See [right_inverse / left_inverse](../cute/layout_algebra.md#right_inverse--left_inverse) for CuTe's implementations.
