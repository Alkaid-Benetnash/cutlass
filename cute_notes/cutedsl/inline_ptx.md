# Inline PTX in CuTeDSL

Emit PTX assembly directly from CuTeDSL, bypassing MLIR/LLVM IR optimization.

## API

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

llvm.inline_asm(
    result_type,       # MLIR type of the result, or None for void
    inputs,            # list of ir.Value operands
    asm_string,        # PTX assembly string
    constraints,       # operand constraint string
    has_side_effects,  # True if the asm has side effects (stores, barriers, etc.)
    is_align_stack,    # typically False
    asm_dialect=llvm.AsmDialect.AD_ATT,  # always AD_ATT for PTX
)
```

Must be called inside a `@dsl_user_op` function.

## Parameters

### `result_type`

The MLIR type of the output value. Use `None` for void (no output).

| Return | `result_type` |
|--------|---------------|
| 32-bit int | `T.i32()` |
| 64-bit int | `T.i64()` |
| f32 | `T.f32()` |
| void | `None` |
| multiple outputs | `llvm.StructType.get_literal([T.f32(), T.f32()])` |

CuTeDSL maps both `Int32` and `Uint32` to signless `T.i32()` at the MLIR level (`base_dsl/typing.py:1621`). Signedness is tracked at the Python wrapper level only. Use `T.i32()` for all 32-bit integer results regardless of signedness.

### `inputs`

List of `ir.Value` operands, obtained via `.ir_value(loc=loc, ip=ip)` on DSL scalars:

```python
[cutlass.Uint32(val).ir_value(loc=loc, ip=ip), cutlass.Uint32(shift).ir_value(loc=loc, ip=ip)]
```

### `asm_string`

PTX assembly text. Operands are referenced as `$0`, `$1`, `$2`, ... numbered across outputs then inputs. For example, with one output and two inputs: `$0` = output, `$1` = first input, `$2` = second input.

```python
"shl.b32 $0, $1, $2;"           # single instruction
```

For multi-instruction blocks, use PTX scoped register declarations:

```python
"{\n\t"
".reg .s32 r1, r2;\n\t"         # declare temporary registers
"shl.b32 r1, $1, 23;\n\t"
"add.s32 $0, r1, $2;\n\t"
"}\n"
```

### `constraints`

LLVM-style operand constraint string. Output constraints are prefixed with `=`, separated from input constraints by `,`.

| Constraint | Meaning |
|------------|---------|
| `r` | 32-bit integer register |
| `l` | 64-bit integer register |
| `f` | 32-bit float register |
| `=r` | 32-bit integer output |
| `=f` | 32-bit float output |

Examples:
- `"=r,r,r"` — one i32 output, two i32 inputs
- `"=f,f,f"` — one f32 output, two f32 inputs
- `"r,r"` — two i32 inputs, no output (void)
- `"=r,=r,f,f"` — two i32 outputs, two f32 inputs (multi-output with struct return)

### `has_side_effects`

- `True` for stores, atomics, barriers, printf — anything with observable effects beyond the return value
- `False` for pure computations (shifts, arithmetic, conversions)

When `False`, the compiler may eliminate the asm if the result is unused.

## Wrapping the Result

Wrap the raw `ir.Value` returned by `llvm.inline_asm` in the appropriate DSL type:

```python
return cutlass.Uint32(llvm.inline_asm(T.i32(), ...))
return cutlass.Float32(llvm.inline_asm(T.f32(), ...))
```

For multi-output (struct return), extract each field with `llvm.extractvalue`:

```python
out = llvm.inline_asm(llvm.StructType.get_literal([T.f32(), T.f32()]), ...)
a = Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
b = Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
```

## Complete Example

```python
@dsl_user_op
def shl_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shl.b32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
```

## Why Inline PTX

CuTeDSL compiles through MLIR → LLVM IR. LLVM IR inherits C/C++ semantics where shifting by >= the type width is undefined behavior (the result is poison). The LLVM optimizer exploits this to eliminate code paths it can prove involve such shifts.

PTX has no such restriction — "Shift amounts greater than the register width N are clamped to N" (PTX ISA §9.7.8.8). Inline PTX bypasses LLVM IR entirely: the instruction is emitted verbatim into PTX, where the hardware-defined clamping semantics apply.

## Source References

- `llvm.inline_asm` dialect: `cutlass/_mlir/dialects/llvm.py`
- MLIR type constructors: `cutlass/_mlir/extras/types.py`
- Upstream examples: `cutlass/cute/arch/nvvm_wrappers.py` (barrier_sync, vote_sync, exp2, log2_of_pow2_int, etc.)
