# CuTeDSL Experiment Guide

How to run CuTeDSL experiments, inspect generated PTX, and debug kernels.

## Environment Setup

```bash
source ~/py3env/bin/activate
```

## Running Experiments

### Basic Run

```bash
python my_experiment.py
```

### Dumping PTX and CUBIN

Set environment variables to keep intermediate artifacts:

```bash
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=/absolute/path/to/dump \
    python my_experiment.py
```

- `CUTE_DSL_KEEP_PTX=1` — retain generated PTX files
- `CUTE_DSL_KEEP_CUBIN=1` — retain compiled CUBIN files
- `CUTE_DSL_DUMP_DIR` — directory for dumped files (use **absolute path**; relative paths may not resolve correctly)

Output files are named after the kernel signature, e.g.:
`cutlass_launch_Tensorgmemo1_Tensorgmemo1_1.sm_100a.ptx`

Cache is automatically disabled when dump is enabled (warns: "Cache is disabled as user wants to generate PTX/ASM").

### Disassembling CUBIN to SASS

To inspect the actual GPU assembly (SASS) from a compiled CUBIN:

```bash
nvdisasm my_kernel.sm_100a.cubin > my_kernel.sass
```

## Trace Time vs Runtime

CuTeDSL has two execution phases:

| Phase | When | Print mechanism | Runs on |
|-------|------|----------------|---------|
| Trace time | During `@cute.jit` / `@cute.kernel` compilation | Python `print()` | CPU |
| Runtime | During kernel execution on GPU | `cute.printf()` | GPU |

- **Trace time**: Layout algebra, `Constexpr` values, control flow over constexpr conditions — all resolved here. Python `print()` statements inside `@cute.jit` execute during tracing and show results even if the kernel launch later fails.
- **Runtime**: Dynamic values (`Int32`, `Uint32`, tensor elements) only exist at runtime. Use `cute.printf()` to inspect them.

```python
@cute.kernel
def my_kernel(out: cute.Tensor, flag: Int32):
    # Runtime printing (GPU) — shows dynamic values
    tidx = cute.arch.thread_idx()[0]
    if tidx == 0:
        cute.printf("flag=%d\n", flag)
```

## Kernel Templates

### Minimal Kernel with `cute.compile`

Use `cute.compile` when you need to control which arguments are dynamic vs baked-in:

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Uint32

@cute.kernel
def my_kernel(
    output: cute.Tensor,
    dynamic_arg: Int32,
    constexpr_arg: cutlass.Constexpr[int],
):
    # constexpr_arg is baked into the kernel at compile time
    # dynamic_arg is passed at runtime
    ...

@cute.jit
def launch(
    output: cute.Tensor,
    dynamic_arg: Int32,
    constexpr_arg: cutlass.Constexpr[int],
):
    my_kernel(output, dynamic_arg, constexpr_arg).launch(
        grid=[1, 1, 1],
        block=[32, 1, 1],
    )

# Compile once per constexpr value
out = torch.ones(32, device="cuda", dtype=torch.float32)
compiled = cute.compile(
    launch,
    from_dlpack(out).mark_layout_dynamic(),
    -1,             # dynamic_arg placeholder (value doesn't matter at compile time)
    42,             # constexpr_arg (baked in)
)

# Run — only pass dynamic arguments (constexpr args are already baked in)
compiled(
    from_dlpack(out).mark_layout_dynamic(),
    -1,             # dynamic_arg actual value
)
torch.cuda.synchronize()
```

Key points:
- `cutlass.Constexpr[int]` arguments are baked into the compiled kernel — a separate kernel is compiled per distinct constexpr value
- Dynamic scalar arguments (`Int32`, `Uint32`, etc.) are passed at runtime
- `cute.compile` returns a callable that takes only the dynamic arguments
- Tensor arguments use `from_dlpack(tensor).mark_layout_dynamic()` for dynamic shapes

### Trace-Time Layout Exploration

For exploring layout algebra without a GPU kernel:

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def test():
    L = cute.make_layout((4, 8), stride=(8, 1))
    print(f"L = {L}")
    print(f"shape = {L.shape}, stride = {L.stride}")

if __name__ == "__main__":
    test()
```

The CUDA launch error at the end (`cudaErrorInsufficientDriver`) can be ignored —
all `print()` output from the trace phase has already been emitted.
