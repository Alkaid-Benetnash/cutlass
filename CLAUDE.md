# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Local Git Repository

**This is a local git repository clone of CUTLASS.** All source code is available locally on disk. When searching for code, implementations, or examples:

- **Use basic Linux tools**: `grep`, `egrep`, `rg` (ripgrep), and `find` via Bash
- **Do NOT use meta code search agents or MCP search tools** - use direct file system searches instead
- All files can be read directly using the Read tool once located

### Recommended Search Commands

```bash
# Search for a pattern in all files
grep -r "pattern" include/cute/

# Search with context lines
grep -rn -A5 -B5 "tma_partition" .

# Search specific file types
grep -r --include="*.hpp" "elect_one" include/

# Find files by name
find . -name "*.py" -path "*/CuTeDSL/*"

# Case-insensitive search
grep -ri "mbarrier" python/CuTeDSL/
```

## Maintaining This File

When updating CLAUDE.md:
1. **Always use example code and evidence from the CUTLASS/CuTe/CuTeDSL source repository** when possible. Search for existing implementations, comments, and patterns in the codebase.
2. **Only use code generated during conversation** when no equivalent or similar code exists in the upstream repository.
3. **If unsure whether upstream examples exist**, explicitly ask the user for judgment before adding generated code to CLAUDE.md.
4. **Include source file references** (e.g., `include/cute/arch/cluster_sm90.hpp:180`) when citing evidence from the codebase.

## Repository Overview

CUTLASS is NVIDIA's CUDA Templates for Linear Algebra Subroutines - a header-only C++ template library for high-performance matrix multiplication (GEMM) and related computations. It supports architectures from Volta (SM70) through Blackwell (SM100/SM120).

CUTLASS 4.x adds **CuTe DSL**, a Python-native interface for writing high-performance CUDA kernels without C++ expertise.

## Build Commands

```bash
# Set CUDA compiler
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

# Configure (specify target architecture to reduce compile time)
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80        # Ampere
cmake .. -DCUTLASS_NVCC_ARCHS=90a       # Hopper (use 'a' suffix for arch-specific features)
cmake .. -DCUTLASS_NVCC_ARCHS=100a      # Blackwell datacenter
cmake .. -DCUTLASS_NVCC_ARCHS=120a      # Blackwell GeForce RTX 50 series

# Build all unit tests
make test_unit -j

# Build specific test targets
make cutlass_test_unit_gemm -j
make cutlass_test_unit_conv -j
make cutlass_test_unit_cute -j

# Build profiler
make cutlass_profiler -j16

# Build specific examples
make 00_basic_gemm -j

# Build subset of kernels (faster builds)
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*
```

## Running Tests

```bash
# Run all unit tests via CTest
cd build && ctest

# Run specific test binary directly
./test/unit/gemm/cutlass_test_unit_gemm

# Run with GTest filter
./test/unit/gemm/cutlass_test_unit_gemm --gtest_filter=*SM80*

# Test levels (0=Sanity, 1=Release, 2=Exhaustive)
cmake .. -DCUTLASS_TEST_LEVEL=1
```

## Python Interface

```bash
# Install CUTLASS Python package
pip install .                    # or pip install -e . for development
pip install nvidia-cutlass       # from PyPI

# Install cutlass_library only
python python/setup_library.py develop --user

# Run CuTe DSL examples
python examples/python/CuTeDSL/ampere/elementwise_apply.py
```

## Architecture

### Core Libraries (Header-Only)

**include/cutlass/** - Main CUTLASS abstractions:
- `gemm/` - GEMM kernel implementations and policies
- `conv/` - Convolution implementations (implicit GEMM)
- `epilogue/` - Epilogue operations (bias, activation, scaling)
- `arch/` - Architecture-specific features and instruction wrappers
- `layout/` - Memory layout definitions (RowMajor, ColumnMajor, TensorNHWC)
- `transform/` - Data transformation utilities
- `pipeline/` - Async pipeline abstractions for Hopper+

**include/cute/** - CuTe (CUDA Templates) library:
- `layout.hpp`, `tensor.hpp` - Core abstractions for hierarchical layouts
- `atom/` - MMA and Copy atoms (hardware instruction wrappers)
- `algorithm/` - Copy, GEMM operations on CuTe tensors
- `arch/` - PTX wrapper structs

### CUTLASS Version Hierarchy

- **CUTLASS 2.x API**: `cutlass::gemm::device::Gemm` - Older, explicit template parameters
- **CUTLASS 3.x API**: Uses CuTe, `cutlass::gemm::collective::CollectiveMma` - More composable
- Hopper (SM90) and Blackwell (SM100) kernels primarily use 3.x patterns

### Tools

- `tools/profiler/` - `cutlass_profiler` command-line benchmarking tool
- `tools/library/` - Kernel instantiation and code generation
- `tools/util/` - Host/device utilities for tensor management

### CuTe DSL

- **`python/CuTeDSL/`** - The CuTe DSL language runtime implementation
- **`examples/python/CuTeDSL/`** - Example programs written in CuTe DSL (organized by architecture: `ampere/`, `hopper/`, `blackwell/`, `notebooks/`)

### Python Packages (`python/`)

- `cutlass_cppgen/` - Python interface for compiling/running CUTLASS kernels
- `cutlass_library/` - Kernel enumeration and C++ code emission
- `pycute/` - Python bindings for CuTe concepts

### Test Organization (`test/unit/`)

Tests mirror the library structure: `gemm/`, `conv/`, `cute/`, `epilogue/`, `layout/`, `transform/`, `pipeline/`

## Key Patterns

### Template Structure
CUTLASS uses deeply nested C++ templates. Key template parameters:
- Element types (input A/B, output C/D, accumulator)
- Layouts (RowMajor, ColumnMajor)
- Tile sizes (CTA tile, warp tile, instruction tile)
- Pipeline stages
- Epilogue operations

### Architecture Targeting
- Use `sm_90a`, `sm_100a` (with 'a' suffix) for Hopper/Blackwell-specific features like WGMMA, TMA
- Architecture-specific code in `arch/`, with `sm90_`, `sm100_` prefixes
- Feature detection via compute capability checks

### CuTe Concepts
- `Layout`: Maps logical coordinates to memory offsets
- `Tensor`: Data pointer + Layout
- `TiledMma` / `TiledCopy`: Hardware-aware tiled operations
- Composition via `make_tiled_copy()`, `partition_*()` functions

### TMA Operations and Thread Hierarchy

**TMA `cp.async.bulk.tensor` is NOT a warp-collective instruction.** It should be issued by a **single elected thread**, not all threads in a warp.

#### Evidence from CUTLASS:
```cpp
// From sm90_epilogue_tma_warpspecialized.hpp:475-476
// Predication for TMA load (one thread issues TMA load)
bool issue_tma_load = cute::elect_one_sync();

// From sm90_epilogue_array_tma_warpspecialized.hpp:795-796
// Predication for TMA store (a single thread from one warp issues TMA store)
bool issue_tma_store = ((thread_idx / NumThreadsPerWarp) == 0) && cute::elect_one_sync();
```

#### Correct TMA Usage Pattern (C++):
```cpp
// Single thread from entire thread block issues TMA
bool issue_tma = (warp_idx == 0) && cute::elect_one_sync();
if (issue_tma) {
    copy(tma_atom.with(*barrier, mcast_mask), src, dst);
}
```

#### Why Single Thread?
1. **TMA is hardware-accelerated**: The TMA unit operates independently; only one thread initiates the transfer
2. **Barrier coordination**: The mbarrier tracks transaction bytes; multiple threads issuing the same TMA would cause incorrect byte counts
3. **Efficiency**: Avoids redundant work from multiple threads

#### Thread Election APIs (C++):
- `cute::elect_one_sync()` - Warp-level election, returns `true` for one thread (see `include/cute/arch/cluster_sm90.hpp`)
- Combine with `warp_idx == 0` for block-level single-thread selection

#### Key Functions (C++):
- `tma_partition()` - Called by ALL threads to partition tensors (see `include/cute/atom/copy_traits_sm90_tma.hpp`)
- `copy()` with TMA atom - Called by SINGLE elected thread only

## Profiler Usage

```bash
# Profile GEMM kernels
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_*gemm* --m=4096 --n=4096 --k=4096

# Profile convolution
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_*fprop* --n=8 --h=224 --w=224 --c=128 --k=128
```

## CuTe DSL Programming Guide

### Understanding CuTe DSL APIs

**When CuTe DSL runtime code or documentation is unclear, always search for similar concepts and implementations in the C++ CuTe library (`include/cute/`).** CuTe DSL mirrors C++ CuTe abstractions, so the C++ headers often provide clearer documentation, more detailed comments, and reference implementations that explain the underlying concepts.

Key C++ reference locations:
- `include/cute/layout.hpp` - Layout algebra and composition
- `include/cute/tensor.hpp` - Tensor abstractions
- `include/cute/atom/mma_atom.hpp` - MMA atom concepts
- `include/cute/atom/copy_atom.hpp` - Copy atom concepts
- `include/cute/algorithm/` - Core algorithms (copy, gemm, etc.)
- `include/cute/arch/` - Architecture-specific PTX wrappers

### Key Runtime Modules

| Module | Purpose |
|--------|---------|
| `cutlass.cute.arch.mbar` | Low-level mbarrier operations |
| `cutlass.pipeline` | High-level pipeline abstractions |
| `cutlass.pipeline.helpers` | `MbarrierArray`, `PipelineState`, `NamedBarrier` |
| `cutlass.pipeline.sm90` | Hopper pipeline classes |
| `cutlass.pipeline.sm100` | Blackwell pipeline classes |

### Pipeline Synchronization (mbar)

**Prefer high-level pipeline abstractions over raw mbarrier calls.**

#### Creating a Pipeline

```python
import cutlass.pipeline as pipeline

# Create producer/consumer with PipelineTmaUmma (Blackwell) or PipelineTmaAsync (Hopper)
producer, consumer = pipeline.PipelineTmaUmma.create(
    barrier_storage=smem_barrier_ptr,    # Pointer to shared memory (8-byte aligned)
    num_stages=num_stages,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_count),
    tx_count=bytes_per_stage,            # Expected TMA transaction bytes
    cta_layout_vmnk=cluster_layout,
    defer_sync=True,                     # Defer barrier init sync
).make_participants()

# When using defer_sync=True, must call init functions
pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_shape, is_relaxed=True)
# ... other initialization ...
pipeline.pipeline_init_wait(cluster_shape_mn=cluster_shape)
```

#### Producer Pattern

```python
producer.reset()
for k in range(k_tiles):
    handle = producer.acquire_and_advance()  # Wait for empty buffer
    cute.copy(tma_atom, src, dst, tma_bar_ptr=handle.barrier)
producer.tail()  # Required cleanup
```

#### Consumer Pattern

```python
consumer.reset()
for k in range(k_tiles):
    handle = consumer.wait_and_advance()  # Wait for full buffer
    cute.gemm(tiled_mma, acc, A, B, acc)  # Use data
    handle.release()                       # Signal buffer empty
```

#### Pipeline Types

| Class | Use Case |
|-------|----------|
| `PipelineTmaAsync` | TMA load → AsyncThread consume (Hopper) |
| `PipelineTmaUmma` | TMA load → UMMA consume (Blackwell) |
| `PipelineAsync` | Generic async producer-consumer |
| `PipelineUmmaAsync` | UMMA produce → AsyncThread consume |
| `PipelineTmaStore` | TMA store sync (fences, no mbarriers) |

#### Low-Level mbarrier Functions (when needed)

```python
from cutlass.cute.arch.mbar import (
    mbarrier_init,              # Initialize with arrival count
    mbarrier_arrive,            # Signal arrival
    mbarrier_wait,              # Block until phase complete
    mbarrier_arrive_and_expect_tx,  # For TMA operations
    mbarrier_try_wait,          # Non-blocking wait
)
```

### Best Practices

1. **Use `defer_sync=True`** and explicit `pipeline_init_arrive/wait` for cluster sync
2. **Use `try_acquire()`/`try_wait()`** for non-blocking peeks to overlap computation
3. **Always call `producer.tail()`** before exiting producer warps
4. **Use `elect_one()`** context for single-thread barrier operations
5. **Allocate barrier storage** in shared memory with 8-byte alignment

### Thread Election APIs

CuTe DSL provides thread election for single-thread operations (TMA, barrier management).

#### Warp-Level Election

```python
from cutlass.cute.arch.elect import elect_one

# Only one thread in the warp executes this block
with elect_one():
    mbarrier_arrive(barrier_ptr)
```

#### Block-Level Election (Manual Pattern)

No built-in API exists for block-level election. Combine warp index check with `elect_one()`:

```python
warp_idx = thread_idx // 32
if warp_idx == 0:
    with elect_one():
        # Only one thread in entire thread block executes this
        pass
```

### TMA Usage in CuTe DSL

**TMA operations must be issued by a single elected thread**, not all threads.

#### Correct Pattern:
```python
# Single thread issues TMA load
warp_idx = thread_idx // 32
if warp_idx == tma_warp_id:
    with elect_one():
        cute.copy(tma_atom, src, dst, tma_bar_ptr=barrier, mcast_mask=mask)
```

#### cute.copy() with TMA - Called by SINGLE Thread:
```python
# Only elected thread issues the actual TMA copy
with elect_one():
    cute.copy(tma_atom, tiled_src, tiled_dst, tma_bar_ptr=barrier)
```

### Reference Examples

- `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py` - Hopper GEMM with TMA pipelines
- `examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py` - Blackwell GEMM with UMMA pipelines
- `examples/python/CuTeDSL/notebooks/async_pipeline.ipynb` - Pipeline tutorial

## Important Notes

- CUTLASS is header-only; include `include/` in your project
- CUDA 12.8+ required for Blackwell (SM100), CUDA 12.0+ for Hopper WGMMA
- Windows builds are currently broken for CUTLASS 4.x
- GCC 8.1-8.3 has known SFINAE issues; use GCC 7.5 or GCC 9+
