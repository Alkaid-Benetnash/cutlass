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
4. **Use `elect_one()`** context for single-thread barrier operations (but NOT for TMA copies - see below)
5. **Allocate barrier storage** in shared memory with 8-byte alignment
6. **Never wrap `cute.copy()` with TMA atoms in `elect_one()`** - the copy operation handles election internally

### Thread Election APIs

CuTe DSL provides thread election for single-thread operations (barrier management, certain synchronization primitives).

**WARNING:** Do NOT use `elect_one()` around operations that handle election internally (like `cute.copy()` with TMA atoms). This causes nested `elect.sync` instructions that deadlock. See "TMA Usage in CuTe DSL" below.

#### Warp-Level Election

```python
from cutlass.cute.arch.elect import elect_one

# Only one thread in the warp executes this block
with elect_one():
    mbarrier_arrive(barrier_ptr)
```

#### Block-Level Election (Manual Pattern)

No built-in API exists for block-level election. Combine warp index check with `elect_one()` for barrier operations (NOT for TMA copies):

```python
warp_idx = thread_idx // 32
if warp_idx == 0:
    with elect_one():
        # Only one thread in entire thread block executes this
        # Use for barrier ops, NOT for cute.copy() with TMA
        mbarrier_arrive(barrier_ptr)
```

### TMA Usage in CuTe DSL

**IMPORTANT: `cute.copy()` handles thread election internally for TMA atoms.** Do NOT wrap TMA copies in `elect_one()` - this causes nested `elect.sync` instructions that deadlock.

#### Why This Matters

The `elect.sync` PTX instruction requires ALL threads specified in its mask to participate. When you wrap `cute.copy()` in `elect_one()`:
1. Outer `elect_one()` generates `elect.sync -1` → 31 threads branch away
2. Inner election from `cute.copy()` generates another `elect.sync -1` → expects 32 threads, only 1 present
3. **DEADLOCK**: The single thread waits forever for 31 threads that already diverged

#### Evidence from Source Code

From `python/CuTeDSL/cutlass/cute/algorithm.py:364-365`:
```
For Copy Atoms requiring single-threaded execution, thread election is managed automatically by the
copy operation. External thread selection mechanisms are not necessary.
```

#### Correct Pattern (CuTe DSL):
```python
# Restrict to first warp, but do NOT use elect_one() around cute.copy()
if warp_idx == 0:
    tAsA, tAgA = cpasync.tma_partition(tma_atom, ...)
    # cute.copy() handles election internally - just call it directly
    cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=barrier, mcast_mask=mask)
```

See `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py:756-773` for reference usage.

#### Incorrect Pattern (causes deadlock):
```python
# DO NOT DO THIS - nested elect.sync causes deadlock
if warp_idx == 0:
    with elect_one():  # ← WRONG: cute.copy() already handles election
        cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=barrier)
```

#### When to Use `elect_one()`

Use `elect_one()` for operations that do NOT handle election internally:
```python
# Correct: mbarrier operations need explicit election
with elect_one():
    mbarrier_arrive(barrier_ptr)
    mbarrier_arrive_and_expect_tx(barrier_ptr, tx_bytes)
```

#### C++ vs CuTe DSL Difference

Note that C++ CuTe requires explicit election for TMA (using `cute::elect_one_sync()`), but CuTe DSL's `cute.copy()` wraps this automatically. This is a key API difference between the two.

### Reference Examples

- `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py` - Hopper GEMM with TMA pipelines
- `examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py` - Blackwell GEMM with UMMA pipelines
- `examples/python/CuTeDSL/notebooks/async_pipeline.ipynb` - Pipeline tutorial

## Important Notes

- CUTLASS is header-only; include `include/` in your project
- CUDA 12.8+ required for Blackwell (SM100), CUDA 12.0+ for Hopper WGMMA
- Windows builds are currently broken for CUTLASS 4.x
- GCC 8.1-8.3 has known SFINAE issues; use GCC 7.5 or GCC 9+

---

## NVIDIA Reference Documentation

**Ground truth reference manuals are available in `manual/` for authoritative information on CUDA and PTX.** When uncertain about CUDA programming concepts, PTX instruction semantics, or hardware behavior, consult these documents.

### How to Search the Manuals

```bash
# Search entire document (slow but comprehensive)
pdftotext manual/cuda-programming-guide.pdf - | grep -i "search term"
pdftotext manual/ptx_isa_9.1.pdf - | grep -i "search term"

# PREFERRED: Extract specific pages by topic (fast, use page index below)
pdftotext -f 403 -l 425 manual/ptx_isa_9.1.pdf -    # mbarrier instructions
pdftotext -f 569 -l 620 manual/ptx_isa_9.1.pdf -    # wgmma (Hopper MMA)
pdftotext -f 623 -l 740 manual/ptx_isa_9.1.pdf -    # tcgen05 (Blackwell)
pdftotext -f 308 -l 340 manual/ptx_isa_9.1.pdf -    # TMA async copy

# Search with context
pdftotext manual/cuda-programming-guide.pdf - | grep -i -A10 -B2 "unified memory"
```

### Methodology: Using the Built-in Table of Contents

Both PDFs have comprehensive built-in Tables of Contents at the beginning that map section names to page numbers. **However, the TOC page numbers have an offset from actual PDF page numbers.**

**Page Number Offsets (IMPORTANT):**
| Document | TOC Pages | Offset | Formula |
|----------|-----------|--------|---------|
| PTX ISA 9.1 | 1-20 | +12 | actual_page = toc_page + 12 |
| CUDA Programming Guide | 3-16 | +16 | actual_page = toc_page + 16 |

**Step 1: Extract and search the TOC to find the logical page number**
```bash
# PTX ISA TOC (use -layout to preserve page numbers on right side)
pdftotext -layout -f 1 -l 8 manual/ptx_isa_9.1.pdf - | grep -i "wgmma"
# Output: "9.7.15  Warpgroup MMA Instructions . . . 569"

# CUDA Programming Guide TOC
pdftotext -layout -f 3 -l 10 manual/cuda-programming-guide.pdf - | grep -i "unified"
```

**Step 2: Apply offset and extract the target pages**
```bash
# PTX: TOC says wgmma at 569 → actual page = 569 + 12 = 581
pdftotext -f 581 -l 632 manual/ptx_isa_9.1.pdf -

# CUDA: TOC says Ch2 at 17 → actual page = 17 + 16 = 33
pdftotext -f 33 -l 100 manual/cuda-programming-guide.pdf -
```

**Why offsets exist:** The TOC uses "logical" page numbers starting from the first content page (Chapter 1 = page 1 or similar), but the actual PDF includes front matter (title page, TOC itself, lists of figures/tables) before the content begins.

**Verified examples:**
| Section | TOC Page | Offset | Actual PDF Page |
|---------|----------|--------|-----------------|
| PTX Ch 1 Introduction | 9 | +12 | 21 |
| PTX Ch 9 Instruction Set | 119 | +12 | 131 |
| PTX wgmma (9.7.15) | 569 | +12 | 581 |
| PTX mbarrier (9.7.13.15) | 403 | +12 | 415 |
| CUDA Ch 1 Introduction | 3 | +16 | 19 |
| CUDA Ch 2 Programming | 17 | +16 | 33 |

### CUDA Programming Guide (Release 13.1)

**File:** `manual/cuda-programming-guide.pdf` (Dec 2025, 638 pages)

*Note: Offset of +16 has been applied. These are actual `pdftotext -f/-l` page numbers.*

#### Chapter Index (with Actual PDF Page Numbers)

| Chapter | Actual PDF Pages | Topics Covered |
|---------|------------------|----------------|
| **1. Introduction to CUDA** | 19-32 | GPU architecture, programming model, thread hierarchy, memory hierarchy, compute capability, PTX/cubin/fatbin |
| **2. Programming GPUs in CUDA** | 33-103 | Kernels, memory management, unified memory, streams, events, async execution, cooperative groups, NVCC |
| **3. Advanced CUDA** | 105-150 | Advanced APIs, clusters, driver API, multi-GPU, features tour |
| **4. CUDA Features** | 151-438 | Unified memory, graphs, cooperative groups, pipelines, TMA, virtual memory, dynamic parallelism |
| **5. Technical Appendices** | 439-622 | Compute capabilities, environment variables, C++ support, floating-point, intrinsics |
| **6. Notices** | 623-638 | Legal notices |

#### Chapter 2: Programming GPUs in CUDA (Detailed)

| Section | Actual PDF Pages | Topics |
|---------|------------------|--------|
| 2.1 Intro to CUDA C++ | 33-49 | Kernels, thread indices, memory management, synchronization, error checking |
| 2.1.10 Thread Block Clusters | 50-51 | Cluster launch, triple chevron notation |
| 2.2 Writing CUDA SIMT Kernels | 51-70 | Thread hierarchy, memory spaces (global, shared, registers, local, constant), coalescing, bank conflicts, atomics |
| 2.3 Asynchronous Execution | 72-86 | Streams, events, callbacks, stream ordering, synchronization |
| 2.4 Unified and System Memory | 86-95 | Unified virtual address space, managed memory, page-locked host memory |
| 2.5 NVCC Compiler | 96-103 | Compilation workflow, PTX/cubin generation, compiler options |

#### Chapter 3: Advanced CUDA (Detailed)

| Section | Actual PDF Pages | Topics |
|---------|------------------|--------|
| 3.1 Advanced CUDA APIs | 105-115 | cudaLaunchKernelEx, cluster launch, streams, programmatic dependent launch |
| 3.2 Advanced Kernel Programming | 116-133 | PTX inline, hardware implementation, SIMT model, thread scopes, async barriers, pipelines |
| 3.3 CUDA Driver API | 134-141 | Context, module, kernel execution, runtime interop |
| 3.4 Multi-GPU Programming | 141-146 | Device enumeration, peer-to-peer transfers, managed memory |
| 3.5 Tour of CUDA Features | 146-150 | Feature overview: barriers, TMA, pipelines, graphs, lazy loading, IPC |

#### Chapter 4: CUDA Features (Detailed)

| Section | Actual PDF Pages | Topics |
|---------|------------------|--------|
| 4.1 Unified Memory | 151-177 | Full support, performance tuning, Windows/Tegra, prefetch, advise |
| 4.2 CUDA Graphs | 178-223 | Graph structure, creation, instantiation, execution, conditional nodes, memory nodes, device launch |
| 4.3 Stream-Ordered Memory | 225-237 | cudaMallocAsync, memory pools, allocation/free |
| 4.4 Cooperative Groups | 237-244 | Group types, synchronization, collective operations |
| 4.5 Programmatic Dependent Launch | 244-247 | cudaTriggerProgrammaticLaunchCompletion |
| 4.6 Green Contexts | 247-267 | SM partitioning, resource descriptors |
| 4.7 Lazy Loading | 268-270 | CUDA_MODULE_LOADING |
| 4.9 Asynchronous Barriers | 272-290 | cuda::barrier, phases, completion functions, producer-consumer |
| 4.10 Pipelines | 291-297 | cuda::pipeline, stages, async data movement |
| 4.11 Asynchronous Data Copies | 299-340 | **LDGSTS (cp.async)**, **TMA (Tensor Memory Accelerator)**, STAS |
| 4.12 Cluster Launch Control | 343-351 | Work stealing, thread block cancellation |
| 4.13 L2 Cache Control | 351-355 | Persistent cache, access policies |
| 4.14 Memory Sync Domains | 356-357 | Fence interference isolation |
| 4.15 Interprocess Communication | 358-360 | IPC handles, virtual memory IPC |
| 4.16 Virtual Memory Management | 360-377 | cuMemCreate, cuMemMap, multicast, IPC |
| 4.17 Extended GPU Memory | 378-382 | EGM platforms, socket identifiers |
| 4.18 Dynamic Parallelism | 383-391 | Device-side kernel launch, streams/events |
| 4.19 CUDA Interoperability | 393-409 | OpenGL, Direct3D, Vulkan, NVSCI |
| 4.20 Driver Entry Point Access | 426-436 | cuGetProcAddress |

#### Chapter 5: Technical Appendices (Detailed)

| Section | Actual PDF Pages | Topics |
|---------|------------------|--------|
| 5.1 Compute Capabilities | 439-447 | Feature availability, technical specs per SM version |
| 5.2 Environment Variables | 447-454 | CUDA_VISIBLE_DEVICES, cache, JIT, lazy loading |
| 5.3 C++ Language Support | 455-513 | C++11/14/17/20, lambdas, restrictions |
| 5.4 C/C++ Language Extensions | 514-571 | `__device__`, `__global__`, `__shared__`, built-in variables, atomics, warp functions, WMMA |
| 5.5 Floating-Point | 578-605 | IEEE-754, data types, intrinsics, accuracy |
| 5.6 Device-Callable APIs | 606-622 | Memory barriers, cluster sync |

#### Key Sections Quick Reference (Actual PDF Page Numbers)

| Topic | Actual PDF Pages | Command to Extract |
|-------|------------------|-------------------|
| **Kernels & Thread Hierarchy** (2.1-2.2) | 33-70 | `pdftotext -f 33 -l 70 manual/cuda-programming-guide.pdf -` |
| **Memory Spaces** (2.2.3) | 52-57 | `pdftotext -f 52 -l 57 manual/cuda-programming-guide.pdf -` |
| **Shared Memory & Bank Conflicts** (2.2.4) | 59-69 | `pdftotext -f 59 -l 69 manual/cuda-programming-guide.pdf -` |
| **Streams & Events** (2.3) | 72-86 | `pdftotext -f 72 -l 86 manual/cuda-programming-guide.pdf -` |
| **Unified Memory** (4.1) | 151-177 | `pdftotext -f 151 -l 177 manual/cuda-programming-guide.pdf -` |
| **CUDA Graphs** (4.2) | 178-223 | `pdftotext -f 178 -l 223 manual/cuda-programming-guide.pdf -` |
| **Cooperative Groups** (4.4) | 237-244 | `pdftotext -f 237 -l 244 manual/cuda-programming-guide.pdf -` |
| **Asynchronous Barriers** (4.9) | 272-290 | `pdftotext -f 272 -l 290 manual/cuda-programming-guide.pdf -` |
| **Pipelines** (4.10) | 291-297 | `pdftotext -f 291 -l 297 manual/cuda-programming-guide.pdf -` |
| **TMA (Tensor Memory Accelerator)** (4.11.2) | 315-340 | `pdftotext -f 315 -l 340 manual/cuda-programming-guide.pdf -` |
| **Cluster Launch Control** (4.12) | 343-351 | `pdftotext -f 343 -l 351 manual/cuda-programming-guide.pdf -` |
| **Virtual Memory Management** (4.16) | 360-377 | `pdftotext -f 360 -l 377 manual/cuda-programming-guide.pdf -` |
| **Dynamic Parallelism** (4.18) | 383-391 | `pdftotext -f 383 -l 391 manual/cuda-programming-guide.pdf -` |
| **Compute Capabilities** (5.1) | 439-447 | `pdftotext -f 439 -l 447 manual/cuda-programming-guide.pdf -` |
| **C++ Extensions** (5.4) | 514-571 | `pdftotext -f 514 -l 571 manual/cuda-programming-guide.pdf -` |
| **Warp Functions & Shuffle** (5.4.6) | 547-555 | `pdftotext -f 547 -l 555 manual/cuda-programming-guide.pdf -` |
| **WMMA (Warp Matrix)** (5.4.11) | 572-578 | `pdftotext -f 572 -l 578 manual/cuda-programming-guide.pdf -` |
| **Floating-Point** (5.5) | 578-605 | `pdftotext -f 578 -l 605 manual/cuda-programming-guide.pdf -` |

**Use this document for:** Memory management, streams, occupancy, CUDA graphs, cooperative groups, async barriers, pipelines, TMA, compute capabilities, C++ extensions, floating-point standards.

### PTX ISA Reference (Version 9.1)

**File:** `manual/ptx_isa_9.1.pdf` (Jan 2026, 896 pages)

#### Chapter Index (with Actual PDF Page Numbers)

*Note: Offset of +12 has been applied. These are actual `pdftotext -f/-l` page numbers.*

| Chapter | Actual PDF Pages | Topics |
|---------|------------------|--------|
| **1. Introduction** | 21-23 | PTX goals, ISA versioning, version 9.1 features |
| **2. Programming Model** | 25-29 | Thread hierarchy (CTA, clusters, grids), memory hierarchy |
| **3. PTX Machine Model** | 31-33 | SIMT multiprocessors, independent thread scheduling |
| **4. Syntax** | 35-43 | Source format, statements, identifiers, constants |
| **5. State Spaces, Types, Variables** | 45-100 | State spaces, types, tensors, TMA descriptors |
| **6. Instruction Operands** | 101-111 | Operand syntax, addressing, vectors |
| **7. Abstracting the ABI** | 103-111 | Function definitions, calling conventions |
| **8. Memory Consistency Model** | 113-128 | Scopes, ordering, fences, axioms |
| **9. Instruction Set** | 131-776 | Complete instruction reference (see below) |
| **10. Special Registers** | 777-805 | `%tid`, `%ctaid`, `%laneid`, `%warpid`, `%clock`, etc. |
| **11. Directives** | 807-851 | `.version`, `.target`, `.maxntid`, `.maxnreg`, etc. |
| **12. Pragma Strings** | 853-871 | Compiler hints |
| **13. Release Notes** | 873-901 | Version history |

#### Key Sections Quick Reference (Actual PDF Page Numbers)

*Offset +12 applied. Ready for direct use with `pdftotext -f <start> -l <end>`.*

| Topic | Actual PDF Pages | Command to Extract |
|-------|------------------|-------------------|
| **State Spaces** (Ch 5.1) | 45-52 | `pdftotext -f 45 -l 52 manual/ptx_isa_9.1.pdf -` |
| **Data Types** (Ch 5.2) | 53-60 | `pdftotext -f 53 -l 60 manual/ptx_isa_9.1.pdf -` |
| **Tensors/TMA Descriptors** (Ch 5.5) | 67-100 | `pdftotext -f 67 -l 100 manual/ptx_isa_9.1.pdf -` |
| **Memory Consistency** (Ch 8) | 113-128 | `pdftotext -f 113 -l 128 manual/ptx_isa_9.1.pdf -` |
| **Integer Arithmetic** (9.7.1) | 139-167 | `pdftotext -f 139 -l 167 manual/ptx_isa_9.1.pdf -` |
| **Floating Point** (9.7.3) | 174-213 | `pdftotext -f 174 -l 213 manual/ptx_isa_9.1.pdf -` |
| **Data Movement/ld/st** (9.7.9) | 261-319 | `pdftotext -f 261 -l 319 manual/ptx_isa_9.1.pdf -` |
| **TMA/Async Copy** (9.7.9.25) | 320-352 | `pdftotext -f 320 -l 352 manual/ptx_isa_9.1.pdf -` |
| **Control Flow** (9.7.12) | 377-384 | `pdftotext -f 377 -l 384 manual/ptx_isa_9.1.pdf -` |
| **Parallel Sync (bar, fence, atom)** (9.7.13) | 385-414 | `pdftotext -f 385 -l 414 manual/ptx_isa_9.1.pdf -` |
| **mbarrier** (9.7.13.15) | 415-436 | `pdftotext -f 415 -l 436 manual/ptx_isa_9.1.pdf -` |
| **Warp MMA/wmma** (9.7.14) | 441-544 | `pdftotext -f 441 -l 544 manual/ptx_isa_9.1.pdf -` |
| **Warpgroup MMA/wgmma** (9.7.15) | 581-634 | `pdftotext -f 581 -l 634 manual/ptx_isa_9.1.pdf -` |
| **TensorCore 5th Gen/tcgen05** (9.7.16) | 635-752 | `pdftotext -f 635 -l 752 manual/ptx_isa_9.1.pdf -` |
| **Special Registers** (Ch 10) | 777-805 | `pdftotext -f 777 -l 805 manual/ptx_isa_9.1.pdf -` |

### Quick Reference: When to Consult Each Manual

| Question Type | Primary Source |
|--------------|----------------|
| "How do I use unified memory?" | CUDA Programming Guide Ch. 2 |
| "What is the syntax for `cp.async.bulk`?" | PTX ISA 9.7.9.25-9.7.9.27 |
| "How do mbarriers work?" | PTX ISA 9.7.13 (mbarrier instructions) |
| "What MMA shapes are supported on SM90?" | PTX ISA 9.7.15 (wgmma) |
| "What are the memory consistency rules?" | PTX ISA Ch. 8 |
| "How does warp divergence affect performance?" | CUDA Programming Guide Ch. 2.2 |
| "What is the thread hierarchy for clusters?" | PTX ISA 2.2.2, CUDA Guide 2.1.10 |
| "How do I use TMA for tensor copies?" | PTX ISA 9.7.9.25 (cp.async.bulk.tensor) |
| "What compute capability do I need for feature X?" | CUDA Programming Guide Ch. 5, PTX ISA release notes |
| "What are tcgen05 instructions?" | PTX ISA 9.7.16 (Blackwell TensorCore) |
