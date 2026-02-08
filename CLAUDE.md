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

### Compaction Guidelines

When CLAUDE.md grows too large, apply these principles:

1. Identify redundant content by looking for the same concept explained in multiple locations or example commands with the same set of non-trivial flags.
2. Cross-reference instead of duplicate. When the same concept appears in multiple sections, keep each section self-contained but replace duplicated explanations with brief summaries and cross-references to the authoritative section
3. Propose a high-level summary, then proceed to each compaction step individually and wait for user instructions before proceeding

### Evidence-Based Claims (Critical)

**NEVER make claims or assumptions about hardware behavior, synchronization requirements, or instruction semantics without explicit evidence from authoritative sources.** Common mistakes to avoid:

1. **Unjustified extrapolation**: Do not assume how one component works based on how another component works. For example, do not assume the TMA's mbarrier access uses the async proxy just because the TMA's data copy uses the async proxy.

2. **Plausible-sounding speculation**: Statements like "this might internally use X" or "this probably works by Y" without evidence are harmful. Either find evidence or explicitly state "I don't know."

3. **Conflating agent and method**: The *agent* performing an operation (e.g., TMA hardware) is distinct from the *method* of memory access (e.g., async proxy vs generic proxy). Do not conflate these.

**Before making any claim about PTX/CUDA behavior:**
1. Search the PTX ISA manual (`manual/ptx_isa_9.1.pdf`) for explicit statements
2. Search the CUDA Programming Guide (`manual/cuda-programming-guide.pdf`) for context
3. If no evidence exists, state "The documentation does not specify this" rather than guessing

## Repository Overview

CUTLASS is NVIDIA's CUDA Templates for Linear Algebra Subroutines - a header-only C++ template library for high-performance matrix multiplication (GEMM) and related computations. It supports architectures from Volta (SM70) through Blackwell (SM100/SM120).

CUTLASS 4.x adds **CuTe DSL**, a Python-native interface for writing high-performance CUDA kernels without C++ expertise.

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

### TMA and mbarrier Synchronization Model (PTX Level)

This section documents the correct model for how TMA (`cp.async.bulk`) interacts with mbarrier for synchronization. Understanding this requires distinguishing between the **data plane** and **control plane** of TMA operations.

#### Data Plane vs Control Plane

TMA operations have two distinct aspects:

| Aspect | Operations | Proxy Used | Agent |
|--------|------------|------------|-------|
| **Data Plane** | Read from global, write to shared | Async proxy | TMA hardware |
| **Control Plane** | mbarrier access (complete-tx) | Generic proxy | TMA hardware |

**Key insight:** The TMA hardware unit is the agent for both, but it uses **different proxies** for different memory accesses.

#### Evidence from PTX ISA Manual

From §9.7.9.25.2 (Async Proxy):
> "The `cp{.reduce}.async.bulk` operations are performed in the asynchronous proxy (or async proxy)."

From §9.7.9.25.4.1 (cp.async.bulk):
> "This instruction accesses its mbarrier operand using **generic-proxy**."

These statements are complementary, not contradictory:
- "Operations" (data movement) use async proxy
- "mbarrier operand" access uses generic proxy

#### Synchronization Requirements

| Sync Point | Purpose | Required Instruction |
|------------|---------|---------------------|
| After `mbarrier.init`, before other threads use it | CTA-wide visibility of initialized mbarrier | `bar.sync` (CTA barrier) |
| After `mbarrier.try_wait` returns True | Visibility of data written by TMA | Implicit with `.acquire` qualifier |

**Critical clarification:** `fence.proxy.async` is needed only for ordering between generic proxy and async proxy operations on **data buffers** (e.g., threads write a buffer, then TMA reads it, or vice versa). It is NOT needed for mbarrier operations, since both threads and TMA access the mbarrier via generic proxy. The manual's `mbarrier.init` example uses `bar.sync`, not `fence.proxy.async`, for this reason.

#### Complete-tx Semantics

From PTX ISA §9.7.9.25.4.1:
> "The copy operation in `cp.async.bulk` is treated as a weak memory operation and the complete-tx operation on the mbarrier has **.release semantics at .cluster scope**."

This means:
1. The data write (async proxy) happens-before the complete-tx (generic proxy)
2. Any thread observing phase completion via `mbarrier.try_wait.acquire` will see the data
3. The release/acquire pairing provides the necessary memory ordering

#### Correct Synchronization Pattern (PTX)

```asm
// === INITIALIZATION (one elected thread) ===
elect.sync _|p, 0xffffffff;
@p mbarrier.init.shared::cta.b64 [mbar], 1;
@p mbarrier.expect_tx.shared::cta.b64 [mbar], 4096;

bar.sync 0;  // CTA barrier — all threads see initialized mbarrier

// === PRODUCER (one elected thread) ===
@p cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
      [smem_buffer], [gmem_ptr], 4096, [mbar];

// === CONSUMER (all threads) ===
mbarrier.try_wait.acquire.shared::cta.b64 complete, [mbar], phase;
ld.shared.b32 r0, [smem_buffer];  // Data now visible
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
4. **Allocate barrier storage** in shared memory with 8-byte alignment
5. **Use `elect_one()` correctly** — see "Thread Election and TMA Usage" below for when to use vs. avoid it

### Thread Election and TMA Usage

#### `elect_one()` — For Barrier Operations Only

```python
from cutlass.cute.arch.elect import elect_one

# Warp-level: one thread in the warp executes this block
with elect_one():
    mbarrier_arrive(barrier_ptr)

# Block-level: combine warp check with elect_one()
warp_idx = thread_idx // 32
if warp_idx == 0:
    with elect_one():
        mbarrier_arrive(barrier_ptr)
```

#### TMA Copies — No `elect_one()` Needed

**Key difference from C++:** In CuTe DSL, `cute.copy()` handles thread election internally for TMA atoms (see `algorithm.py:364-365`). In C++ CuTe, explicit election via `cute::elect_one_sync()` is required (see "TMA Operations and Thread Hierarchy" above).

**Do NOT wrap `cute.copy()` in `elect_one()`** — this nests two `elect.sync` instructions, causing deadlock (outer election diverges 31 threads; inner election expects all 32).

```python
# CORRECT: cute.copy() handles election internally
if warp_idx == 0:
    cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=barrier, mcast_mask=mask)

# WRONG: nested elect.sync causes deadlock
if warp_idx == 0:
    with elect_one():  # ← causes deadlock
        cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=barrier)
```

See `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py:756-773` for reference.

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

### Required Tools: `pdfgrep`, `pdftotext`, and `pdfimages`

These tools are required for searching the reference manuals. If any are missing, ask the user to install them via their system's package manager.

Packages needed: `pdftotext`, `pdfimages` → `poppler-utils`; `pdfgrep` → `pdfgrep`

### How to Search the Manuals

**PREFERRED: Use `pdfgrep` for fast, direct PDF searching.** The `-n` flag returns page numbers that correspond directly to `pdftotext -f/-l` page numbers (no offset needed).

```bash
pdfgrep -n "search term" manual/ptx_isa_9.1.pdf         # Search with page numbers
pdfgrep -ni "mbarrier" manual/ptx_isa_9.1.pdf           # Case-insensitive
pdfgrep -n -m 10 "cp.async.bulk" manual/ptx_isa_9.1.pdf # Limit results
pdfgrep -c "wgmma" manual/ptx_isa_9.1.pdf               # Count matches

# Then read specific pages using pdftotext with page numbers from pdfgrep
pdftotext -f 326 -l 330 manual/ptx_isa_9.1.pdf -        # Read pages around a match
```

**Workflow:** Use `pdfgrep -n` to locate the page, then `pdftotext -f <page> -l <page+N>` to read the full content around that page.

**Regex caveat:** `pdfgrep` uses ERE (like `grep -E`), so use `"A|B"` for alternation, NOT `"A\|B"` (BRE convention — silently matches nothing in ERE).

### Extracting and Viewing Figures from PDFs

The manuals contain diagrams and figures (e.g., matrix fragment layouts, memory hierarchy diagrams) that `pdftotext` silently skips. Use `pdfimages` to extract them, then view with the `Read` tool.

**Workflow:**
1. **Locate the figure** — use `pdfgrep -n` or `pdftotext` to find the figure number and its page
2. **Extract** — use `pdfimages -png` with the page range containing the figure
3. **View** — use the `Read` tool on the extracted PNG file

```bash
# Step 1: Find which page mentions the figure
pdfgrep -n "Figure 183" manual/ptx_isa_9.1.pdf

# Step 2: Extract images from that page (and adjacent page in case figure spans pages)
mkdir -p /tmp/ptx_figures
pdfimages -png -f 641 -l 642 manual/ptx_isa_9.1.pdf /tmp/ptx_figures/fig183
# Output files: /tmp/ptx_figures/fig183-000.png, fig183-001.png, ...

# Step 3: View extracted image with the Read tool
# Read /tmp/ptx_figures/fig183-000.png
```

**Identifying figures on multi-figure pages:** Use `pdfimages -list` to see metadata for each image without extracting. The `num` column matches the `-NNN` suffix in extracted filenames. Cross-reference the `page` column with figure captions from `pdftotext` to identify which image is which — images appear in document order within each page.

```bash
# List image metadata (page, index, dimensions) without extracting
pdfimages -list -f 694 -l 696 manual/ptx_isa_9.1.pdf
# Output:
# page   num  type   width height ...
#  694     0 image    1566   415  ...   ← Figure 207 (wide layout table)
#  694     1 image    1543   417  ...   ← Figure 208 (address table)
#  695     2 image    1090   581  ...   ← Figure 209 (taller, sparse layout)
#  ...

# Then extract only the page you need
pdfimages -png -f 694 -l 694 manual/ptx_isa_9.1.pdf /tmp/figs/out
# Produces out-000.png (Figure 207) and out-001.png (Figure 208)
```

**Tips:**
- Output files are named `<prefix>-NNN.png` where NNN matches the `num` column from `-list`; images within a page appear in document order
- Use `-f` and `-l` to limit to specific pages; without them it extracts ALL images from the entire PDF
- The `width`/`height` columns help distinguish figure types (e.g., wide tables vs tall diagrams) when captions are ambiguous
- `pdfgrep -n` page numbers work directly as `-f`/`-l` arguments (no offset needed)

### Methodology: Using the Built-in Table of Contents

All three PDFs have built-in Tables of Contents, but TOC page numbers differ from actual PDF page numbers due to front matter (title page, TOC itself, figure lists).

**Page Number Offsets:**
| Document | TOC Pages | Offset | Formula |
|----------|-----------|--------|---------|
| PTX ISA 9.1 | 1-20 | +12 | actual_page = toc_page + 12 |
| CUDA Programming Guide | 3-16 | +16 | actual_page = toc_page + 16 |
| CUDA Driver API | 1-25 | +38 | actual_page = toc_page + 38 |

**Determining the offset for a new manual:** Search the TOC for a known section, then verify against `pdfgrep -n` results. Example using PTX ISA:

```bash
# Step 1: Extract TOC and find a section's logical page number
pdftotext -layout -f 1 -l 8 manual/ptx_isa_9.1.pdf - | grep -i "wgmma"
# Output: "9.7.15  Warpgroup MMA Instructions . . . 569"

# Step 2: Use pdfgrep to find the actual PDF page
pdfgrep -n "Warpgroup MMA" manual/ptx_isa_9.1.pdf
# Shows actual page 581 → offset = 581 - 569 = +12

# Step 3: Verify with a few more sections
# PTX Ch 1 (TOC: 9) → pdfgrep confirms page 21 → 21 - 9 = +12 ✓
# PTX Ch 9 (TOC: 119) → pdfgrep confirms page 131 → 131 - 119 = +12 ✓
```

All page numbers below are actual PDF page numbers (offsets already applied). To extract pages: `pdftotext -f <start> -l <end> manual/<filename>.pdf -`

### CUDA Programming Guide (Release 13.1)

**File:** `manual/cuda-programming-guide.pdf` (Dec 2025, 638 pages, offset +16)

**Use for:** Memory management, streams, CUDA graphs, cooperative groups, async barriers, pipelines, TMA, compute capabilities, C++ extensions, floating-point.

| Section | Pages | Topics |
|---------|-------|--------|
| **Ch 1. Introduction** | 19-32 | GPU architecture, programming model, compute capability |
| **Ch 2. Programming GPUs** | 33-103 | Kernels, memory, streams, async execution, NVCC |
| 2.1 Intro to CUDA C++ | 33-49 | Kernels, thread indices, memory management, synchronization |
| 2.1.10 Thread Block Clusters | 50-51 | Cluster launch, triple chevron notation |
| 2.2 Writing CUDA SIMT Kernels | 51-70 | Thread hierarchy, memory spaces, coalescing, bank conflicts, atomics |
| 2.2.3 Memory Spaces | 52-57 | Global, shared, registers, local, constant |
| 2.2.4 Shared Memory & Bank Conflicts | 59-69 | Bank conflicts, access patterns |
| 2.3 Asynchronous Execution | 72-86 | Streams, events, callbacks, synchronization |
| 2.4 Unified and System Memory | 86-95 | Unified virtual address space, managed memory, page-locked |
| 2.5 NVCC Compiler | 96-103 | Compilation workflow, PTX/cubin generation |
| **Ch 3. Advanced CUDA** | 105-150 | Advanced APIs, clusters, driver API, multi-GPU |
| 3.1 Advanced CUDA APIs | 105-115 | cudaLaunchKernelEx, cluster launch, programmatic dependent launch |
| 3.2 Advanced Kernel Programming | 116-133 | PTX inline, SIMT model, thread scopes, async barriers |
| 3.3 CUDA Driver API | 134-141 | Context, module, kernel execution, runtime interop |
| 3.4 Multi-GPU Programming | 141-146 | Device enumeration, peer-to-peer, managed memory |
| 3.5 Tour of CUDA Features | 146-150 | Barriers, TMA, pipelines, graphs, lazy loading, IPC |
| **Ch 4. CUDA Features** | 151-438 | Unified memory, graphs, pipelines, TMA, virtual memory |
| 4.1 Unified Memory | 151-177 | Performance tuning, prefetch, advise |
| 4.2 CUDA Graphs | 178-223 | Graph creation, instantiation, conditional/memory nodes |
| 4.3 Stream-Ordered Memory | 225-237 | cudaMallocAsync, memory pools |
| 4.4 Cooperative Groups | 237-244 | Group types, synchronization, collective operations |
| 4.5 Programmatic Dependent Launch | 244-247 | cudaTriggerProgrammaticLaunchCompletion |
| 4.6 Green Contexts | 247-267 | SM partitioning, resource descriptors |
| 4.7 Lazy Loading | 268-270 | CUDA_MODULE_LOADING |
| 4.9 Asynchronous Barriers | 272-290 | cuda::barrier, phases, completion functions |
| 4.10 Pipelines | 291-297 | cuda::pipeline, stages, async data movement |
| **4.11 Asynchronous Data Copies** | 299-340 | LDGSTS (cp.async), TMA, STAS |
| 4.11.2 TMA (Tensor Memory Accelerator) | 315-340 | TMA operations, tensor maps |
| 4.12 Cluster Launch Control | 343-351 | Work stealing, thread block cancellation |
| 4.13 L2 Cache Control | 351-355 | Persistent cache, access policies |
| 4.14 Memory Sync Domains | 356-357 | Fence interference isolation |
| 4.15 Interprocess Communication | 358-360 | IPC handles, virtual memory IPC |
| 4.16 Virtual Memory Management | 360-377 | cuMemCreate, cuMemMap, multicast, IPC |
| 4.17 Extended GPU Memory | 378-382 | EGM platforms, socket identifiers |
| 4.18 Dynamic Parallelism | 383-391 | Device-side kernel launch, streams/events |
| 4.19 CUDA Interoperability | 393-409 | OpenGL, Direct3D, Vulkan, NVSCI |
| 4.20 Driver Entry Point Access | 426-436 | cuGetProcAddress |
| **Ch 5. Technical Appendices** | 439-622 | Compute capabilities, C++ support, floating-point |
| 5.1 Compute Capabilities | 439-447 | Feature availability, specs per SM version |
| 5.2 Environment Variables | 447-454 | CUDA_VISIBLE_DEVICES, cache, JIT, lazy loading |
| 5.3 C++ Language Support | 455-513 | C++11/14/17/20, lambdas, restrictions |
| 5.4 C/C++ Language Extensions | 514-571 | `__device__`, `__global__`, `__shared__`, atomics, warp functions |
| 5.4.6 Warp Functions & Shuffle | 547-555 | Warp vote, shuffle, match |
| 5.4.11 WMMA (Warp Matrix) | 572-578 | Warp matrix multiply-accumulate |
| 5.5 Floating-Point | 578-605 | IEEE-754, intrinsics, accuracy |
| 5.6 Device-Callable APIs | 606-622 | Memory barriers, cluster sync |
| **Ch 6. Notices** | 623-638 | Legal notices |

### PTX ISA Reference (Version 9.1)

**File:** `manual/ptx_isa_9.1.pdf` (Jan 2026, 896 pages, offset +12)

**Use for:** PTX instruction syntax/semantics, memory consistency, MMA shapes, mbarrier, TMA async copy, tcgen05 (Blackwell).

| Section | Pages | Topics |
|---------|-------|--------|
| **Ch 1. Introduction** | 21-23 | PTX goals, ISA versioning, version 9.1 features |
| **Ch 2. Programming Model** | 25-29 | Thread hierarchy (CTA, clusters, grids), memory hierarchy |
| **Ch 3. PTX Machine Model** | 31-33 | SIMT multiprocessors, independent thread scheduling |
| **Ch 4. Syntax** | 35-43 | Source format, statements, identifiers, constants |
| **Ch 5. State Spaces, Types, Variables** | 45-100 | State spaces, types, tensors, TMA descriptors |
| 5.1 State Spaces | 45-52 | Register, shared, global, local, const, param |
| 5.2 Data Types | 53-60 | Fundamental and sub-byte types |
| 5.5 Tensors/TMA Descriptors | 67-100 | Tensor types, TMA descriptor format |
| **Ch 6. Instruction Operands** | 101-111 | Operand syntax, addressing, vectors |
| **Ch 7. Abstracting the ABI** | 103-111 | Function definitions, calling conventions |
| **Ch 8. Memory Consistency Model** | 113-128 | Scopes, ordering, fences, axioms |
| **Ch 9. Instruction Set** | 131-776 | Complete instruction reference |
| 9.7.1 Integer Arithmetic | 139-167 | add, sub, mul, mad, div, rem, abs |
| 9.7.3 Floating Point | 174-213 | fadd, fmul, fma, frcp, fsqrt |
| 9.7.9 Data Movement (ld/st) | 261-319 | Load, store, move, prefetch |
| 9.7.9.25 TMA/Async Copy | 320-352 | cp.async.bulk, tensor copy |
| 9.7.12 Control Flow | 377-384 | bra, call, ret, exit |
| 9.7.13 Parallel Sync | 385-414 | bar, fence, atom, red |
| 9.7.13.15 mbarrier | 415-436 | mbarrier init, arrive, wait, try_wait |
| 9.7.14 Warp MMA/wmma | 441-544 | Warp-level matrix multiply |
| 9.7.15 Warpgroup MMA/wgmma | 581-634 | Hopper warpgroup MMA |
| 9.7.16 TensorCore 5th Gen/tcgen05 | 635-752 | Blackwell TensorCore |
| **Ch 10. Special Registers** | 777-805 | `%tid`, `%ctaid`, `%laneid`, `%warpid`, `%clock` |
| **Ch 11. Directives** | 807-851 | `.version`, `.target`, `.maxntid`, `.maxnreg` |
| **Ch 12. Pragma Strings** | 853-871 | Compiler hints |
| **Ch 13. Release Notes** | 873-901 | Version history |

### CUDA Driver API Reference

**File:** `manual/CUDA_Driver_API.pdf` (Jan 2026, 803 pages, offset +38)

Low-level Driver API (`cuda.h`) with explicit control over contexts, modules, and kernel loading. All functions use the `cu` prefix.

**Date caveat:** Title page says "January 2024" — this is a template error. PDF metadata creation date is January 8, 2026; content includes compute capabilities up to `sm_121a`.

**Use for:** Driver API function signatures, return codes, cuMemAlloc/cuMemcpy, cuLaunchKernel, cuTensorMapEncodeTiled, context/module/library management, virtual memory, graph APIs, green contexts, occupancy, device attributes.

| Section | Pages | Topics |
|---------|-------|--------|
| **Ch 1. Driver vs Runtime APIs** | 39-40 | Complexity vs control, interoperability |
| **Ch 2. API Synchronization** | 41-42 | API call synchronization semantics |
| **Ch 3. Stream Synchronization** | 43-44 | Legacy vs per-thread default stream |
| **Ch 4. Graph Thread Safety** | 45 | Thread safety rules for graph objects |
| **Ch 5. Version Mixing Rules** | 46 | Driver/runtime version compatibility |
| **Ch 6. API Modules** | 47-707 | All API functions |
| 6.1 Data Types | 48-135 | Enums, typedefs, structs (`CUdevice_attribute`, `CUresult`, `CUtensorMap`) |
| 6.1: CUdevice_attribute | 61-69 | Device attribute enum values |
| 6.1: CUresult Error Codes | 110-118 | Error code enum values |
| 6.2 Error Handling | 136-137 | `cuGetErrorString`, `cuGetErrorName` |
| 6.3 Initialization | 137-138 | `cuInit` |
| 6.4 Version Management | 138-139 | `cuDriverGetVersion` |
| 6.5 Device Management | 138-150 | `cuDeviceGet`, `cuDeviceGetAttribute`, etc. |
| 6.6 Device Management [DEPRECATED] | 150-152 | Legacy device property queries |
| 6.7 Primary Context Management | 152-157 | `cuDevicePrimaryCtx*` |
| 6.8 Context Management | 157-178 | `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSynchronize`, etc. |
| 6.9 Context Management [DEPRECATED] | 178-182 | Legacy context APIs |
| 6.10 Module Management | 182-195 | `cuModuleLoad`, `cuModuleGetFunction`, etc. |
| 6.11 Module Management [DEPRECATED] | 195-196 | Legacy module APIs |
| 6.12 Library Management | 196-213 | `cuLibraryLoadFromFile`, `cuKernelGetFunction`, etc. |
| **6.13 Memory Management** | 213-309 | `cuMemAlloc`, `cuMemFree`, `cuMemcpy*`, `cuMemAllocHost`, etc. |
| **6.14 Virtual Memory Management** | 309-324 | `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, etc. |
| 6.15 Stream Ordered Memory | 324-339 | `cuMemAllocAsync`, `cuMemPoolCreate`, etc. |
| 6.16 Multicast Objects | 339-349 | `cuMulticastCreate`, `cuMulticastAddDevice`, etc. |
| 6.17 Unified Addressing | 349-369 | `cuPointerGetAttribute`, `cuMemPrefetchAsync`, etc. |
| 6.18 Stream Management | 369-393 | `cuStreamCreate`, `cuStreamSynchronize`, etc. |
| 6.19 Event Management | 393-399 | `cuEventCreate`, `cuEventRecord`, `cuEventElapsedTime` |
| 6.20 External Resource Interop | 399-414 | `cuImportExternalMemory`, semaphores |
| 6.21 Stream Memory Operations | 414-419 | `cuStreamBatchMemOp`, `cuStreamWriteValue32` |
| **6.22 Execution Control** | 419-441 | `cuLaunchKernel`, `cuLaunchKernelEx`, `cuFuncSetAttribute` |
| 6.23 Execution Control [DEPRECATED] | 441-452 | Legacy kernel launch APIs |
| **6.24 Graph Management** | 452-535 | `cuGraphCreate`, `cuGraphInstantiate`, `cuGraphLaunch`, etc. |
| 6.25 Occupancy | 535-543 | `cuOccupancyMaxActiveBlocksPerMultiprocessor`, etc. |
| 6.26 Texture Ref [DEPRECATED] | 543-563 | Legacy texture reference APIs |
| 6.27 Surface Ref [DEPRECATED] | 563-564 | Legacy surface reference APIs |
| 6.28 Texture Object Management | 564-571 | `cuTexObjectCreate`, `cuTexObjectDestroy` |
| 6.29 Surface Object Management | 571-573 | `cuSurfObjectCreate`, `cuSurfObjectDestroy` |
| **6.30 Tensor Map Objects** | 573-589 | `cuTensorMapEncodeTiled`, `cuTensorMapEncodeIm2col` |
| 6.31 Peer Context Memory Access | 589-594 | `cuCtxEnablePeerAccess`, `cuDeviceCanAccessPeer` |
| 6.32 Graphics Interoperability | 594-600 | `cuGraphicsMapResources`, etc. |
| 6.33 Driver Entry Point Access | 600-602 | `cuGetProcAddress` |
| 6.34 Coredump Attributes | 602-610 | `cuCoredumpGetAttribute`, `cuCoredumpSetAttribute` |
| **6.35 Green Contexts** | 610-628 | `cuGreenCtxCreate`, `cuSmResourceSplit` |
| 6.36 Error Log Management | 628-631 | `cuGetErrorLog` |
| 6.37 CUDA Checkpointing | 631-635 | `cuCheckpointLock`, `cuCheckpointRestore` |
| 6.38 Profiler Control [DEPRECATED] | 635-636 | Legacy profiler APIs |
| 6.39 Profiler Control | 636-637 | `cuProfilerStart`, `cuProfilerStop` |
| 6.40-6.45 Graphics Interop | 637-707 | OpenGL, Direct3D 9/10/11, VDPAU, EGL |
| **Ch 7. Data Structures** | 708-803 | Struct/union definitions for API parameters |

### Quick Reference: When to Consult Each Manual

For questions that map directly to a section name, use the per-manual indexes above. This table covers cases where the right source is non-obvious or spans multiple manuals:

| Question Type | Primary Source |
|--------------|----------------|
| "What is the thread hierarchy for clusters?" | PTX ISA 2.2.2 **and** CUDA Guide 2.1.10 |
| "What compute capability do I need for feature X?" | CUDA Guide Ch. 5 **and** PTX ISA release notes (Ch 13) |
| "How do I use TMA for tensor copies?" | CUDA Guide 4.11.2 (programming model) **and** PTX ISA 9.7.9.25 (instruction syntax) |
| "What MMA shapes are supported on SM90?" | PTX ISA 9.7.15 (wgmma), not CUDA Guide |
| "How does warp divergence affect performance?" | CUDA Guide Ch. 2.2, not PTX ISA |
