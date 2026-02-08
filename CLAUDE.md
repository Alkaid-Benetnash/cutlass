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

**Critical clarification:** `fence.proxy.async` is needed for ordering between generic proxy and async proxy operations on the **data buffers**, NOT for mbarrier operations. Since both threads and TMA access the mbarrier via generic proxy, no cross-proxy fence is needed for the mbarrier itself.

#### Example from PTX ISA Manual (mbarrier.init)

```asm
mbarrier.init.shared::cta.b64 [shMem], 12;
bar.sync 0;   // CTA barrier - NOT fence.proxy.async
// ... other mbarrier operations on shMem
```

The manual uses `bar.sync`, not `fence.proxy.async`, because mbarrier operations are entirely within the generic proxy domain.

#### When fence.proxy.async IS Needed

`fence.proxy.async` is needed when:
1. Threads write to a buffer via generic proxy, then TMA reads it via async proxy
2. TMA writes to a buffer via async proxy, then threads read it via generic proxy (though this has an implicit fence on completion)

It is NOT needed for mbarrier synchronization between `mbarrier.init` and `cp.async.bulk`.

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

// CTA barrier ensures all threads see initialized mbarrier
// (both thread and TMA access mbarrier via generic proxy, so bar.sync suffices)
bar.sync 0;

// === PRODUCER (one elected thread) ===
// Issue TMA - no fence.proxy.async needed for mbarrier
@p cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
      [smem_buffer], [gmem_ptr], 4096, [mbar];

// === CONSUMER (all threads) ===
// Wait with acquire semantics - provides ordering for data visibility
mbarrier.try_wait.acquire.shared::cta.b64 complete, [mbar], phase;

// Data is now visible - safe to read
ld.shared.b32 r0, [smem_buffer];
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

### Required Tools: `pdfgrep`, `pdftotext`, and `pdfimages`

**CRITICAL: `pdfgrep`, `pdftotext`, and `pdfimages` are essential tools for searching the reference manuals. Before proceeding with any manual lookup, verify they are installed.** If either tool is missing, **stop and ask the user to install them** using their system's package manager:

| Tool | Package | Purpose |
|------|---------|---------|
| `pdftotext` | `poppler-utils` | Extract text from PDF pages |
| `pdfimages` | `poppler-utils` | Extract embedded images/figures from PDF pages |
| `pdfgrep` | `pdfgrep` | Search text within PDFs with page numbers |

Installation hints by distro:
```bash
# Debian/Ubuntu
sudo apt install poppler-utils pdfgrep

# Fedora/RHEL
sudo dnf install poppler-utils pdfgrep

# Arch Linux
sudo pacman -S poppler pdfgrep

# macOS (Homebrew)
brew install poppler pdfgrep
```

### How to Search the Manuals

**PREFERRED: Use `pdfgrep` for fast, direct PDF searching.** The `-n` flag returns page numbers that correspond directly to `pdftotext -f/-l` page numbers (no offset needed).

```bash
# Search with page numbers (PREFERRED - fast, gives page locations)
pdfgrep -n "search term" manual/ptx_isa_9.1.pdf
pdfgrep -n "search term" manual/cuda-programming-guide.pdf

# Case-insensitive search
pdfgrep -ni "mbarrier" manual/ptx_isa_9.1.pdf

# Limit results
pdfgrep -n -m 10 "cp.async.bulk" manual/ptx_isa_9.1.pdf

# Count matches
pdfgrep -c "wgmma" manual/ptx_isa_9.1.pdf

# Then read specific pages using pdftotext with page numbers from pdfgrep
pdftotext -f 326 -l 330 manual/ptx_isa_9.1.pdf -    # Read pages around a match
```

**Workflow:** Use `pdfgrep -n` to locate the page, then `pdftotext -f <page> -l <page+N>` to read the full content around that page.

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
- `pdfimages` names output files as `<prefix>-NNN.png` where NNN matches the `num` column from `-list`
- Use `-f` and `-l` to limit to specific pages; without them it extracts ALL images from the entire PDF
- Images within a page appear in document order, so the Nth image on a page corresponds to the Nth figure caption from `pdftotext`
- The `width`/`height` columns can help distinguish figure types (e.g., wide tables vs tall diagrams) when captions alone are ambiguous
- The `pdfgrep -n` page numbers work directly as `-f`/`-l` arguments (no offset needed)

**Fallback: `pdftotext | grep` for when you need context lines or pipe-based filtering:**

```bash
# Search entire document (slower but supports context lines)
pdftotext manual/cuda-programming-guide.pdf - | grep -i -A10 -B2 "unified memory"

# Extract specific pages by topic (fast, use page index below)
pdftotext -f 403 -l 425 manual/ptx_isa_9.1.pdf -    # mbarrier instructions
pdftotext -f 569 -l 620 manual/ptx_isa_9.1.pdf -    # wgmma (Hopper MMA)
pdftotext -f 623 -l 740 manual/ptx_isa_9.1.pdf -    # tcgen05 (Blackwell)
pdftotext -f 308 -l 340 manual/ptx_isa_9.1.pdf -    # TMA async copy
```

### Methodology: Using the Built-in Table of Contents

All three PDFs have comprehensive built-in Tables of Contents at the beginning that map section names to page numbers. **However, the TOC page numbers have an offset from actual PDF page numbers.**

**Page Number Offsets (IMPORTANT):**
| Document | TOC Pages | Offset | Formula |
|----------|-----------|--------|---------|
| PTX ISA 9.1 | 1-20 | +12 | actual_page = toc_page + 12 |
| CUDA Programming Guide | 3-16 | +16 | actual_page = toc_page + 16 |
| CUDA Driver API | 1-25 | +38 | actual_page = toc_page + 38 |

**Step 1: Extract and search the TOC to find the logical page number**
```bash
# PTX ISA TOC (use -layout to preserve page numbers on right side)
pdftotext -layout -f 1 -l 8 manual/ptx_isa_9.1.pdf - | grep -i "wgmma"
# Output: "9.7.15  Warpgroup MMA Instructions . . . 569"

# CUDA Programming Guide TOC
pdftotext -layout -f 3 -l 10 manual/cuda-programming-guide.pdf - | grep -i "unified"

# CUDA Driver API TOC
pdftotext -layout -f 1 -l 25 manual/CUDA_Driver_API.pdf - | grep -i "memory management"
```

**Step 2: Apply offset and extract the target pages**
```bash
# PTX: TOC says wgmma at 569 → actual page = 569 + 12 = 581
pdftotext -f 581 -l 632 manual/ptx_isa_9.1.pdf -

# CUDA: TOC says Ch2 at 17 → actual page = 17 + 16 = 33
pdftotext -f 33 -l 100 manual/cuda-programming-guide.pdf -

# Driver API: TOC says Memory Management at 175 → actual page = 175 + 38 = 213
pdftotext -f 213 -l 308 manual/CUDA_Driver_API.pdf -
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
| Driver API Ch 1 | 1 | +38 | 39 |
| Driver API Ch 6 Modules | 9 | +38 | 47 |
| Driver API 6.13 Memory Mgmt | 175 | +38 | 213 |
| Driver API 6.30 Tensor Map | 535 | +38 | 573 |

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

### CUDA Driver API Reference

**File:** `manual/CUDA_Driver_API.pdf` (803 pages)

**Date caveat:** The title page says "January 2024", but this is a template error. The PDF metadata creation date is **January 8, 2026**, and the content includes compute capabilities up to `sm_121a` (via `CUjit_target` enum), confirming it covers hardware through Blackwell and beyond. This is a recent document despite the misleading title page date.

*Note: Offset of +38 has been applied. These are actual `pdftotext -f/-l` page numbers.*

This is the low-level CUDA Driver API reference. Unlike the Runtime API (`cuda_runtime.h`), the Driver API (`cuda.h`) provides explicit control over contexts, modules, and kernel loading. All functions use the `cu` prefix (e.g., `cuMemAlloc`, `cuLaunchKernel`).

#### Chapter Index (with Actual PDF Page Numbers)

| Chapter | Actual PDF Pages | Topics |
|---------|------------------|--------|
| **1. Driver vs Runtime APIs** | 39-40 | Complexity vs control, context management, interoperability |
| **2. API Synchronization Behavior** | 41-42 | API call synchronization semantics |
| **3. Stream Synchronization Behavior** | 43-44 | Legacy vs per-thread default stream |
| **4. Graph Object Thread Safety** | 45 | Thread safety rules for graph objects |
| **5. Rules for Version Mixing** | 46 | Driver/runtime version compatibility |
| **6. Modules** | 47-707 | All API functions (see detailed index below) |
| **7. Data Structures** | 708-803 | Struct/union definitions for API parameters |

#### Section 6: API Modules (Detailed)

| Section | Actual PDF Pages | Topics |
|---------|------------------|--------|
| 6.1 Data Types | 48-135 | Enums, typedefs, structs, constants (`CUdevice_attribute`, `CUresult`, `CUtensorMap`, etc.) |
| 6.2 Error Handling | 136-137 | `cuGetErrorString`, `cuGetErrorName` |
| 6.3 Initialization | 137-138 | `cuInit` |
| 6.4 Version Management | 138-139 | `cuDriverGetVersion` |
| 6.5 Device Management | 138-150 | `cuDeviceGet`, `cuDeviceGetAttribute`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetUuid`, etc. |
| 6.6 Device Management [DEPRECATED] | 150-152 | Legacy device property queries |
| 6.7 Primary Context Management | 152-157 | `cuDevicePrimaryCtxGetState`, `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxSetFlags` |
| 6.8 Context Management | 157-178 | `cuCtxCreate`, `cuCtxDestroy`, `cuCtxGetCurrent`, `cuCtxSetCurrent`, `cuCtxSynchronize`, `cuCtxGetDevice`, etc. |
| 6.9 Context Management [DEPRECATED] | 178-182 | Legacy context APIs |
| 6.10 Module Management | 182-195 | `cuModuleLoad`, `cuModuleGetFunction`, `cuModuleGetGlobal`, `cuModuleLoadData`, etc. |
| 6.11 Module Management [DEPRECATED] | 195-196 | Legacy module APIs |
| 6.12 Library Management | 196-213 | `cuLibraryLoadFromFile`, `cuLibraryGetKernel`, `cuKernelGetFunction`, etc. |
| **6.13 Memory Management** | 213-309 | `cuMemAlloc`, `cuMemFree`, `cuMemcpy`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemAllocHost`, `cuMemHostAlloc`, `cuMemGetInfo`, etc. |
| **6.14 Virtual Memory Management** | 309-324 | `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemAddressReserve`, `cuMemRelease`, `cuMemUnmap`, multicast |
| 6.15 Stream Ordered Memory Allocator | 324-339 | `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemPoolCreate`, `cuMemPoolSetAccess` |
| 6.16 Multicast Object Management | 339-349 | `cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindAddr`, `cuMulticastBindMem` |
| 6.17 Unified Addressing | 349-369 | `cuPointerGetAttribute`, `cuPointerSetAttribute`, `cuMemPrefetchAsync`, `cuMemAdvise`, `cuMemRangeGetAttribute` |
| 6.18 Stream Management | 369-393 | `cuStreamCreate`, `cuStreamSynchronize`, `cuStreamWaitEvent`, `cuStreamQuery`, `cuStreamBeginCapture`, etc. |
| 6.19 Event Management | 393-399 | `cuEventCreate`, `cuEventRecord`, `cuEventSynchronize`, `cuEventElapsedTime`, `cuEventQuery` |
| 6.20 External Resource Interop | 399-414 | `cuImportExternalMemory`, `cuExternalMemoryGetMappedBuffer`, `cuImportExternalSemaphore`, `cuSignalExternalSemaphoresAsync` |
| 6.21 Stream Memory Operations | 414-419 | `cuStreamBatchMemOp`, `cuStreamWriteValue32`, `cuStreamWaitValue32` |
| **6.22 Execution Control** | 419-441 | `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLaunchCooperativeKernel`, `cuFuncSetAttribute`, `cuFuncGetAttribute`, `cuFuncSetCacheConfig` |
| 6.23 Execution Control [DEPRECATED] | 441-452 | Legacy kernel launch APIs (`cuLaunch`, `cuParamSet*`) |
| **6.24 Graph Management** | 452-535 | `cuGraphCreate`, `cuGraphAddKernelNode`, `cuGraphInstantiate`, `cuGraphLaunch`, `cuGraphExecUpdate`, conditional nodes, memory nodes |
| 6.25 Occupancy | 535-543 | `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize`, etc. |
| 6.26 Texture Ref Management [DEPRECATED] | 543-563 | Legacy texture reference APIs |
| 6.27 Surface Ref Management [DEPRECATED] | 563-564 | Legacy surface reference APIs |
| 6.28 Texture Object Management | 564-571 | `cuTexObjectCreate`, `cuTexObjectDestroy`, `cuTexObjectGetTextureDesc` |
| 6.29 Surface Object Management | 571-573 | `cuSurfObjectCreate`, `cuSurfObjectDestroy` |
| **6.30 Tensor Map Object Management** | 573-589 | `cuTensorMapEncodeTiled`, `cuTensorMapEncodeIm2col`, `cuTensorMapReplaceAddress` |
| 6.31 Peer Context Memory Access | 589-594 | `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`, `cuDeviceCanAccessPeer`, `cuDeviceGetP2PAttribute` |
| 6.32 Graphics Interoperability | 594-600 | `cuGraphicsMapResources`, `cuGraphicsUnmapResources`, `cuGraphicsResourceGetMappedPointer` |
| 6.33 Driver Entry Point Access | 600-602 | `cuGetProcAddress` |
| 6.34 Coredump Attributes Control | 602-610 | `cuCoredumpGetAttribute`, `cuCoredumpSetAttribute` |
| **6.35 Green Contexts** | 610-628 | `cuGreenCtxCreate`, `cuDeviceGetDevResource`, `cuCtxFromGreenCtx`, `cuSmResourceSplit` |
| 6.36 Error Log Management | 628-631 | `cuGetErrorLog` |
| 6.37 CUDA Checkpointing | 631-635 | `cuCheckpointLock`, `cuCheckpointUnlock`, `cuCheckpointRestore`, `cuCheckpointCheckpoint` |
| 6.38 Profiler Control [DEPRECATED] | 635-636 | Legacy profiler APIs |
| 6.39 Profiler Control | 636-637 | `cuProfilerStart`, `cuProfilerStop` |
| 6.40-6.45 Graphics Interop | 637-707 | OpenGL, Direct3D 9/10/11, VDPAU, EGL interoperability |

#### Key Sections Quick Reference (Actual PDF Page Numbers)

| Topic | Actual PDF Pages | Command to Extract |
|-------|------------------|-------------------|
| **Driver vs Runtime APIs** (Ch 1) | 39-40 | `pdftotext -f 39 -l 40 manual/CUDA_Driver_API.pdf -` |
| **Data Types & Enums** (6.1) | 48-135 | `pdftotext -f 48 -l 135 manual/CUDA_Driver_API.pdf -` |
| **CUresult Error Codes** (6.1) | 110-118 | `pdftotext -f 110 -l 118 manual/CUDA_Driver_API.pdf -` |
| **CUdevice_attribute** (6.1) | 61-69 | `pdftotext -f 61 -l 69 manual/CUDA_Driver_API.pdf -` |
| **Initialization** (6.3) | 137-138 | `pdftotext -f 137 -l 138 manual/CUDA_Driver_API.pdf -` |
| **Device Management** (6.5) | 138-150 | `pdftotext -f 138 -l 150 manual/CUDA_Driver_API.pdf -` |
| **Context Management** (6.8) | 157-178 | `pdftotext -f 157 -l 178 manual/CUDA_Driver_API.pdf -` |
| **Module Management** (6.10) | 182-195 | `pdftotext -f 182 -l 195 manual/CUDA_Driver_API.pdf -` |
| **Library Management** (6.12) | 196-213 | `pdftotext -f 196 -l 213 manual/CUDA_Driver_API.pdf -` |
| **Memory Management** (6.13) | 213-309 | `pdftotext -f 213 -l 309 manual/CUDA_Driver_API.pdf -` |
| **Virtual Memory Management** (6.14) | 309-324 | `pdftotext -f 309 -l 324 manual/CUDA_Driver_API.pdf -` |
| **Stream Ordered Memory** (6.15) | 324-339 | `pdftotext -f 324 -l 339 manual/CUDA_Driver_API.pdf -` |
| **Multicast Objects** (6.16) | 339-349 | `pdftotext -f 339 -l 349 manual/CUDA_Driver_API.pdf -` |
| **Unified Addressing** (6.17) | 349-369 | `pdftotext -f 349 -l 369 manual/CUDA_Driver_API.pdf -` |
| **Stream Management** (6.18) | 369-393 | `pdftotext -f 369 -l 393 manual/CUDA_Driver_API.pdf -` |
| **Event Management** (6.19) | 393-399 | `pdftotext -f 393 -l 399 manual/CUDA_Driver_API.pdf -` |
| **Execution Control / Kernel Launch** (6.22) | 419-441 | `pdftotext -f 419 -l 441 manual/CUDA_Driver_API.pdf -` |
| **Graph Management** (6.24) | 452-535 | `pdftotext -f 452 -l 535 manual/CUDA_Driver_API.pdf -` |
| **Occupancy** (6.25) | 535-543 | `pdftotext -f 535 -l 543 manual/CUDA_Driver_API.pdf -` |
| **Tensor Map Object Management** (6.30) | 573-589 | `pdftotext -f 573 -l 589 manual/CUDA_Driver_API.pdf -` |
| **Green Contexts** (6.35) | 610-628 | `pdftotext -f 610 -l 628 manual/CUDA_Driver_API.pdf -` |
| **CUDA Checkpointing** (6.37) | 631-635 | `pdftotext -f 631 -l 635 manual/CUDA_Driver_API.pdf -` |
| **Data Structures** (Ch 7) | 708-803 | `pdftotext -f 708 -l 803 manual/CUDA_Driver_API.pdf -` |

**Use this document for:** Driver API function signatures, parameter details, return codes, `cuMemAlloc`/`cuMemcpy` variants, `cuLaunchKernel`/`cuLaunchKernelEx`, `cuTensorMapEncodeTiled`, context/module/library management, virtual memory (`cuMemCreate`/`cuMemMap`), graph APIs, green contexts, occupancy queries, device attributes.

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
| "What are the parameters for `cuLaunchKernel`?" | CUDA Driver API 6.22 (Execution Control) |
| "How do I use `cuMemAlloc` / `cuMemcpy`?" | CUDA Driver API 6.13 (Memory Management) |
| "What is `cuTensorMapEncodeTiled`?" | CUDA Driver API 6.30 (Tensor Map Object Management) |
| "How do I create/manage CUDA contexts?" | CUDA Driver API 6.8 (Context Management) |
| "What are green contexts / `cuGreenCtxCreate`?" | CUDA Driver API 6.35 (Green Contexts) |
| "How do I use virtual memory (`cuMemCreate`)?" | CUDA Driver API 6.14 (Virtual Memory Management) |
| "What device attributes can I query?" | CUDA Driver API 6.1 (`CUdevice_attribute`) |
| "What are `CUresult` error codes?" | CUDA Driver API 6.1 (`CUresult` enum) |
