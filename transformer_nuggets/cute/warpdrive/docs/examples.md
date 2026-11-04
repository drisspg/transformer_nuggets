# Examples

[Back to warpdrive](../README.md)

- [gemm.py](../examples/gemm.py): three roles and two rings, one CTA per output tile with
  `WorkStream(schedule=OneShot())`.
- [persistent_gemm.py](../examples/persistent_gemm.py): all four schedules, double-buffered
  accumulators, TMA stores, CTA pairs and configurable `tile_n`.
- [fa4_fwd.py](../examples/fa4_fwd.py): two softmax instances, a correction role, slots,
  fan-out/join, turnstiles, S/P aliasing, split-P, persistence and CTA pairs.
- [packed_work.py](../examples/packed_work.py): a mapping for a device-built work table with
  a host-visible capacity and device-side length.

## GEMM

```python
import torch
from transformer_nuggets.cute.warpdrive.examples.gemm import gemm

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
d = gemm(a, b)  # A @ B.T
```

To use a persistent two-CTA kernel:

```python
from transformer_nuggets.cute.warpdrive import ClusterLaunchControl, CtaPair
from transformer_nuggets.cute.warpdrive.examples.persistent_gemm import persistent_gemm

d = persistent_gemm(a, b, schedule=ClusterLaunchControl(), cluster=CtaPair())
```

Both wrappers take BF16 `a: (M, K)` and `b: (N, K)` on one CUDA device. Inputs must be
contiguous and 16-byte aligned, with K and N multiples of 8. `out`, if supplied, must be
BF16 `(M, N)` on the same device, contiguous, aligned and disjoint from both inputs.
Empty outputs return early; K=0 writes zeros.

`PersistentGemm` uses two accumulator stages by default so one tile's epilogue can overlap
the next tile's mainloop. Changing stage counts still has to fit SMEM and TMEM.

## Attention forward

```python
from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import fa4_fwd

out, lse = fa4_fwd(q, k, v)
```

This is a teaching kernel for unmasked attention, not a replacement for the full FA4 API.
Inputs are contiguous, 16-byte-aligned BF16 `(batch, seqlen, heads, 128)` tensors on one CUDA
device. K and V have matching shapes; all inputs have matching batch and head counts. Key
length must be a positive multiple of 128. Query length can be ragged. There is no varlen,
GQA or other head dimension. The example imports `flash_attn.cute` and `quack` helpers.

O has Q's shape. LSE is fp32 `(batch, heads, seqlen_q)` in natural-log units. The input checks
are not a complete validator of all these requirements; see the example-wrapper gap in
[GUARANTEES.md](../GUARANTEES.md#known-gaps).

`fa4_fwd(..., schedule=..., mapping=...)` lets you change index assignment and work order.
See [scheduling](scheduling.md) for `Raster` and the FA4 mappings. `roles=Fa4Roles(...)`
changes placement and register budgets; see [roles](roles.md#choose-placement-in-the-constructor).

K and V share channel-bound SMEM. The `FlashAttentionForward` constructor also accepts
`overlap_o_q=True` to reuse Q's bytes for epilogue SMEM; the `fa4_fwd` wrapper does not expose
this option. It requires a non-persistent schedule (`OneShot()` among the built-ins).
Reusing it across persistent work items needs an additional epilogue-to-load ordering edge;
sharing an address does not create that edge.

### Hardware row max on SM103

```python
from transformer_nuggets.cute.warpdrive import ClusterLaunchControl, CtaPair

out, lse = fa4_fwd(
    q, k, v, cluster=CtaPair(), schedule=ClusterLaunchControl(),
    split_p=True, use_ldred_rowmax=True,
)
```

`use_ldred_rowmax=True` uses `tcgen05.ld.red` to load scores and reduce each 32-column
fragment's maximum together. Four fragment maxima replace the software reduction of 128
scores. This is SM103-only; requesting it for an SM100 target raises `ValueError`. The
option defaults to false, so the existing software path and its codegen are unchanged on
both targets. The rescale threshold, exponentials, BF16 P conversion and row-sum arithmetic
are unchanged. The four hardware maxima feed the existing software updater, preserving its
fast-exp2 flush-to-zero rescale behavior. Tests compare O and LSE bitwise with the software
path, including ragged Q, nonuniform scales, repeated launches, static-persistent channel
wraps and a subnormal rescale amplified by large finite V. A generated-SASS check and a
public-wrapper kernel-name trace ensure that the hardware path actually runs.

Run the fixed-buffer, interleaved CUDA-graph comparison against FA4's default:

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 60s \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_fwd \
  --shape 0 --output agent_space/fa4_gap.json
```

Repeat with `--shape 1` and `--shape 2`. Shapes are `(batch, heads, query/key length)`:
`(4,16,4096)`, `(2,16,8192)` and `(8,16,1024)`, all BF16, head dimension 128, non-causal.
Each graph has 100 forwards. The script settles DVFS for three seconds, alternates variant
order over seven rounds, and saves all 15 replay samples per round. Numbers below are the
minimum round median, in microseconds per forward; setup/capture are excluded. FA4 timing
uses `flash_attn.cute.flash_attn_func(..., causal=False)` with its defaults; the example
also writes LSE. Inputs use seed 0, normal-distributed `torch.randn` values rounded to BF16.
Buffers are reused, not rotated for cold HBM timing.

GB300, CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, Triton `3.8.0+gitc01b6774`,
FA4 `00807a02`, warpdrive base `e4f3ede` (November 2026), normal DVFS without clock locking.
GPUs were shared; sub-percent differences are not stable wins.

| Forward | 4x16x4096 | 2x16x8192 | 8x16x1024 |
|---|---:|---:|---:|
| CLC split-P, software max | 342.56 | 645.15 | 57.19 |
| CLC split-P, hardware max | 336.67 | 632.54 | 56.70 |
| Static persistent split-P, hardware max | 337.49 | 634.53 | 54.28 |
| FA4 default | 336.79 | 634.74 | 55.21 |

| Change | Time delta (lower is better) | Kept? |
|---|---|---|
| Hardware max, same CLC/register split | -1.72%, -1.95%, -0.85% in shape order above | Opt-in |
| Static instead of CLC, with hardware max | +0.24%, +0.31%, -4.27% (repeat: -2.24% small shape) | Existing schedule option |
| Match FA4 registers: 176/80/80 instead of 192/80/48 | No consistent gain; +0.56% at 8x16x1024 | No default change |
| exp2 emulation, frequency 16 or 8 | +2.85% / +3.86% at 8x16x1024 (exploratory) | Removed |
| Registers 200/64/48 or 184/96/48 | +0.26% / +0.31% at 8x16x1024 (exploratory) | No default change |

Hardware max leaves CLC **-0.04%, -0.35%, +2.69%** relative to FA4. The small-shape gap is
mostly scheduling: this FA4 revision defaults to **static persistence**, not CLC. Matching
that schedule leaves **+0.21%, -0.03%, -1.69%** in this run, but the small-shape static minimum
was a transient low round. A separate repeat was **+0.37%** (55.34 vs 55.14 us), while CLC
was +2.67%. Static reaches approximate parity, not a repeatable advantage. These results do
not justify a shape-based schedule selector.

Matched NCU 2025.4.1 kernel-replay reports (`--cache-control none`, 16 passes) explain the
row-max change: at 4x16x4096, dynamic warp instructions fell from 128.43M to 112.41M,
with 128 launch registers/thread and 232,448 bytes dynamic SMEM unchanged. SASS replaced
264 `FMNMX` instructions with 10 and adds `LDTM.STAT.x32.MAX.F32`; the register split and
local accesses are unchanged. NCU duration fell 428.06 -> 416.80 us; it is a different
replay/cache/clock contract from graph timing, not the timing authority (default NCU clock
control, average GPC clocks about 1.20 GHz and memory clocks 4.00 GHz). DRAM traffic stays
near 201 MB read and 73 MB written: hardware max reduces instructions, not input traffic.

At 8x16x1024, CLC's grid has 512 CTAs while static and FA4 have 152, with capacity for
76 resident clusters of two CTAs. Static avoids the work-broadcast/CLC machinery:
NCU duration 79.62 -> 76.74 us, eligible warps per scheduler 0.370 -> 0.386
(FA4: 76.74 us, 0.426). DRAM read/write traffic stays near 100/39 MB and resources are
unchanged, so this is not a traffic or residency improvement. The same mbarrier stats
channel remains in the faster static example. Named-barrier stats and correction-warp rescheduling were not changed; these
measurements do not identify either as the remaining bottleneck.

For NCU, use `--profile` with `--profile-from-start off`: the script warms all variants,
then profiles one complete forward per variant between CUDA profiler start/stop. All
seven standard example configurations keep identical normalized SASS versus `origin/main`
on SM100 and SM103; only the opt-in hardware-max variant intentionally changes it.

## Historical timings

The [original README at 49b9713](https://github.com/drisspg/transformer_nuggets/blob/49b9713/transformer_nuggets/cute/warp_roles/README.md)
records the November 2026 GB300 GEMM/FA4 experiments. Those were BF16 CUDA-graph replay medians
with CuTeDSL 4.8.0, a PyTorch 2.15 nightly cu132 build and FA4 at `849b5daa`.

They are measurements of that revision, not of the storage-binding or work-API changes.
Keep the original shapes, schedule and reference implementation with the numbers when comparing;
a FA4 default schedule and a matched single-CTA schedule answer different questions.

## Still missing

Larger clusters, TMA multicast, round-robin role instances and cp.async producer mechanisms
are not supported. Using CUTLASS pipeline objects where their semantics match remains a
possible implementation cleanup, not a reason to change the role bodies.

The alias diagnostic still needs IR-inferred accesses, partial K-block footprints, CTA-pair
analysis and completion-aware epoch checks. Its current limits are in
[storage](storage.md#read-the-result).

For the normal and quick-fuzz commands, see [Tests](../README.md#tests). Long campaigns and
what their oracles cover are in [GUARANTEES.md](../GUARANTEES.md#fuzzing-and-evidence).
