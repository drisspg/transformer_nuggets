# Examples

[Back to warpdrive](../README.md)

- [gemm.py](../examples/gemm.py): three roles and two rings, one CTA per output tile with
  `WorkStream(schedule=OneShot())`.
- [persistent_gemm.py](../examples/persistent_gemm.py): all four schedules, double-buffered
  accumulators, TMA stores, CTA pairs, larger clusters with TMA multicast and configurable
  `tile_n`.
- [fa4_fwd.py](../examples/fa4_fwd.py): two softmax instances, a correction role, slots,
  fan-out/join, turnstiles, S/P aliasing, split-P, persistence and CTA pairs.
- [packed_work.py](../examples/packed_work.py): a mapping for a device-built work table with
  a host-visible capacity and device-side length.
- [fa4_decode_swapab.py](../examples/fa4_decode_swapab.py): FA4's swap-AB single-token decode:
  split-KV work items with per-sequence split counts, round-robin softmax warpgroups,
  named-barrier mailboxes and P staged through SMEM.

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

`cluster=Cluster(2, 2, pair=True)` runs two pairs side by side along N that multicast A
([clusters](clusters.md#larger-clusters)). With `tile_n=224` that is cuBLAS's 256x224 2x2
configuration for 3000x5000x4096.

`tile_n` defaults to 224 for `CtaPair()` and 256 otherwise. The width that wins is mostly
wave quantization: how the tile count divides into the resident CTAs (152 CTAs, 76 pairs or 36
2x2 clusters on a GB300) and how full the last wave is, against the better per-FLOP efficiency
of the full 256-wide tile once there are many waves. So no single width wins everywhere; the
defaults are a compromise over the shapes below. CUDA-graph replay, GB300, CuTeDSL 4.8.0,
November 2026, `StaticPersistent`, min of five interleaved rounds' medians of 60 replays
(microseconds, lower is better):

| M x N x K | `torch.mm` | 1 CTA 224 | 1 CTA 256 | pair 192 | pair 224 | pair 256 | 2x2 192 | 2x2 224 | 2x2 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 x 4096 x 4096 | 75.2 | 83.2 | 87.9 | 77.9 | 72.4 | 81.2 | 77.5 | 87.1 | 79.4 |
| 3000 x 5000 x 4096 | 72.9 | 83.4 | 85.6 | 77.3 | 72.8 | 81.1 | 77.4 | 71.8 | 79.6 |
| 2048 x 8192 x 4096 | 79.2 | 83.3 | 86.0 | 77.9 | 72.9 | 81.2 | 78.8 | 87.5 | 81.2 |
| 8192 x 2048 x 4096 | 76.0 | 100.8 | 88.7 | 80.1 | 89.4 | 81.2 | 90.0 | 86.2 | 79.5 |
| 4096 x 4096 x 14336 | 264.2 | 292.9 | 306.9 | 282.3 | 269.0 | 290.2 | 277.5 | 309.4 | 277.3 |
| 4096 x 14336 x 4096 | 244.7 | 282.3 | 265.9 | 258.3 | 262.3 | 253.6 | 263.8 | 264.2 | 260.8 |
| 8192 x 11008 x 4096 | 393.1 | 454.3 | 442.4 | 416.3 | 441.3 | 410.5 | 438.0 | 448.6 | 410.5 |
| 6000 x 3000 x 2048 | 43.1 | 50.7 | 47.5 | 48.4 | 47.8 | 44.4 | 48.9 | 47.3 | 44.7 |
| 1234 x 4096 x 4096 | 30.1 | 38.7 | 42.7 | 34.3 | 39.3 | 42.8 | 34.9 | 39.9 | 43.2 |
| 1024 x 1024 x 8192 | 18.4 | 38.0 | 42.0 | 33.9 | 38.0 | 42.0 | 34.0 | 38.1 | 42.2 |

Two sweeps (3 and 5 rounds, also with `tile_n=128` and `ClusterLaunchControl`) agree within
5% on most table shapes (8% on the two smallest) and rank the widths the same way under CLC; the shapes with
400 us and more (8192 x 11008 and the two left out of the table, 8192^3 and
16384 x 4096 x 14336) moved up to 10% between sweeps and up to 19% between rounds on the shared
GPU, so their single-digit differences mean little. Over the twelve shapes the pair at 224 is
1-3% faster than at 256 in geomean in each of the four sweep x schedule combinations (2.2%
pooled) and a repeatable 7-11% at the 4096-class shapes; 192 is 0.5-3.5% faster than 256
(1.7% pooled), with its gains on the noisy large shapes. One CTA ties between 224 and 256
(within 0.6%), so it keeps 256. For the 2x2 cluster 224 is the worst default (a 448-wide
cluster tile gives a 4096-wide N a fifth wave) and 192 beats 256 by only 1-3%, within that
noise, so it keeps 256. 128 only wins the 32-tile 1024 x 1024 x 8192 shape, by 24% over 192.
A rule on the tile count picks the winner for one- and two-wave problems but is 10-15% off at
many-wave ones, where the full tile's efficiency wins, so there is no shape-based selector:
pass `tile_n` for a shape you care about.

The harness is a CUDA-graph A/B: one graph per configuration, captured once, replayed in
round-robin rounds (`torch.mm` first, then the widths in order) with default clocks and warm
L2, reporting each configuration's best round median. Rerun it at your shapes before trusting
a difference under 5%.

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
are not a complete validator of all these requirements. A wrapper accepting an input does not
establish the advertised rank, head-dimension or device contract; enforce those requirements
when building another wrapper around this teaching kernel.

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

## Single-token decode

```python
from transformer_nuggets.cute.warpdrive.examples.fa4_decode_swapab import fa4_decode

out, lse = fa4_decode(q, k, v, num_splits=8, seqused_k=lengths, splits_per_seq=counts)
```

A port of FA4's `flash_fwd_decode_sm100.py`: S^T = K Q^T and O^T = V^T P^T, so 128 KV tokens
fill the MMA's M and the query heads of one KV head fill N. Q is `(batch, 1, heads, 128)`, K and
V `(batch, kv_len, kv_heads, 128)`, all BF16 and contiguous, with up to 32 query heads per KV
head. `seqused_k` (int32 per sequence) shortens sequences; KV rows past it must be finite, since
FA4's bounded V descriptor isn't ported. `splits_per_seq` caps each sequence's splits, as FA4's
prepare kernel does. `decode_partials` returns the fp32 split partials in FA4's combine layout;
`fa4_decode` combines them with FA4's combine kernel. No paging, varlen Q, masks or score mods.

The roles match FA4's placement: a QK and a PV MMA warp, a K/V load warp that also writes the
split's LSE, a Q-load/O-store warp, two softmax warpgroups alternating KV blocks and a
correction warpgroup. What it needed from the layer:

- Work items `(split, kv_head, batch)` from a `SplitKV` mapping. With per-sequence counts it
  reads the table after `PdlWait()` and drops the splits past each count.
- `round_robin` rings for S, P and the row-max/row-sum mailboxes, so softmax[i] takes blocks
  i, i+2, ... with one warpgroup's counts. The schedule stays `OneShot`: with an odd block
  count a second item would start on softmax[1]'s epochs.
- Named-barrier channels for the thread mailboxes and done flags, with a turnstile around the
  shared row-max exchange.
- P in a 4-stage SMEM ring read as the PV MMA's B operand; no TMEM alias.

FA4 reads one K/V ring with two MMA cursors, keeping K two blocks ahead of V with two extra named
barriers. Here K and V have their own rings, 7 and 5 stages at FA4's total of 12 (8-head tiles).
The other changes: the first PV into each O slot overwrites it instead of the correction
warpgroup zero-filling TMEM first, done flags also sync on a pre-seeded empty barrier, and an
empty split runs the normal tail with zero sums.

Split partials match FA4's bit for bit on every shape below, with FA4's own split count and
per-sequence table (captured at its combine call); the combined outputs are identical too.
The initial timings below are CUPTI medians of CUDA-graph replays, in microseconds; the better
of two interleaved rounds of 200 replays. Graph times also include FA4's prepare kernel (per-sequence
rows) and the shared combine. Buffers are reused, so short shapes run from L2. These initial
numbers used the earlier benchmark; the command below runs the updated method described
under decode setup costs.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_decode
```

GB300, CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`, November 2026,
shared GPUs.

| batch x kv_len, heads/kv_heads | splits | FA4 decode | port | ratio | graph FA4 / port |
|---|---|---:|---:|---:|---:|
| 1x8192, 8/1 | 16 | 4.22 | 4.70 | 1.114 | 16.40 / 16.69 |
| 1x32768, 32/1 | 32 | 7.04 | 7.90 | 1.123 | 19.39 / 19.18 |
| 4x16384, 32/8 | 4 | 39.81 | 41.18 | 1.035 | 49.02 / 49.09 |
| 8x32768, 32/8 | 2 | 151.10 | 149.70 | 0.991 | 159.41 / 157.63 |
| 8x65536 ragged, 8/1 | per-seq <= 32 | 26.24 | 28.18 | 1.074 | 36.80 / 36.80 |
| 16x32768 ragged, 64/8 | per-seq <= 10 | 180.38 | 180.37 | 1.000 | 190.30 / 190.11 |
| 2x131072 ragged, 16/1 | per-seq <= 32 | 9.28 | 9.47 | 1.021 | 21.41 / 21.38 |

In that initial run, long splits tied and short ones were 2-12% (0.2-1.9 us) slower. Under NCU
(1x8192, cold caches) the port issued 5% more instructions and spent more stall cycles on long
and short scoreboards. Its other resources matched FA4's: 68 registers, no spills, 512 threads.
Changes measured and ruled out as the cause, each within noise: allocating TMEM from warp 1 as
FA4 does, aligned instead of unaligned named barriers, K/V stage splits from 4:7 to 9:3
(FA4's shared ring has no direct equivalent), decoding the item and reading `seqused_k` once in
the prologue, skipping all setup for dropped splits (slower: it serializes on the PDL wait), and
prefetching the TMA descriptors (kept, as FA4 does).

### Decode setup costs

`SplitKV.bind()` now builds `FastDivmodDivisorV2` values for the launch's split/head mapping,
as FA4's scheduler does. The roles use them instead of repeating general integer division.
These depend only on shapes; the per-sequence split counts are still read after `PdlWait()`.
Batch, head counts and splits remain dynamic, and partials are still bitwise equal to FA4.

Matched before/after, GB300, same versions as above and Triton `3.8.0`, November 4, 2026.
The benchmark source SHA-256 prefix was `78d54f3cf2c58af0`, against the mapping at `c205c64`
and fast-divmod source `18088cc627d69597`. These source hashes were recorded before the
jit_cache migration; the three decode builds' normalized SASS stayed identical after it.
One GPU reservation covered three process pairs in A/B, B/A, A/B order, each with three
alternating FA4/port rounds of 200 replays after three seconds of warmup.
Fixed buffers, seed 0, normal-distributed BF16 inputs, normal DVFS without clock locking.
The table gives CUPTI decode-kernel medians across the process medians, in microseconds;
lower is better. Prepare and combine are excluded from these kernel times; whole-graph
samples are recorded too, but were noisy. No synchronization or layer code changed.

| batch x kv_len, heads/kv_heads | before | fast divmod | delta |
|---|---:|---:|---:|
| 1x8192, 8/1 | 4.863 | 4.704 | -3.27% |
| 1x32768, 32/1 | 8.001 | 7.872 | -1.61% |
| 4x16384, 32/8 | 41.055 | 40.433 | -1.52% |
| 8x32768, 32/8 | 149.984 | 149.090 | -0.60% |
| 8x65536 ragged, 8/1 | 28.223 | 28.065 | -0.56% |
| 16x32768 ragged, 64/8 | 180.673 | 180.193 | -0.27% |
| 2x131072 ragged, 16/1 | 9.600 | 9.279 | -3.34% |

To repeat the comparison, archive `c205c64`'s `transformer_nuggets` package into scratch space,
copy the current `bench_fa4_decode.py` into that archive, and run the same benchmark with
`python -P` and `PYTHONPATH` pointing at each source tree. Interleave both under one
`gpu-run` reservation. `--output` records the actual kernel source hashes, so the archive
(which has no Git checkout) is not mislabeled as the current commit.

Matched NCU 2025.4.1 at 1x8192 (`--cache-control none --clock-control none`, kernel replay,
12 passes) counts 121,600 -> 111,184 executed warp instructions (-8.57%). FA4 counts 117,200:
removing the extra instructions alone does not close the gap. SASS for the static 8-head
mapping falls from 2,992 to 2,664 instructions: two mapping divides in each of seven role
instances lose 14 `MUFU.RCP`, 14 `F2I` and 28 `IABS` sites. Both port builds still use
68 registers and 216,096 bytes dynamic SMEM, at 0.11 launch waves; no spills. NCU duration is
7.296 -> 6.912 us, under its separate replay contract. The fixed buffers fit in L2, and this
report's DRAM reads are nearly zero. Source-level PC sampling has too few samples to assign
hundreds of nanoseconds to one particular load or wait.

Sparse IKET ranges on all 16 CTAs and 16 warps localize the original port's delay. Both kernels
have a kernel-entry marker and four marker pairs per warp. The port reaches its post-prologue
CTA synchronization around 480 ns, versus FA4's 672 ns, relative to each CTA's first marker.
Its role bodies start at 736-928 ns: item decoding, TMEM pointer retrieval and role setup
happen after that synchronization. FA4 starts its dispatched work at 672 ns. The critical
O-store warp finishes around 3,904 ns in the port and 3,200 ns in FA4; the correction and
statistics also finish later. These are instrumented phase boundaries, not production latency
estimates. With body markers, median CTA lifetime envelopes are FA4/port 3,936/4,512 ns;
trampoline-only IKET (`--warp-trace`) gives 3,904/4,512 ns on the same GPU. Both outputs match
bitwise. This supports a post-sync setup and body/tail delay, not a slower
barrier-init/TMEM-allocation phase or a full critical-cycle model.

| Suspected cause | Evidence | Change tested | Time delta / decision |
|---|---|---|---|
| Repeated mapping divides | Source/SASS and -8.57% executed instructions | Host-built fast divmod | Kept; -1.6 to -3.3% on the three small cases above |
| TMEM base round trip | FA4 allocates all 512 columns and uses constant addresses | Full allocation and constant TMEM views | -0.5% at 1x8192; noise, removed |
| Q load after the CTA sync | FA4 starts Q TMA before synchronization | Prologue-owned raw Q barrier and early TMA | +5.9% at 1x8192; removed |
| Long-hint mbarrier waits | Port/FA4 execute 3,335/1,315 `NANOSLEEP.SYNCS` instructions in the first matched report | Try-wait first / hintless retry in channel waits | +0.4% / -1.1% at 1x8192; no general win, removed |
| O-staging handoff | Port uses an mbarrier ring; FA4 a one-shot named barrier | Raw one-shot named O-done barrier | -0.8% / -1.7% at 1x8192 / 1x32768; not kept |

The wait prototypes changed all seven standard configurations' SASS. At FA4 2-CTA CLC
split-P, try-first adds 144 static instructions; hintless retry removes 40 and removes 70
suspend sites. Neither is a general improvement: two interleaved runs made this FA4 case
0.7-2.3% slower. CLC persistent-GEMM results changed direction between runs; the static pair
was faster in one run but was not repeated. No layer patch was retained. The normal
seven-config SASS gate is unchanged. The raw O-done exchange's less-than-2% gain did not
justify a manually reserved barrier ID and giving up the managed epilogue handoff.

Run the benchmark with `--output` to save versions, source hashes, kernel names and every
replay sample. It now uses three seconds of warmup and seven alternating rounds, reporting
the median of round medians. In the matched A/B above, the updated port still trails FA4
by 7.4-9.3% on 1x8192, 1x32768 and 8x65536 ragged, and by 1.7% at 4x16384; long splits
approximately tie. The ragged gap and the remaining body/tail delay are not yet attributed
to a specific instruction sequence.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_decode \
  --output agent_space/decode.json
```

For matched NCU source attribution, add `--profile --shapes 0` and run with
`CUTE_DSL_LINEINFO=1 ncu --profile-from-start off --cache-control none --import-source yes`.
The script compiles and warms both paths before profiling one complete call each, including
prepare/combine where present; filter the reports to `attn_kernel|warpdrive_fa4_decode`.

Original code size, counting lines that are neither blank, comments nor docstrings: FA4's file
had 1,496, of which 1,347 were the kernel without the interface adapter, including paging,
masks, score mods and varlen Q. The initial port had 993, including its wrappers and mapping.

## Historical timings

The [original README at 49b9713](https://github.com/drisspg/transformer_nuggets/blob/49b9713/transformer_nuggets/cute/warp_roles/README.md)
records the November 2026 GB300 GEMM/FA4 experiments. Those were BF16 CUDA-graph replay medians
with CuTeDSL 4.8.0, a PyTorch 2.15 nightly cu132 build and FA4 at `849b5daa`.

They are measurements of that revision, not of the storage-binding or work-API changes.
Keep the original shapes, schedule and reference implementation with the numbers when comparing;
a FA4 default schedule and a matched single-CTA schedule answer different questions.

## Still missing

cp.async producer mechanisms are not supported. A ring read by several consumer cursors (FA4
decode's shared K/V ring) is not either; the decode example splits it into two rings. Clusters
cannot fill other CTAs' stages from a leader-only TMA producer or multicast to a `rank0` role.
Using CUTLASS pipeline objects where their semantics match remains a possible implementation
cleanup, not a reason to change the role bodies.

The alias diagnostic still needs IR-inferred accesses, partial K-block footprints, cluster
analysis and completion-aware epoch checks. Its current limits are in
[storage](storage.md#read-the-result).

For the normal and quick-fuzz commands, see [Tests](../README.md#tests). Long campaigns and
what their oracles cover are in the [fuzz harness guide](../../../../test/warpdrive_fuzz/README.md).
