# Examples

[Back to warpdrive](../README.md)

- [gemm.py](../examples/gemm.py): three roles and two rings, one CTA per output tile with
  `WorkStream(schedule=OneShot())`.
- [persistent_gemm.py](../examples/persistent_gemm.py): all four schedules, double-buffered
  accumulators, TMA stores, CTA pairs, larger clusters with TMA multicast and configurable
  `tile_n`.
- [grouped_gemm.py](../examples/grouped_gemm.py): MoE grouped GEMM with group sizes read on
  the device: a `WorkMap` that decodes `(group, rows, n_tile)` from the offsets, one TMA
  descriptor per operand across all groups and a ragged TMA store
  ([details](#grouped-gemm)).
- [fa4_fwd.py](../examples/fa4_fwd.py): two softmax instances, a correction role, slots,
  fan-out/join, turnstiles, S/P aliasing, split-P, persistence and CTA pairs.
- [fa4_bwd.py](../examples/fa4_bwd.py): FA4's SM100 1-CTA attention backward: 14 warps in
  four roles, eleven channels (seven with `Warps` sides), five MMAs per tile software-pipelined
  across overlapping handles, three TMEM aliases and two SMEM epilogue overlaps.
- [packed_work.py](../examples/packed_work.py): a mapping for a device-built work table with
  a host-visible capacity and device-side length.
- [fa4_decode_swapab.py](../examples/fa4_decode_swapab.py): FA4's swap-AB single-token decode:
  split-KV work items with per-sequence split counts, round-robin softmax warpgroups,
  named-barrier mailboxes and P staged through SMEM.
- [bench_overhead.py](../examples/bench_overhead.py): handoff and fixed-cost microbenchmarks
  against hand-written twins ([results](#warpdrive-fixed-costs)).
- [bench_persistent_gemm.py](../examples/bench_persistent_gemm.py): the persistent GEMM's
  schedules, clusters, tile widths and accumulator handoff against `torch.mm`
  ([results](#tile-boundaries)).

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

### Tile boundaries

Three things decide how much of a tile's epilogue hides behind the next tile's mainloop: the
MMA warp's `tcgen05.commit` publishes the accumulator asynchronously (the layer's `Umma`
side), two TMEM stages let it start the next tile in the other buffer, and the epilogue hands
a stage back as soon as its last subtile's `tcgen05.ld` has completed (`tcgen05.wait::ld`,
`early_release=True`, the default) rather than after that subtile's SMEM staging and TMA store
(`early_release=False`). In SASS the early release only moves the accumulator's arrive up to
the last `LDTM`; a single CTA's epilogue (`cp.async.bulk.wait_group` pacing included) is well
shorter than its mainloop at every shape below, so it mostly matters for CTA pairs at short K.

`bench_persistent_gemm.py --variants mm single single_late single_clc single_acc1 pair
pair_late pair_clc pair_clc_late pair_atomic pair_acc1 pair_n256 pair_n256_late pair_n256_clc`,
GB300, CuTeDSL 4.8.0, November 2026, min of five interleaved rounds' medians of 10 replays of
20 calls, microseconds (`clc` is `ClusterLaunchControl()`, `atomic` `AtomicPersistent()`,
`late` is `early_release=False`, `acc1` is `acc_stages=1`; the pair is `tile_n=224` unless
`256`). The shared GPU ran about 10% slower than for the table above (`torch.mm` 82.4 vs 75.2
at 4096^3), so compare within a table:

| M x N x K | `torch.mm` | 1 CTA | 1 CTA late | 1 CTA clc | 1 CTA acc1 | pair | pair late | pair clc | pair clc late | pair atomic | pair acc1 | pair 256 | pair 256 late | pair 256 clc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 x 12800 x 1152 | 18.6 | 21.5 | 21.5 | 21.6 | 22.5 | 23.3 | 23.5 | 23.2 | 23.1 | 23.1 | 26.2 | 21.0 | 20.9 | 21.2 |
| 2048 x 8192 x 1024 | 20.2 | 25.8 | 25.2 | 25.0 | 27.9 | 22.3 | 22.5 | 22.3 | 22.7 | 22.9 | 25.7 | 24.9 | 24.9 | 24.7 |
| 4096 x 4096 x 512 | 11.8 | 15.7 | 15.7 | 15.4 | 16.7 | 13.0 | 13.3 | 13.0 | 13.3 | 13.0 | 15.8 | 14.0 | 14.4 | 14.0 |
| 4096 x 4096 x 4096 | 82.4 | 94.3 | 94.6 | 96.9 | 96.7 | 81.2 | 81.5 | 86.2 | 86.3 | 82.1 | 85.6 | 90.8 | 90.4 | 90.3 |
| 3000 x 5000 x 4096 | 78.5 | 90.3 | 90.2 | 93.5 | 91.2 | 77.9 | 77.8 | 81.9 | 79.1 | 79.1 | 82.1 | 88.0 | 86.9 | 85.8 |

The second accumulator stage is worth 1-22% (most at 4096 x 4096 x 512 on a pair, where the
epilogue is as long as the 8-k-tile mainloop; least at K = 4096 on one CTA). Early release is worth 0-3% on pairs at short K
(repeatable at 4096 x 4096 x 512 in two sweeps for both pair widths) and within noise elsewhere,
including by NCU duration at 1024 x 12800 x 1152 (`--clock-control none`, five launches each:
single 22.0 vs 21.6, pair 256 20.8 vs 21.1, pair 224 22.8 vs 23.1, pair clc 24.4 vs 24.2 early
vs late, against cuBLAS's 128x256 2-CTA kernel at 19.4). The short-K shapes trail `torch.mm`
by 10-25% for wave quantization (1024 x 12800: 400 single tiles or 200 pair tiles over 148 SMs
are 2.7 waves) and the store-bound 4096 x 4096 x 512 (32 MB out), not for tile-boundary gaps.

Before this measurement `ClusterLaunchControl` and `AtomicPersistent` pairs ran 8-20% behind
`StaticPersistent` at short K (24.2-24.4 vs 22.4-22.6, 23.1 vs 21.2, 15.2-15.4 vs 12.7-12.8 us
on the first three shapes), and deeper work rings did not help. Per-role `%globaltimer` probes
at 4096 x 4096 x 512 (pair, CLC) found every role waiting 0.7-1.2 us per item not for the
scheduler (its ring slot was already full) but inside its own ring *release*: the pair ring
derived a cluster-scope `mbarrier.arrive`, a `MEMBAR.ALL.GPU` per receiving thread, while the
scheduler's claim took 0.26-0.29 us (CLC) or 0.32 us (atomic) and decode plus `st.async`
publication 0.32 us. The ring's payload is completed by bytes and its releases publish
nothing, so the layer now derives `remote_scope="cta"` for it ([scheduling](scheduling.md));
the probes then show releases of 0.03-0.13 us and the MMA's item period equal to static's.
With that, CLC ties static on the three short-K shapes and trails it by 0-6% at K = 4096 across
runs (the table's 86.2 vs 81.2 was the largest; earlier sweeps had 82.8 vs 81.8), which is
unexplained: neither the first-item path (the roles start within 0.1 us of static's) nor the
claim chain is on the critical path there. The CLC path's remaining per-item cost is the
0.6 us scheduler chain above, one work ring slot ahead of the slowest role.

## Grouped GEMM

```python
from transformer_nuggets.cute.warpdrive.examples.grouped_gemm import grouped_gemm

a = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)  # the experts' rows, concatenated
b = torch.randn(8, 2048, 7168, device="cuda", dtype=torch.bfloat16)  # one (N, K) weight per expert
offsets = torch.tensor([700, 700, 1500, 2100, 2100, 3000, 4000, 4096], device="cuda", dtype=torch.int32)
out = grouped_gemm(a, b, offsets)  # out[s_g:e_g] = a[s_g:e_g] @ b[g].T
```

`offsets` holds each group's end row (`torch._grouped_mm`'s `offs`) and is read only on the
device, so one compiled kernel serves every `G`, `total_M`, `N`, `K` and routing; groups may be
empty or any size. Rows past `offsets[-1]` are left untouched. Operand limits are those of
the GEMMs above. Keyword options: `schedule` (default `StaticPersistent()`), `cluster`
(`SingleCta()` or `CtaPair()`), `max_ctas`, `tile_n`, `ab_stages`, `acc_stages`, `c_stages`,
`epi_tile_n` and `roles`.

The host cannot count the groups' tiles, so `GroupedTiles` declares
`ceil(total_M / tile_m) + G` m-tiles, an upper bound (each group's ragged tail adds at most
one). The decode numbers the groups' real m-tiles back to back and drops only the indices
past the last one. Finding a tile's group then takes a prefix sum of the groups' tile counts,
which the mapping's [decode state](scheduling.md#decode-state) carries from item to item: each
decode resumes the walk where the previous one stopped, testing eight groups per round trip.

The state made exact tile numbering affordable, and that is what fixed static persistence.
The earlier stateless decode found a group by binary search over a padded space with one
unused slot per group. Those dropped slots were scattered through each unit's strided
sequence, so static units got unequal work (up to 40 items against an even 36 at G=128) and
drifted apart; static ran 10-27% slower than the broadcast schedules. Carrying the search's
group in a state alone changed nothing, because a unit's next index is usually several groups
on. With exact numbering every tile costs the same and static ties `AtomicPersistent` and
`ClusterLaunchControl` within 2%. It needs no scheduler warp or workspace, hence the default.

A and `out` use one TMA descriptor each, addressed at the item's row with
`cute.domain_offset`; B is a 3-D `(N, K, G)` view so the group is a TMA coordinate. The output
is stored through a ragged TMA view (a `(2^30, N, 2^31 - 1)` tensor whose row mode ends where
TMA clips and whose third mode selects the group's end row), so a tile that ends inside a
group is clipped by the TMA store like a tile at the tensor's edge.

CUDA-graph replay on a shared GB300 (`bench_grouped_gemm.py`, 20 calls per graph, min of
five interleaved rounds' medians, PyTorch 2.15 nightly cu132), skewed routing with 1/8 of the
experts idle; microseconds per call and TFLOP/s on the real rows. `grouped_mm` is
`torch._grouped_mm` (`F.grouped_mm` is the same aten op and measured within 1%), `mm loop` a
`torch.mm` per group, `Triton` Inductor's persistent TMA grouped GEMM template
(`max_autotune` with Triton as the only GEMM backend), `QUACK` FlexGEMM's QUACK backend with
an identity epilogue (its untuned pick, a 128 x 128 tile on a 2-CTA MMA with a dynamic
persistent scheduler) and `QUACK tuned` the same with Inductor's `tuned=True` shortlist;
`pair` is `CtaPair()`:

| G, tokens, N x K | `grouped_mm` | `mm loop` | `Triton` | `QUACK` | `QUACK tuned` | 1 CTA static | 1 CTA atomic | 1 CTA CLC | pair static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8, 4096, 2048 x 7168 | 115 (1049) | 121 | 102 | 106 (1135) | 122 | 95 (1268) | 94 | 92 | 116 |
| 8, 16384, 7168 x 2048 | 318 (1511) | 319 | 361 | 382 (1260) | 323 | 337 (1429) | 335 | 338 | 319 |
| 64, 8192, 4096 x 4096 | 397 (693) | 533 | 382 | 370 (742) | 381 | 378 (726) | 373 | 378 | 396 |
| 64, 32768, 2048 x 7168 | 788 (1220) | 936 | 808 | 890 (1081) | 837 | 767 (1255) | 787 | 783 | 810 |
| 128, 16384, 7168 x 2048 | 700 (687) | 860 | 661 | 681 (706) | 657 | 639 (753) | 628 | 624 | 688 |
| 128, 32768, 4096 x 4096 | 1065 (1032) | 1252 | 1119 | 1207 (911) | 1136 | 1091 (1007) | 1104 | 1102 | 1131 |

Small skewed groups waste rows in 128-row tiles for every kernel (1.5x the real rows at
G=64 with 8192 tokens and G=128 with 16384, 1.13-1.25x at 32768); a pair's 256-row tiles
waste about twice that, so `CtaPair()` leads only on the G=8 shape with N=7168.

The untuned QUACK kernel leads only at G=64 with 8192 tokens (by 1-2%), where its
64-row-per-CTA tiles waste the fewest rows, and trails 1 CTA static by 7-16% on the other
five. Tuning helps QUACK on the G=8 shape with N=7168 (a 256 x 256 tile on a 2-CTA MMA with a
2 x 2 cluster, so A is multicast across the two n-neighbours; it ties `pair` there) but hurts
on the smallest: Inductor benchmarks the shortlist with evenly spaced offsets, so the winner
is picked for balanced routing. FlexGEMM's TRITON backend without `max_autotune` lowers to
the extern `_grouped_mm`, and its NVGEMM backend has no grouped GEMM choice
(`NoValidChoicesError`). Like ours, QUACK's varlen scheduler numbers the groups' tiles
exactly with a prefix sum it caches across tiles (a warp prefix sum over the offsets). It
also rasters a group's tiles in swizzled bands for L2 reuse, which shows up as no gap in
these timings.

`offsets` must be nonnegative, nondecreasing and at most `total_M`, and must not change while
the kernel runs; the kernel clamps each group's end to the previous end and to `A`, so other
values compute wrong rows but stay in bounds. `out` must not overlap `a`, `b` or `offsets`.

## Attention forward

```python
from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import fa4_fwd

out, lse = fa4_fwd(q, k, v)
```

This is a teaching kernel for unmasked attention, not a replacement for the full FA4 API.
Inputs are contiguous, 16-byte-aligned BF16 `(batch, seqlen, heads, head_dim)` tensors on one
CUDA device, head dimension 64 or 128 (the default tiling; 2 x (`tile_n` + `head_dim`) TMEM
columns must fit in 512). K and V have matching shapes; all inputs have matching batch and
head counts. Key length must be a positive multiple of `tile_n` (128, or 64). Query length can
be ragged. There is no varlen or GQA. The example imports `flash_attn.cute` and `quack`
helpers.

O has Q's shape. LSE is fp32 `(batch, heads, seqlen_q)` in natural-log units. The input checks
are not a complete validator of all these requirements. A wrapper accepting an input does not
establish the advertised rank, head-dimension or device contract; enforce those requirements
when building another wrapper around this teaching kernel.

`fa4_fwd(..., schedule=..., mapping=...)` lets you change index assignment and work order.
See [scheduling](scheduling.md) for `Raster` and the FA4 mappings. `roles={...}` changes
placement and register budgets by role name; see
[roles](roles.md#choose-placement-in-the-constructor). `tile_n=64` halves the K/V tile,
`rescale_threshold` sets FA4's lazy-rescale threshold, and head dimension 64 or 128 follows
the tensors; [configuration](configuration.md) lists every knob and what stays fixed.

K and V share channel-bound SMEM. The `FlashAttentionForward` constructor also accepts
`overlap_o_q=True` to reuse Q's bytes for epilogue SMEM; the `fa4_fwd` wrapper does not expose
this option. It requires a non-persistent schedule (`OneShot()` among the built-ins).
Reusing it across persistent work items needs an additional epilogue-to-load ordering edge;
sharing an address does not create that edge.

### Hardware row max on SM103

```python
from transformer_nuggets.cute.warpdrive import ClusterLaunchControl, CtaPair

out, lse = fa4_fwd(
    q,
    k,
    v,
    cluster=CtaPair(),
    schedule=ClusterLaunchControl(),
    split_p=True,
    use_ldred_rowmax=True,
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

## Attention backward

```python
from transformer_nuggets.cute.warpdrive.examples.fa4_bwd import fa4_bwd

dq, dk, dv = fa4_bwd(q, k, v, out, dout, lse, causal=False)
```

A port of the main kernel of FA4's `flash_bwd_sm100.py` at its 1-CTA configuration for head
dimension 128 (FA4 selects it with `disable_2cta`; its default on SM100 is the 2-CTA kernel).
FA4's own preprocess (`dPsum = rowsum(dO * O)`, `LSE * log2(e)`, zeroing the fp32 dQ
accumulator) and dQ postprocess kernels run around it unchanged. Inputs are contiguous,
16-byte-aligned BF16 `(batch, seqlen, heads, 128)` tensors on one CUDA device, with matching
batch and head counts (no GQA), and the forward's fp32 `(batch, heads, seqlen_q)` LSE.
Sequence lengths are arbitrary: TMA zero-fills partial tiles, keys past `seqlen_k` are masked
and the dK/dV stores are clipped. `causal=True` is FA4's bottom-right-aligned mask. No varlen,
paging, block sparsity, score mods, softcap or deterministic mode.

A work item is one 128-key tile, in FA4's `SingleTileScheduler` order, which `Raster()` over
`(n_blocks, heads, batch)` under `OneShot()` reproduces. The roles keep FA4's warp layout:
dQ reduce on warps 0-3, two compute warpgroups sharing each tile on 4-11, MMA on 12 and load
on 13. The [module docstring](../examples/fa4_bwd.py) has the channel table, the TMEM map and
the per-tile MMA order.

### Correctness

On every shape below and in the tests, dK and dV are bit for bit FA4's 1-CTA backward (same
inputs, same config), non-causal and causal, including ragged lengths. dQ differs only by the
order in which CTAs' TMA reduce-adds land in the fp32 accumulator (max |diff| up to 9.8e-4
against values of order 1; FA4 is not bitwise reproducible there either, and some
runs are bitwise equal). All three are checked against an fp32 autograd reference, within
FA4's own error. dK/dV are deterministic across launches; dQ is not, as in FA4.

### Timings

Fixed-buffer CUDA-graph replays of one whole backward call (preprocess, main kernel,
postprocess) per variant, seven interleaved rounds of 30 replays after three seconds of
warmup; CUPTI gives the main kernel. Minimum round medians in microseconds, lower is better.
GB300, CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`, normal DVFS,
shared GPUs, November 2026; `fa4_bwd.py` SHA-256 prefix `ceb4035159cd5e55` (this commit's
kernel, measured before the rebase that only moved its base):

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=10s 3000s \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_bwd \
  --output agent_space/bwd.json  # add --causal for the second table
```

| batch x heads x seqlen_q x seqlen_k | port | FA4 1-CTA | FA4 default (2-CTA) | port / 1-CTA | port / default | graph port / 1-CTA |
|---|---:|---:|---:|---:|---:|---:|
| 2x16x1024x1024 | 42.4 | 41.6 | 40.3 | 1.019 | 1.053 | 1.031 |
| 4x16x4096x4096 | 972.9 | 967.8 | 893.8 | 1.005 | 1.089 | 1.005 |
| 1x16x8192x8192 | 953.9 | 943.6 | 853.6 | 1.011 | 1.117 | 1.011 |
| 4x16x3000x2900 | 536.1 | 531.4 | 524.6 | 1.009 | 1.022 | 1.008 |
| causal 2x16x1024x1024 | 37.3 | 36.5 | 36.1 | 1.020 | 1.033 | 1.000 |
| causal 4x16x4096x4096 | 563.5 | 552.2 | 509.6 | 1.020 | 1.106 | 1.018 |
| causal 1x16x8192x8192 | 550.6 | 536.7 | 479.7 | 1.026 | 1.148 | 1.023 |
| causal 4x16x3000x2900 | 338.0 | 330.6 | 311.5 | 1.022 | 1.085 | 1.018 |

The port trails the matched FA4 kernel by 0.5-2% non-causal and 2-2.6% causal. Repeated in-process A/Bs of 4x16x4096 put it
between +0.7% and +3.6% from run to run, while FA4's minimum stayed within 968-975 us. The
2-CTA default is a different kernel (pair MMAs, a DSMEM dS exchange) and is 3-14% faster
than either 1-CTA kernel; it is not ported.

What was measured, 4x16x4096 unless noted (NCU 2025.4.1 kernel replay, `--cache-control
none`, and interleaved in-process CUPTI A/Bs):

| Change | Effect | Kept? |
|---|---|---|
| One elected arrival per warp (`Warps`) instead of 256 per-thread arrivals per release | NCU cycles 1.889M -> 1.862M (FA4 1.822M); stall long-scoreboard 6.6 -> 5.8 warps per issue | Yes |
| dV/dK published right after their last MMA, as FA4 does, instead of after the item's last dQ MMA | -1.7% at 2x16x1024, -2.8% at 4x16x4096 (minimum round medians) | Yes |
| S_0 waits only for K; V before the first dP (FA4 arms V with dO_0) | -0.9% at 2x16x1024 | Yes |
| FA4's `gemm_ptx_w_idx` for the SMEM/TMEM MMAs instead of `cute.gemm` | Within noise | Yes (FA4's code) |
| No peek wait before S_{t+1} (timing-only, incorrect) | Within noise | Peek kept |
| One release fence for dPsum and dS instead of two | Within noise | No |
| FA4's two extra compute-wide barriers before the S and dS signals | +0.6% | No |
| Dropping the second dS-over-dP barrier | +6% | No |
| In-graph workspace zero fill (`launch_args` before the placeholder fix) | +4 us whole-graph at 2x16x1024 | Fixed in the layer |

With `Warps` the port executes 1.6% more warp instructions than FA4 (285M vs 280M: the
`bar.warp.sync` and elect per signal; with per-thread arrivals it executed 3% fewer), with
the same opcode mix otherwise, 128 launch registers and 231 vs 232 KB of SMEM. The remaining
excess is in mbarrier wait stalls (about +4,000 of 66,700 PC samples, at the barrier-pointer
lines both kernels' waits are attributed to), not in issued work; which wait carries it is not
attributed. Swapping `Threads` for `Warps` in the forward, decode and persistent GEMM examples
moved them by -0.8% to +2.3% (outputs bitwise equal), so `Threads` stays their default.

### What it needed from the layer

- `Warps` channel sides, added for this port (above).
- `fence_async_shared()` had never run under tracing: the event's `kind` field collided with
  the recorder's parameter. Fixed.
- `launch_args()` zeroed a per-stream workspace even for `OneShot`, so graphs captured on a new
  stream replayed a memset. Fixed: one untouched placeholder per device.
- [Previews](channels.md#previews): S_{t+1} needs Q_{t+1} while Q_t's handle stays open for
  dK_t. FA4 rotates `handle_Q = handle_Q_next`; a handle carried into the next loop iteration
  would keep the pre-loop handle's trace record, so the checker would report it closed twice.
  The port first waited on the next stage's full barrier itself (`q.full_barrier(q.index)`,
  `q.phase()`) and read it through `params.sQ`, outside any handle. It now takes
  `q.peek()` and reads `preview.view("Q", read=True, by="umma")`; the next tile's `q.wait()`
  (the peeled tile's, for the loop's last preview) owns that epoch. Both modes' SASS is
  identical to the raw wait's, so the timings above stand, and the alias report is unchanged.

Dropped down, each documented at its site:

- The last tile's S acquire has no S_{t+1} to publish and is retired. That, the dK/dV
  epilogue tiles over Q and dO, and `Q_STAGE = 2` (one Q stage deadlocks: S_{t+1} waits for
  a load that needs Q_t's release) are protocol facts the layer cannot check; the kernel stays
  `OneShot`, as FA4's 1-CTA kernel does.
- dQ reduce and dK/dV epilogues order their SMEM staging with declared `NamedBarrier`s (allocated
  IDs 3-6, the hand-picked IDs they replaced) and `bulk_commit_group`/`bulk_wait_group`, as in
  the persistent GEMM epilogue.

Diagnostics that stay incomplete:

- The alias diagnostic reports P~S, dS~dP, dQ~dP and dQ~dS as `unverified` (unknown epoch
  correspondence), not as potential hazards: the channel round trips do order them, but the
  graph matches epochs only when the traced per-item counts agree, and the MMA role peels its
  last tile, so its s/dp/dq/ds counts (two unrolled iterations and the peeled tile) differ from
  the compute and reduce roles' (two iterations). The report names the channels and counts;
  no warning is emitted. Matching those epochs needs loop bounds and peeled iterations in the
  model (not loop counts chosen to balance the channels, which could hide a missing
  publication or release). The compute role's TMEM accesses declare their exact columns, so its
  cross-warpgroup ordering is checked.

### Size

Counting lines that are neither blank, comments nor docstrings: `fa4_bwd.py` is 924 lines,
including its tensor wrapper, compile helper and input checks; compiling it executes 864 of
them. Compiling FA4's 1-CTA causal backward executes 2,679 of `flash_bwd_sm100.py`'s 4,448 code
lines (the rest are the 2-CTA, varlen, GQA, block-sparse, score-mod, deterministic and folded
paths), not counting the pipeline, mask and copy helpers both use.

## Single-token decode

```python
from transformer_nuggets.cute.warpdrive.examples.fa4_decode_swapab import fa4_decode

out, lse = fa4_decode(q, k, v, num_splits=8, seqused_k=lengths, splits_per_seq=counts)
```

A port of FA4's `flash_fwd_decode_sm100.py`: S^T = K Q^T and O^T = V^T P^T, so 128 KV tokens
fill the MMA's M and the query heads of one KV head fill N. Q is `(batch, 1, heads, d)`, K
`(batch, kv_len, kv_heads, d)` and V `(batch, kv_len, kv_heads, d_v)`, all BF16 and contiguous,
with up to 32 query heads per KV head; `d` is a multiple of the K step (64 by default) and
`d_v` of 128 (128/128 by default; 256 works for either).
`seqused_k` (int32 per sequence) shortens sequences; KV rows past it must be finite, since
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
- Named-barrier channels for the thread mailboxes, with a turnstile around the shared row-max
  exchange, and three one-shot `NamedBarrier` done flags for the split statistics.
- P in a 4-stage SMEM ring read as the PV MMA's B operand; no TMEM alias.

The KV block (`tile_n`, 256), K step (`tile_k`, 64), S and P ring depths (4 and 4), head
dimensions and the roles' register budgets are constructor arguments of `SwapABDecode`;
`decode_partials` and `fa4_decode` forward them as keywords. The measurements below are at the
defaults, which match FA4; see [configuration](configuration.md) for the valid ranges.

FA4 reads one K/V ring with two MMA cursors, keeping K two blocks ahead of V with two extra named
barriers. By default K and V have their own rings here, 7 and 5 stages at FA4's total of 12 (8-head
tiles); `shared_kv=True` selects FA4's shared ring (see [below](#shared-kv-ring)).
The other changes: the first PV into each O slot overwrites it instead of the correction
warpgroup zero-filling TMEM first, and an empty split runs the normal tail with zero sums.

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
FA4 does (later measured at -0.4 to -0.7% and made the default; see
[fixed costs](#warpdrive-fixed-costs)), aligned instead of unaligned named barriers, K/V stage splits from 4:7 to 9:3
(FA4's shared ring has no direct equivalent), decoding the item and reading `seqused_k` once in
the prologue, skipping all setup for dropped splits (slower: it serializes on the PDL wait), and
prefetching the TMA descriptors (kept, as FA4 does).

[Try tokens](channels.md#try-tokens) were measured on the split rings as FA4 uses them, in
`fa4_decode_swapab.py` as of `5c8db04` (SHA-256 prefix `5aac3058290b9a8f`) behind a temporary
constructor option, not kept: the PV
warp probes V before its P wait and O acquire, consumes the token at the first V wait and
re-probes after each V release; the QK warp probes K before its S acquire and likewise. An
"all" variant also probed P, O and S as FA4 does. Partials stayed bitwise equal to FA4's in
every variant. Decode-kernel CUPTI medians of graph-replayed `decode_partials` (nine or eleven
rotating rounds of 200 replays after three seconds of warmup, three processes on one GB300,
CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`, November 5, 2026), in
microseconds, median over the processes; lower is better:

| batch x kv_len, heads/kv_heads | no tokens | K/V tokens | delta |
|---|---:|---:|---:|
| 1x8192, 8/1 | 4.289 | 4.352 | +1.5% |
| 4x16384, 32/8 | 40.256 | 40.352 | +0.2% |
| 8x32768, 32/8 | 148.368 | 148.448 | +0.1% |
| 8x65536 ragged, 8/1 | 28.065 | 28.064 | 0.0% |
| 16x32768 ragged, 64/8 | 179.810 | 179.794 | 0.0% |
| 1x32768, 32/1 | 7.360 | 7.328 | -0.4% |
| 2x131072 ragged, 16/1 | 8.928 | 8.928 | 0.0% |

The K/V tokens measured 4.352 us at 1x8192 in all three processes, while the baseline's
process medians ranged over 4.288-4.416 us. 4x16384 and 8x32768 were 0.01-0.3% slower in
every process; the remaining shapes moved by at most one 32 ns CUPTI tick, in both
directions. The "all" variant, measured in two of the processes, moved within 0.4% of the baseline
except at 1x8192 (+1.5% in one process, -3.6% in the other). The ragged case, where the shared-ring
prototype below gained 0.06 us from its K/V prechecks, gained nothing on split rings. Not
neutral-or-better on the smallest shape, so the example keeps blocking waits. The consumer
tokens never cross a loop iteration (FA4's loader carries its producer token across blocks;
that was not tried).

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
approximately tie. [Where the remaining gap was](#where-the-remaining-gap-was) locates that
delay and narrows it.

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

### Where the remaining gap was

`clock64` stamps from lane 0 of every warp, in instrumented copies of both kernels (1x8192,
8/1, 16 splits, partials still bitwise equal), put the port's Q-store finish about 970 cycles
behind FA4's. About 550 cycles came at the head: the port reads `seqused_k` after the CTA
sync and issues Q's TMA after that, while FA4 has a static length and issues Q before its
sync. The other ~450 came in the correction tail: the port waited for both final O slots
before its row-sum/1/L round trip with the stats warp, FA4 does the round trip first. The
stamps perturb the port more than FA4 (+384 ns versus +128 ns uninstrumented), so they
locate phases but do not price them; the kernel times below do.

The correction now runs the round trip first, and one CTA allocates TMEM off the
barrier-init warp (see [fixed costs](#warpdrive-fixed-costs)). Decode-kernel CUPTI medians,
in microseconds, from a scratch harness that graph-replays each decode kernel alone (fixed
buffers, seed 0, 200 replays per round, nine rounds in rotating order after three seconds of
warmup, normal DVFS) and checks partials bitwise against FA4's. GB300, November 4, 2026,
CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`; "before" is the port at
`47a1183` (the allocator change), "now" at `e4faad3`, both in one process:

| batch x kv_len, heads/kv_heads | FA4 | port before | port now | now / FA4 |
|---|---:|---:|---:|---:|
| 1x8192, 8/1 | 4.224 | 4.672 | 4.447 | 1.053 |
| 1x32768, 32/1 | 7.072 | 7.936 | 7.361 | 1.041 |
| 4x16384, 32/8 | 39.87 | 40.61 | 40.23 | 1.009 |
| 8x65536 ragged, 8/1 | 26.27 | 28.32 | 28.03 | 1.067 |
| 2x131072 ragged, 16/1 | 9.184 | 9.216 | 8.928 | 0.972 |

Only columns of one run compare: on another GPU the same 1x8192 build measured 4.768 us.
Tried and not kept, with the tail order in place: a static
length without `seqused_k` (no load before the first TMA, as FA4 does) moved 1x8192 by one
CUPTI tick (-31 ns) and 1x32768 by +31 ns; computing the split range in the kernel prologue
added nothing on top. Before the allocator change, both had been worth about 2%. Try-wait-
first channel waits, patched in for every wait of the port, changed nothing.

### Per-instruction attribution of the ragged gap

The remaining `8x65536`, `8/1` ragged gap is mostly the K/V stage-ownership and forward-progress
protocol, not barrier initialization or correction arithmetic. A scratch port using FA4's shared
K/V ring and two skipping MMA cursors closes most of it. It is not the example's default: the
same change regresses other shapes, and currently needs raw synchronization.

Matched November 4, 2026 captures used GB300, CuTeDSL 4.8.0, PyTorch
`2.15.0.dev20260922+cu132`, Triton 3.8.0, FA4 `00807a02`, and the port source SHA-256 prefix
`2a592081cfa1db52`. The committed benchmark source prefix was `949e2f06ee0a44cd` (before the
stateful-WorkMap migration; the seven standard configurations' normalized SASS stayed identical
through that migration). NCU 2025.4.1 collected `SourceCounters`, `WarpStateStats`,
`InstructionStats` and `LaunchStats`, with source import, kernel replay, no cache flushing and
no clock locking. Both paths used the same inputs and FA4 split table.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  env CUTE_DSL_LINEINFO=1 TRANSFORMER_NUGGETS_CUTE_CACHE_DIR=$PWD/agent_space/decode-cache \
  ncu --profile-from-start off --replay-mode kernel --cache-control none --clock-control none \
  --section SourceCounters --section WarpStateStats --section InstructionStats \
  --section LaunchStats --warp-sampling-interval 0 --warp-sampling-max-passes 10 \
  --import-source yes -k 'regex:attn_kernel|warpdrive_fa4_decode' -o agent_space/decode-source \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_decode \
  --profile --profile-reps 5 --shapes 3 --output agent_space/decode-profile.json
ncu -i agent_space/decode-source.ncu-rep --page source --print-source sass --csv
```

`--profile-reps` alternates FA4/port and port/FA4 pairs after three seconds of warmup;
`--output` records source hashes, split counts and launch order, not NCU timings. Read the
report's effective sampling interval: requesting interval 0 did not always produce 32-cycle
sampling. The ragged repeat used two aggregated passes and 128 or 256 cycles. The following
means are normalized to 256-cycle, two-pass equivalent samples, not nanoseconds. Static PCs
include out-of-line wait stubs assigned to the phase they return to. FA4 shares its softmax
body between warpgroups; the port's counts combine its two copies.

| Phase | static PCs FA4 / port | equivalent samples FA4 / port | dominant reason |
|---|---:|---:|---|
| Entry, mapping and role setup | 370 / 619 | 1830 / 1980 | memory dependencies, instruction fetch, barriers |
| KV load | 586 / 435 | 806 / 914 | long scoreboard |
| PV MMA, including its waits | 290 / 367 | 831 / 1260 | long scoreboard, barriers |
| Softmax S wait | 10 / 10 | 5344 / 5494 | long scoreboard |
| Softmax P acquire | 3 / 6 | 14 / 83 | long scoreboard |
| Correction loop | 165 / 157 | 2720 / 2862 | barriers, long scoreboard |
| Correction final O wait/load | 37 / 93 | 41 / 106 | long scoreboard |
| Epilogue wait/store | 58 / 63 | 875 / 884 | FA4 named barrier; port mbarrier |

The source view does not export every sampled PC: the repeat leaves 1381/1630 equivalent
samples uncorrelated. The full phase accounting retains that residual. Source-instrumented
instruction counts also differ from hardware counts collected in another pass. Neither is
an additive model of kernel latency, and parked epilogue warps are not a new critical path.

Native `enable_iket(waits=True)` captures, joined by logical CTA, warp and channel epoch, show
the dependency behind the V stalls. All 76 kept CTAs have 16 blocks; both kernels' wait
cardinalities match. In a warmed representative capture, the port's mid-loop loader V acquires
sum to about 0.5 us per block, versus FA4's 0.16–0.19 us. The port's PV warp waits about
0.7–0.9 us on V; FA4 waits about 0.45–0.59 us at QKIssued first, then roughly 0.16 us on V.
Those waits partly relocate the same dependency, so their widths cannot be subtracted as
extra work. The port's last V-ready observation is about 0.8 us later and its correction and
O-store finishes about 0.9 us later, relative to each CTA's first warp entry.

The stage-lifetime join explains the difference: a split V-ring acquisition often starts
before the prior PV reader is ready. In the warmed capture this occurs for 22.5% of reused
V stages, versus 4.3% in FA4, whose V loads mostly reuse K stages released by QK. The port's
feedback path is PV -> V-stage reuse -> loader -> subsequent K/QK -> V/PV. FA4's shared
capacity and QKIssued/PVIssued rendezvous change that path. The median V consumer-read-ready
to empty-barrier reply interval is similar in both, about 0.2 us; this is not a different
mbarrier implementation.

IKET widths remain perturbed. Seven alternating warmed eager captures, excluding initial
correctness launches, give minimal/native-waits kernel warp-lifetime envelopes of
28.096/28.992 us for the port and 26.368/27.904 us for FA4 on the same GPU. These are separate
from stabilized CUPTI graph timing. `plan_from_warpdrive` joins the seven role ranges, but its
work-item-granular plan promotes the within-item stats/epilogue feedback to the next work
item; its RecMII is not a per-block model of this OneShot decode. The raw per-warp wait
timelines and explicit stage epochs supply the causal evidence instead.

Two ablations rule out tempting alternatives. Moving the PDL wait and validity test before
setup really reduces TMEM allocations from 256 to 76 and hardware warp instructions from
2.403M to 2.117M, but changes CUPTI time by only -32 ns. Keeping the identical 76 partials
while capping the allocated splits from 32 to 16 removes 128 dropped CTAs for -97 ns. A
uniform-count control with no dropped CTAs still has a 5.1% port gap. Increasing K/V from
7/5 to 8/4 regresses the ragged case by 2.1%; blocking on V before P/O changes nothing.

The shared-ring prototype keeps the math and partials bitwise. It uses one TMA producer and
two exclusive UMMA consumers: each cursor skips the other's epochs, with FA4's two-block
K lookahead. The full/empty counts are one, not a join of both readers. A raw-done split-ring
control isolates the three one-ID OneShot stats flags needed to free named-barrier IDs.
The table is a single five-way isolated-decode comparison: nine rotating rounds of 200
fixed-buffer graph replays after three seconds of warmup, CUPTI medians in microseconds.
Shared ring depth equals the original K+V depth (12 for the 8-head tiles). No row may be
compared with a different run's table above.

| batch x kv_len, heads/kv_heads | FA4 | split default | raw-done split | shared, no KV precheck | shared, KV precheck |
|---|---:|---:|---:|---:|---:|
| 1x8192, 8/1 | 4.544 | 4.512 | 4.512 | 4.672 | 4.640 |
| 4x16384, 32/8 | 40.257 | 40.368 | 40.352 | 40.737 | 40.736 |
| 8x32768, 32/8 | 152.559 | 148.530 | 148.625 | 150.498 | 150.337 |
| 8x65536 ragged, 8/1 | 26.849 | 28.033 | 27.936 | 27.136 | 27.073 |
| 16x32768 ragged, 64/8 | 180.065 | 179.969 | 179.986 | 180.529 | 180.354 |
| 1x32768, 32/1 | 7.424 | 7.520 | 7.488 | 7.456 | 7.328 |
| 2x131072 ragged, 16/1 | 9.312 | 9.056 | 9.058 | 9.568 | 9.504 |

For the ragged case, the one-shot flags account for 0.097 us, shared capacity/rendezvous
without KV prechecks for another 0.800 us, and prechecks for 0.063 us. Together they close
81% of this run's 1.184 us gap, leaving 0.224 us unexplained. A matched source capture reduces
the four V-wait branch samples from 592 to 252 and correction O-wait branch samples from
1264 to 1054 (FA4: 1039), despite increasing hardware instructions by 2.9%. NCU duration
sometimes disagrees with the CUPTI direction; it did not select the winner.

The unchanged committed benchmark confirms the direction through the usual wrappers (FA4:
prepare/decode/combine; port: decode/combine), still timing only the decode kernel with CUPTI.
Three process pairs in A/B, B/A, A/B order on one GPU,
each with seven alternating FA4/port rounds of 200 replays, give ragged split/shared medians
27.487/26.592 us (-3.26%; FA4 26.239 us). That closes about 72% of the 1.248 us gap under this
contract. The 32-head small case improves from 7.616 to 7.360 us, while other cases regress
by up to 5.4%. All seven shapes pass the partial-bitwise assertions in every process. These
wrapper measurements and the isolated-kernel table are different contracts, not one pooled
sample set.

The example now declares those three flags as one-shot `NamedBarrier`s (correction arrives,
the load warp syncs, or the reverse for `stats_ready`), cutting its named-barrier IDs from 12
to 9 of 13. With the committed benchmark in A/B/A/B processes on one GB300 (seven rounds of
200 replays each), the port's decode kernel moved by -0.43% to +0.04% across the seven shapes
(mean of two processes per side; ragged 8x65536: 27.392 -> 27.361 us). Partials stayed
bitwise equal to FA4's in every process.

The missing layer capability was a shared TMA ring with several independent, exclusive MMA
cursors, explicit skipped epochs and a bootstrap/skew contract preventing parity aliasing.
`Slots` requires Python-int slot IDs and only advances a reader's visited slots; it does not
supply that dynamic ownership protocol. [Exclusive consumers](channels.md#exclusive-consumers)
now provide the ring and the skips; the skew contract stays the author's, as the rendezvous
recipe there shows. The raw prototype, source phase tables, matched Perfetto tracks, failed
ablations and reproduction scripts are retained in `agent_space/decode_sass/`.

### Decode racecheck

The four reports from the split and shared-K/V decode are **false positives on the row-sum
publication**, not four different races. Rechecked with source-line info on GB300,
compute-sanitizer 2025.4.1 (CUDA 13.1), driver 580.126.20, CuTeDSL 4.8.0 and torch
2.15.0.dev20260922+cu132, using the example at `0535f67`:

- Dense: `(batch, KV length, heads, KV heads, splits) = (2, 1024, 8, 1, 1)`.
- Per-sequence: `(3, 2048, 8, 1, 4)`, lengths `[1, 700, 2048]`, split counts `[1, 2, 4]`.
- Two launches of each shape. Each launch produces one RAW analysis group, so the summary
  is `4 hazards displayed (4 errors, 0 warnings)` for either build.

All four groups map to the same two source accesses in `fa4_decode_swapab.py`: correction
line 960 writes `sRowSums[warp, h]` from lane 0 of warps 12-15; `split_stats` line 610 reads
`sums[0..3, lane]` from load warp 2, lanes 0-7. The four byte hazards per float are not four
independent bugs. In this build, eight scalar stores feed one vectorized `LDS.128` per head:

| build / shape (both launches) | read PC | write PCs (all line 960) |
|---|---|---|
| split / dense | `0x4080` | `0x8d20, 8d40, 8d70, 8d90, 8db0, 8de0, 8df0, 8e00` |
| split / per-sequence | `0x43a0` | `0x9450, 9470, 94a0, 94c0, 94e0, 9510, 9520, 9530` |
| shared / dense | `0x45b0` | `0x9250, 9270, 92a0, 92c0, 92e0, 9310, 9320, 9330` |
| shared / per-sequence | `0x49d0` | `0x9a80, 9aa0, 9ad0, 9af0, 9b10, 9b40, 9b50, 9b60` |

These are kernel-relative byte offsets, not stable API identifiers. With `--racecheck-report
hazard`, the dense run shows writers 384/416/448/480 (lane 0 of each correction warp) and
readers 64-71; the addresses are the four float words per head of `sRowSums`.

The generated PTX and line-attributed SASS preserve this sequence:

```text
correction (128 threads)                 load / split_stats (32 threads)
st.shared sRowSums[warp, h]              barrier.sync   sum_done_id, 160
barrier.arrive sum_done_id, 160          ld.shared.v4.b32 sRowSums[:, lane]
```

The ID is 4 for split and 6 for shared K/V. All correction threads arrive (only the stores
are lane-0-predicated), all load threads sync before the head predicate, and the allocation
is never overwritten after these stores. The [one-shot done-flag recipe](channels.md#one-shot-done-flag)
cites the PTX memory-model guarantee: arrival publishes prior ordinary SMEM writes at barrier
completion, and sync prevents subsequent reads before completion. No extra SMEM fence or
publisher-side wait is required. Each CTA runs at most one item, so the publisher cannot lap
the waiter. This conclusion follows from the access/participation audit and emitted code,
not from numerical tests or a clean synccheck alone. Racecheck does not credit this ordering
in these kernels; its internal reason is not established.

Historical example sources compiled against the same current layer reproduce the pair:

| example source | protocol | store / read lines | four-launch racecheck summary |
|---|---|---|---|
| `fc45c88^` (`cf02a26`) | already one-shot `NamedBarrier` | 897 / 547 | 4 errors, 0 warnings |
| `4b6bda4^` (`f3ce28f`), the actual pre-port version | managed named channels | 906 / 547 | 4 errors, 0 warnings |

The latter used `with sum_done.produce()` and `sum_done.wait()` around exactly these stores
and reads; its extra empty-barrier handshake did not remove the reports. `fc45c88` added
width/schedule guards, not the NamedBarrier port (`4b6bda4`).

FA4's own swap-AB decode at `00807a02` was also checked with the same Q/K/V shapes, lengths
and requested split counts. Its per-sequence split table comes from FA4's prepare path, not
the port's fixed `[1, 2, 4]` table. It reports the same row-sum handoff: correction
`flash_fwd_decode_sm100.py:1502` writes `sRowSumWarp`, and `store_split_stats:1597-1598`
reads the four warp sums after `bar_sum_done.arrive_and_wait()`. Four scalar read PCs yield
four RAW groups per launch, for 16 errors over four launches. There are also four separate
warnings on Q-mbarrier initialization (line 834) versus arrive/expect-tx (line 836), not on
the row-sum data. No FA4 changes or blanket suppression of those warnings were made.

`test/test_cute_warpdrive_decode.py::test_decode_stats_handoff` isolates one dense CTA and
uses a CPU fp32 reference, so unrelated SDPA reference kernels cannot contaminate racecheck.
Its docstring gives the reservation, timeout, `CUTE_DSL_LINEINFO=1` and sanitizer invocation
for each backend and the expected nonzero result: one analysis error for split/shared,
four errors plus one warning for FA4. All five four-launch runs also passed their CPU
reference assertions; split/shared/FA4 pass synccheck on the isolated test. Hazard counts
and grouping depend on launch count and codegen; new access pairs need a fresh audit.

### Shared KV ring

```python
out, lse = fa4_decode(q, k, v, num_splits, seqused_k=lengths, splits_per_seq=counts, shared_kv=True)
```

`SharedKVDecode` (`shared_kv=True` in `fa4_decode`/`decode_partials`) ports the prototype onto
the layer: one `Ring(sum(kv_stages), Tma(...), Umma, consumer_mode="exclusive")` whose stages
hold either a K tile or a V^T tile (the V buffer overlaps K), with the QK and PV warps as exclusive
consumers that `skip()` each other's blocks, and FA4's QKIssued/PVIssued `NamedBarrier`s
([recipe](channels.md#rendezvous-recipe-fa4s-qkissuedpvissued)). Both MMA warps probe their
next stage with a try token after each release, as FA4 does; the loader probes its next stage
after each commit within a block's K or V group (a token cannot cross the dynamic regions
between groups, where FA4's loader carries its own). The default remains the split rings:
the shared ring wins on ragged lengths and the 32-head single-sequence case and loses on the
others. Partials stay bitwise equal to FA4's with either.

Decode-kernel CUPTI medians in microseconds of the committed benchmark
(`bench_fa4_decode.py`, FA4 prepare/decode/combine, port decode/combine; only the decode
kernel timed), `--shared-kv` against the default in three process pairs (A/B, B/A, A/B)
under one reservation, seven alternating FA4/port rounds of 200 replays after three seconds
of warmup, median of the three process medians. GB300, CuTeDSL 4.8.0, PyTorch
`2.15.0.dev20260922+cu132`, Triton 3.8.0, FA4 `00807a02`, November 5, 2026, normal DVFS;
lower is better:

| batch x kv_len, heads/kv_heads | FA4 | split (default) | shared_kv | shared vs split |
|---|---:|---:|---:|---:|
| 1x8192, 8/1 | 4.320 | 4.256 | 4.288 | +0.75% |
| 4x16384, 32/8 | 39.872 | 40.224 | 40.257 | +0.08% |
| 8x32768, 32/8 | 152.097 | 148.162 | 150.432 | +1.53% |
| 8x65536 ragged, 8/1 | 26.945 | 28.162 | 27.104 | -3.76% |
| 16x32768 ragged, 64/8 | 179.906 | 179.842 | 180.019 | +0.10% |
| 1x32768, 32/1 | 7.088 | 7.168 | 6.976 | -2.68% |
| 2x131072 ragged, 16/1 | 9.281 | 8.736 | 9.248 | +5.86% |

The ragged case gains 1.06 us (the raw prototype's wrapper confirmation: 0.895 us, 27.487 to
26.592 us), leaving 0.16 us to FA4. Each variant's three process medians agreed within
0.16 us (0.1%). Levels move more than that between reservations: an earlier
reservation on the same GPU measured the ragged case at 27.36 (split) and 26.37-26.40 us
(shared) with FA4 at 26.18 us, so compare columns of one run only. Pick the protocol per
shape: shared for ragged lengths with a single KV head and for long single sequences with
many heads per KV head; split elsewhere, notably for the 2x131072 ragged case (+5.9%) and the
8x32768 full case (+1.5%).

The try tokens are worth keeping on the shared ring, unlike on the split rings: removing them
(blocking waits throughout), interleaved in three process pairs on shapes 1x8192, 8x65536
ragged, 1x32768 and 2x131072 ragged, measured 4.352/26.432/7.040/9.248 us against
4.288/26.431/6.944/9.248 us with them.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_fa4_decode --shared-kv \
  --output agent_space/decode_shared.json
```

## warpdrive fixed costs

[bench_overhead.py](../examples/bench_overhead.py) times the layer against hand-written
`cute.arch` twins (GB300, same versions and date as above, seven interleaved rounds):

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m transformer_nuggets.cute.warpdrive.examples.bench_overhead \
  --output agent_space/overhead.json
```

Per one-way handoff, in SM cycles on GB300 (warpdrive / raw with
`cute.arch.mbarrier_wait` / raw with one hintless `try_wait` first): Threads 170 / 166 / 107,
named 53 / 53 / 53, UMMA commit 164 / 164 / 105, TMA 317 / 317 / 257, round robin
170 / 172 / 119; streams through 1- and 4-stage rings match as well. The channel code adds
nothing; a try-wait first is about 60 cycles faster when the waiter is already parked
(`mbarrier_wait` lowers to `TRYWAIT`, then `NANOSLEEP.SYNCS` with its 10 ms hint), but on
the decode, GEMM and FA4 kernels it did not help (see above and the setup-cost section).

The fixed-cost suite runs the decode example's 16-warp layout, 48 mbarriers, TMEM and a
OneShot stream with empty roles. On 148 CTAs, before the allocator change, it took 2.14 us against 1.95 us for a twin
that allocates TMEM from warp 0 and 1.82 us for one that allocates from warp 1, as FA4 does.
Teardown (role end to kernel end) costs about 70 cycles in both. The CTA sync to the first
role instruction took about 600 cycles longer than the warp-1 twin, for three reasons:

1. Warp 0 initialized every barrier and then allocated TMEM. One CTA now allocates from the
   lowest other warp of a TMEM role: role start 1260 -> 925 cycles, 1.89-1.92 us.
2. LLVM folds the role dispatch's compares on the warp index into a switch, which ptxas
   lowers to an indexed constant load and an indirect branch (`BRX`), here worth about 116
   cycles. It appears with three or more consecutive single-warp roles, as in the decode,
   not in the other examples. Opaque per-test copies of the warp index removed it but made
   the decode 2% and single-CTA GEMM/FA4 about 1% slower; not kept.
3. Each role decodes its first item behind an `index < total` guard after the sync, so the
   guard's constant load and then the mapping's divisor loads are serialized there; the
   twin's unguarded decode is hoisted above the barrier. A mapping that sets
   `decodes_any_index` drops the guard (role start about 925 -> 858 cycles). Decoding the first
   item once in `begin_roles` instead lengthened warp 0's path to the sync and cost 80 ns.

## Historical timings

The [original README at 49b9713](https://github.com/drisspg/transformer_nuggets/blob/49b9713/transformer_nuggets/cute/warp_roles/README.md)
records the November 2026 GB300 GEMM/FA4 experiments. Those were BF16 CUDA-graph replay medians
with CuTeDSL 4.8.0, a PyTorch 2.15 nightly cu132 build and FA4 at `849b5daa`.

They are measurements of that revision, not of the storage-binding or work-API changes.
Keep the original shapes, schedule and reference implementation with the numbers when comparing;
a FA4 default schedule and a matched single-CTA schedule answer different questions.

## Still missing

cp.async producer mechanisms are not supported. The decode example (both variants) is
OneShot-only. Using CUTLASS pipeline objects where their semantics match remains a possible
implementation cleanup, not a reason to change the role bodies.

The alias diagnostic still needs IR-inferred accesses, partial K-block footprints, cluster
analysis and completion-aware epoch checks. Its current limits are in
[storage](storage.md#read-the-result).

For the normal and quick-fuzz commands, see [Tests](../README.md#tests). Long campaigns and
what their oracles cover are in the [fuzz harness guide](../../../../test/warpdrive_fuzz/README.md).
