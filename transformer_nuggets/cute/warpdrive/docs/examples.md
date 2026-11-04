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
