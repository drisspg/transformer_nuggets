# FlexGEMM constraints

This describes the current FX-to-QUACK path, not a permanent API promise. Source paths below refer to the PyTorch checkout. Check `torch/_inductor/kernel/flex_gemm/` and the corresponding `test/inductor/test_flex_gemm.py` test before generalizing a supported composition.

## API, devices, and layouts

- GEMM ops: `torch.mm`, `torch.addmm`, `torch.bmm`, `torch.baddbmm`, `torch.nn.functional.scaled_mm`, and `torch.nn.functional.grouped_mm`.
- Backends are `TRITON` (default), `QUACK`, and `NVGEMM`. The advanced features below describe QUACK; unsupported QUACK compositions raise rather than silently becoming stock GEMMs.
- QUACK requires CUDA and CuTeDSL. Use tensors on the same device. Dense mm/addmm use rank-2 matrices; bmm/baddbmm use rank-3 matrices. Grouped GEMM has separate contracts below.
- Ordinary dense QUACK has an SM90 Hopper dispatch path; there is no blanket SM100-only backend gate. Do not infer support for advanced epilogues from that dispatch. Grouped-main contraction excludes SM90; Hopper grouped-main/reduction extensions are not part of the validated landed feature set.
- Non-block-scaled matrix operands need a 16-byte-aligned base, one unit-stride matrix dimension, and 16-byte-aligned outer strides. Main and full-shape auxiliary stores have corresponding stride constraints. A logically valid shape can still have no legal kernel config.
- Epilogues must be functional and FX-traceable. There is no built-in backward; see [autograd.md](autograd.md).
- `alpha` and `beta` must be static numeric scalars. QUACK kernel options are `backend`, `autotune`, `fast_math`, and `config`.
- Generic main stores support floating-point and bool dtypes, subject to the backend's dtype support. A contracted packed output may use uint8 storage and a terminal same-width dtype view; this does not enable arbitrary integer main outputs.

Sources: `torch/_higher_order_ops/flex_gemm.py`; `lowering.py:114-148,395-414,1049-1062,1124-1152`; `torch/_vendor/quack/epilogue/frontend.py:208`; `quack_ops/main_store.py:115-167`.

## Captured tensors

Ordinary PyTorch pointwise FX is the expression language. For a 2-D `[M, N]` GEMM, captures can be scalar `[1, 1]`, row `[1, N]`, column `[M, 1]`, or full tile `[M, N]`.

- A scalar or numel-1 rank-1 capture is normalized to `[1, 1]`. A rank-1 `[N]` used directly or as `w[None, :]` is a row; `[M]` used as `w[:, None]` is a column. Direct broadcasting is a row when `M == N`. Using the same rank-1 capture as both row and column rejects.
- `bmm` supports batch-shared scalars/vectors, per-batch `[B, 1, N]` rows and `[B, M, 1]` columns, and `[B, M, N]` tiles. Shared rank-3 forms have leading size 1. Captured bmm epilogues must preserve the main shape; in-body views must preserve batch/row/column axes and dtype, apart from leading singleton dimensions.
- Lowering makes row/column captures contiguous as needed. Batched vectors need unit vector stride and a batch stride divisible by four elements; lowering pads invalid layouts rather than requiring callers to pre-pad. Tile captures can likewise require an aligned/padded copy. Captures are not necessarily zero-copy.
- An additional non-bmm profile reads `[M, 2N]` as two interleaved lanes via `h.view(M, N, 2)[..., j]`. This requires SM100 and no `swap_ab`; it is not a general arbitrary-width tile capture.
- Capture dtype validation uses QUACK's dtype map: FP16, BF16, FP32, supported FP8 formats, uint8, int32, int64, and packed FP4 storage. Bool is passed physically as uint8. Float64 is rejected. Passing this map is not a promise that every operation supports that dtype.
- `addmm` and `baddbmm` do not support captured tensor reads. `bmm` capture support does not imply generic tuple-auxiliary or local-reduction support.
- Contracted main outputs accept only scalar, column, and logical-row captures, not full tiles/interleaved captures. Logical rows are widened over the physical lanes and must be used against selected logical lanes, not physical GEMM columns.

Sources: `lowering.py:296-393,417-615,1094-1110,1153-1187`; `runtime.py:99-107,260-280`; `torch/_vendor/quack/cute_dsl_utils.py:58-73`; `quack_ops/interleaved.py:101-103`.

## Local reductions

Split exactly one physical GEMM output dimension, then reduce that group:

```python
x = acc.float().view(m, -1, group)  # N groups: reduce dim -1
x = acc.float().view(-1, group, n)  # M groups: reduce dim 1
```

Supported ordinary reductions are `sum`, `mean`, `prod`, `amax`, and `amin`. Explicit reduction `dtype=` is unsupported; cast the source instead.

- `group > 1` and the selected dimension must be divisible by it. Groups up to 32 must divide 32; larger groups must be multiples of 32, not necessarily powers of two.
- A legal tile must own the group. Tile dimensions, clusters, architecture, and `swap_ab` impose additional restrictions; the arithmetic group-size rule alone is insufficient. In particular, ordinary N-group stores require `tile_n % group == 0`; the largest available tile bounds the group.
- Reductions are supported for mm and compatible scaled-mm epilogues; grouped_mm supports only the restricted varlen-M case below. They do not compose with addmm/bmm/baddbmm or their C/alpha/beta semantics.
- There is one compressed auxiliary slot, and a main output is still required. Independent statistics are not enabled merely by sharing group geometry.
- Exactly two distinct reductions can form a stored chain: same axis-1 geometry, power-of-two group at most 32, first reduction with `keepdim=True`, feeding the shape-preserving pointwise source of the second. The chain cannot feed main. Equivalent reductions can share a physical reduction.
- Grouped `logsumexp` follows this max-then-sum chain, not an arbitrary multi-plane associative-state API. It does not extend support to axis-M or groups above 32. Full-row `sum`/`softmax`/`logsumexp` without an explicit grouped view reject. Return partials and finalize outside FlexGEMM when the full dimension is too large.
- A compressed-store finalizer can read floating scalar captures, not row/column/tile captures.

### Feeding a reduction back into main

Use a recognized broadcast such as `x - r`, `x * r`, or `x / r`, with a `keepdim=True` reduction.

- Axis 1 supports groups at most 32 through an accumulator prepass, including shape-preserving pointwise reduction sources. Larger groups reject. This permits small grouped normalization, not general full-row RMSNorm.
- Axis 0 supports same-warp groups at most 32. A transformed source such as `x.square().mean(1)` does not have the same support as the axis-1 prepass and rejects. A store-only post-reduction transform of the value that also feeds main rejects; aliases/views of the reduction are different from such a transform.
- Selected-lane reductions have a narrower physical-span-2 contract and must feed main. Do not assume every grouped-main/reduction pairing composes.

Sources: `constraints.py:169-205`; `quack_ops/grouped_reduce.py:189-230`; `fx_cutedsl_codegen.py:166-215,390-449,638-733,860-890`; `torch/_inductor/kernel/gemm_epilogue_analysis.py:637-676,800-839`; `constraints.py:120-124`.

## Grouped-main contraction and expansion

Contraction maps physical `[M, N]` GEMM values to `[M, N // group]`:

- Interleaved lanes: `acc.view(M, -1, group)[..., j]`.
- Chunked lanes: `acc.chunk(group, dim=-1)` or `acc.view(M, group, -1)[:, j, :]`.
- Lane selection need not consume every lane. The output must stay in one recognized lane domain; arbitrary slicing is not a general contraction API.
- Group 2 supports SM100/SM110; group 4 supports SM100. No `swap_ab`. The config must also satisfy store ownership, including `tile_n <= physical_N`.
- Both dynamic M and dynamic N are supported in validated cases. Structural groups/select indices are specialized and guarded; chunk/split forms can specialize widths. Do not assume a single graph covers arbitrary widths or alignment.
- Chunked contraction requires column-major B: normally store a weight as `[group * logical_N, K]` and pass `.t()`. Block-scaled chunked contraction is unsupported.
- Interleaved contractions can return full-physical-shape auxiliaries and compatible reduction stores. Chunked group-2 can return the unchanged accumulator as its sole ordinary auxiliary, with 16-byte-aligned halves. Chunked contraction rejects ordinary physical-span-1 grouped reductions; selected-lane reductions follow the narrower contract above.
- Contraction analysis requires a 2-D GEMM result. Varlen-M can qualify; bmm and varlen-K do not.

Expansion stores two same-dtype GEMM-shaped lanes as `[M, 2N]`: `stack((a, b), -1).view(M, 2N)` for interleaved output, or `cat((a, b), -1)` for chunked output. Interleaved stores require SM100 and no `swap_ab`; chunked stores require 16-byte-aligned halves. Full-shape auxiliaries and stored grouped reductions must be computed before packing; feed-main reductions do not compose with expansion.

Sources: `torch/_inductor/kernel/gemm_epilogue_layout.py:185-330,364-448`; `fx_cutedsl_codegen.py:276-330`; `quack_ops/main_store.py:87-167`; `quack_ops/interleaved.py:101-125`; `lowering.py:962-975,999-1008,1034-1045`; dynamic tests at `test/inductor/test_flex_gemm.py:4119-4254`.

## Grouped GEMM

Call `F.grouped_mm` with `gemm_args=(A, B)` and `gemm_kwargs={"offs": offs}`. Both operands must have matching BF16 or FP16 dtype. `offs` is a contiguous rank-1 int32 tensor on the operand device containing cumulative end offsets, not lengths; lowering prepends zero for QUACK.

- **Varlen-M:** A `[total_M, K]` must be K-major; B is `[E, K, N]`, one matrix per group. Output is `[total_M, N]`. Captures and full-shape auxiliaries are supported. Local reductions may only return stored N-group partials: no M groups, feed-main, reduction-chain prepass, or transformed blocked/transposed reduction storage.
- **Varlen-K:** A `[M, total_K]` must be M-major and B `[total_K, N]` N-major, with matching reduction extents. Output is `[E, M, N]`; main/auxiliary outer strides must be 16-byte aligned. Captures and full-shape auxiliaries are supported, but local reductions, contractions, and expansions reject.
- No 3-D A, `offs=None`, bias, `out_dtype`, or scaled grouped GEMM support. The grouped backend requires a persistent config; its offset/alignment contract is not ordinary dense GEMM broadcasting.

Sources: `torch/_higher_order_ops/flex_gemm.py:549-592,619-627`; `lowering.py:200-248,855-869,938-967,1009-1018`; `torch/_vendor/quack/epilogue/frontend.py:947-979`.

## Auxiliary and transformed stores

Generic tuple epilogues return main first, then auxiliaries in user order. Ordinary auxiliaries must match the physical GEMM shape; only mm, scaled_mm, and grouped_mm support them. They can coexist with the one recognized compressed-reduction output. Indexed gathers are not supported merely because the aux-shape error mentions an indexed row gather.

- `to_blocked(scale)` is a terminal storage transform, not a value-domain grouping operation. Return it directly around a recognized axis-1 reduction/scale expression. Its fused storage gate is **SM100 only** (`device_capacity == 10`), with additional tile-layout constraints. Padded blocked carriers are zero-filled; this may require a separate fill operation.
- A returned 2-D reduction can use a contiguous transpose, e.g. `reduced.t().contiguous()`. This is another terminal storage contract, not arbitrary intermediate transposition, and excludes `swap_ab`.
- A terminal `view(dtype)` reinterprets bits and must preserve shape, stride, and element size. The kernel stores the source dtype, then the wrapper re-views it. Keep storage views separate from numeric conversion.

Sources: `lowering.py:618-643,1321-1333`; `fx_cutedsl_codegen.py:82-120,259-273`; `output_layout.py:76-110,126-134`; `runtime.py:539-545`; `torch/_inductor/kernel/gemm_epilogue_analysis.py:152-206`.

## Precision, quantization, and fast math

FP32 GEMM operands compute in TF32 on QUACK. Opt in with `torch.set_float32_matmul_precision("high")` (or a policy permitting TF32); `highest` rejects rather than falling back.

The epilogue receives an FP32 accumulator. FP16/BF16 casts normally use promoted FP32 compute types, so a cast in Python/FX does not guarantee eager intermediate rounding. `emulate_precision_casts=True` does not by itself change this FlexGEMM cast lowering. Saved low-precision auxiliaries round at storage and can differ from their live forward consumers. Validate exact rounding requirements through the compiled HOP, not just its eager body.

`kernel_options={"fast_math": True}` enables approximate QUACK math, including tanh-based sigmoid/SiLU and tanh-approximate GELU. Scope it to calls where those numerical changes are acceptable.

`F.scaled_mm` requires `(A, B, scale_a, scale_b)`, explicit matching scale recipes, and QUACK. Supported recipes are MXFP8 E4M3 with block-32 E8M0 scales, or NVFP4 with block-16 E4M3 scales and optional scalar FP32 tensorwise scales. Block scales use `SWIZZLE_32_4_4`. No bias, custom contraction dimension, or fast accumulation. The FlexGEMM end-to-end block-scaled tests target SM100/SM110; wider vendor format support is not equivalent to validated FlexGEMM coverage.

For inline PTX, E8M0 conversion, packed FP4, and terminal storage views, use [inline-asm.md](inline-asm.md) and [patterns.md](patterns.md), not private test helpers as a required public API.

Sources: `lowering.py:114-127`; `fx_cutedsl_codegen.py:933-981`; `torch/_inductor/codegen/cutedsl/cutedsl_op_overrides.py:909-926`; `torch/_inductor/codegen/common.py:847-850`; `torch/_higher_order_ops/flex_gemm.py:82-142,383-547`; `test/inductor/test_flex_gemm.py:8427-8432`.

## Configs and dynamic shapes

`kernel_options["config"]` is a partial dictionary of QUACK `GemmConfig` fields that filters supported candidates; it cannot construct an arbitrary config. Unknown fields, exact-type mismatches (including bool versus int), or an empty compatible set reject. `autotune=False` chooses a supported default from the filtered set; `autotune=True` searches the filtered heuristic set, not necessarily every legal config.

SM100/SM110 MN-major TF32 operands need a 32-element per-CTA partition. Two-CTA MMA halves B's N, so MN-major B requires `tile_n % 64 == 0` in that case; account for `swap_ab`. Do not blanket-ban small tile sizes or K-major B.

Dynamic shapes do not erase divisibility, alignment, group-ownership, or minimum-tile constraints. Backed structural integers may specialize and cause guards/recompilation; shape/stride checks can reject new values. Test changed widths and strides, not just changed batch/row counts. Check empty M separately; do not use an empty-output shortcut for K=0 when the epilogue still has nonempty values to compute.

Sources: `lowering.py:765-793,694-711,1260-1300`; `torch/_inductor/kernel/gemm_epilogue_analysis.py:620-634`; `constraints.py:177-205`.

## If a valid-looking graph rejects

1. Confirm physical GEMM dimensions, capture profiles, strides, and dtypes.
2. Express grouping with one dimension split; distinguish logical lanes from physical columns.
3. Separate value transforms from terminal storage transforms.
4. Remove one composition at a time to find the unsupported boundary.
5. Inspect `TORCH_LOGS=+flex_gemm`, selected configs, and generated code. Fix a missing lowering narrowly rather than weakening a central guard.
