# Write a warpdrive kernel

[Back to warpdrive](../README.md) · [Safety contract](../GUARANTEES.md)

Start from a complete kernel, then change its math and data movement. warpdrive owns the
role dispatch and declared handoffs; the code inside each role is ordinary CuTeDSL. You do
not need the numbered rejection catalog to use it.

## Run a complete example

[gemm.py](../examples/gemm.py) is the smallest complete GEMM: declarations, three roles,
SMEM partitions, TMEM views, launch setup, cached compilation and a PyTorch wrapper in one
file. It computes BF16 `D = A @ B.T` with FP32 accumulation on Blackwell SM100/SM103.
Use CuTeDSL 4.8 and the repository's [environment setup](../README.md#tests).

From the checkout root, with that environment selected:

```bash
python -m transformer_nuggets.cute.warpdrive.examples.gemm
pytest -q test/test_cute_warpdrive.py -k test_gemm
```

On a shared GPU host, run each command through the host's GPU reservation tool with a
process timeout. The module prints numerical errors; the tests assert correctness and the
wrapper's boundary behavior. CUDA compilation must read kernel definitions from a source
file, not a REPL or notebook cell.

Read the file in this order:

| Part | What to adapt |
|---|---|
| `Gemm.__init__` | Channels, stage counts, storage and work assignment |
| `load / mma / epilogue` | Math and handoffs for one work item |
| `Gemm.kernel` and `GemmParams` | Initialization, layouts and per-role tensor partitions |
| `Gemm.__call__` | Atoms, TMA descriptors, bound work and launch |
| `compile_gemm / check_gemm_args / gemm` | Compile-time choices, runtime ABI and input contract |

For an operation without MMA, keep the same role/channel structure and omit TMEM. Don't
copy the GEMM's MMA-specific layouts or register split just to get a skeleton.

## Define one work item

Write down the output a CTA or cluster owns and the inputs needed to compute it. For this
GEMM, one item is `(m_tile, n_tile, 0)` and owns a 128 x 256 output tile. It reduces K in
chunks of 64. All three roles visit the same output tiles; the load and MMA roles also agree
on the number of K chunks per item.

Use distinct items for split reductions, such as `(tile, split)`, and reduce their partials
separately. A scheduler choosing the last item does not mean every earlier item is finished.

Start with `OneShot()` unless the algorithm requires a persistent protocol. Move to the
[persistent GEMM](../examples/persistent_gemm.py) when you need reusable roles across items,
CTA pairs, or another schedule. It keeps the launch ABI suitable for schedule changes.

## Draw the handoffs

The GEMM's dataflow is:

```text
warp 4: load A/B -- ab (3 SMEM stages) --> warp 5: MMA
                                              |
                                     acc (1 TMEM stage)
                                              |
                                              v
                                      warps 0–3: store D
```

Before writing the bodies, account for each channel per work item:

| Channel | Producer → consumer | Epochs per item | Reuse becomes legal after… |
|---|---|---|---|
| `ab` | TMA load → MMA | ceil(K / 64) | MMA completion releases the input stage |
| `acc` | MMA → epilogue threads | 1 | All epilogue threads release the accumulator stage |

An epoch is one produced/consumed value, not a stage index: three stages are reused as
0, 1, 2, 0, … across successive epochs. Changing stage count does not change how many values
each role must exchange. Multiple consumers of one value all release before reuse.

For your kernel, add any feedback edges and identify what lets the first iteration run.
Balanced counts are insufficient if everyone starts by waiting for someone else.

## Declare storage and connect the roles

The GEMM declares:

```python
self.ab = Ring(ab_stages, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
self.acc = Ring(acc_stages, Umma, Threads)
self.tmem_plan = TmemPlan().region("acc", TILE[1], acc_stages)
self.work = WorkStream(schedule=OneShot())
```

These are excerpts from the linked source, not a standalone kernel. `a` and `b` are
`SmemPlan` buffers. In this single-CTA GEMM, `stage_bytes` is the combined A/B bytes written
by TMA per stage: `(128 + 256) * 64 * 2` for BF16. Don't multiply by ring depth.

Parameter names bind the ends: `ab: Producer` and `ab: Consumer` both refer to `self.ab`.
`@role(warps=4)` means warp index 4, not four warps. Use `warps=range(4)` for warps 0–3.
TMA and UMMA sides bind to a single-warp role; a `Threads` side counts every bound thread.
See [roles](roles.md) for bindings and valid warpgroup register budgets.

Begin with separate buffers for overlapping lifetimes. Use `SmemPlan` for allocation and
`StageView` for your TMA/MMA partitions; `h.view(...)` then chooses the handle's stage.
You still construct the layouts and must ensure each view actually refers to the named buffer.
See [storage](storage.md) before adding aliases or using buffers outside handles.

## Implement the lifecycle, then the math

The load role's structure is:

```python
@role(warps=4, regs=40)
@cute.jit
def load(self, params: GemmParams, *, ab: Producer, work: WorkCursor):
    ab = ab.views(A=params.tAsA, B=params.tBsB)
    while work.valid:
        m, n, _ = work.item
        for k in cutlass.range(params.num_k_tiles, unroll=1):
            with ab.produce() as h:
                cute.copy(
                    params.tma_a, params.tAgA[None, m, k], h.view("A"), tma_bar_ptr=h.barrier
                )
                cute.copy(
                    params.tma_b, params.tBgB[None, n, k], h.view("B"), tma_bar_ptr=h.barrier
                )
        work.advance()
    return ab, work
```

The consumer uses `with ab.consume() as h:`. Entering waits for the value; exiting releases
it through the declared mechanism. On this channel the MMA consumer's release uses an UMMA
completion, so issuing asynchronous MMA is not mistaken for finished input reads. The MMA
role holds an `acc.produce()` handle around the entire K reduction and publishes once. The
epilogue holds `acc.consume()` until it finishes using that accumulator stage.

Keep handoffs outside lane-dependent predicates. For a ragged output, every epilogue thread
enters and releases the handle, while only in-bounds lanes store. A predicate around the
whole exchange can strand a barrier whose declared count still includes those lanes.

Return every endpoint, cursor and turnstile seat from each role. When a helper advances one,
return and reassign it there too. Don't return handles, params or TMEM views. Close handles
once, within their opening dynamic region; don't keep using their pointers after release.
For non-nested lifetimes, explicit `acquire/wait/commit/release` are available: see
[overlapping handles](channels.md#overlapping-handles).

## Wire the prologue, launch and cache

Keep these boundaries visible in your source:

1. **Entry point:** create atoms/descriptors and bind the runtime work space. Name the inner
   kernel with `set_name_prefix(...)`, and launch with `self.launch_config(bound)`. For a
   persistent schedule, pass `launch_args()` through the entry point as shown in
   [Bind and launch](scheduling.md#bind-and-launch).
2. **Kernel prologue:** call `begin_roles(work=bound, layouts=...)` in converged code, get
   buffers, build partitions and frozen params, then call `run_roles(params)`. Declare every
   TMEM-using role with `tmem: Tmem` so it participates in teardown.
3. **Compile helper:** follow the current example and [compile-key contract](roles.md#compile-keys).
   Use fake compact tensors for a compact runtime ABI. Every static choice that changes code
   belongs in the key; runtime dimensions can stay dynamic within their declared constraints.
4. **Python wrapper:** validate dtype, device, rank, shape relationships, compactness,
   alignment and output overlap before relying on the fake ABI. Define empty-output and
   empty-reduction behavior. A shape with no K work must not publish an uninitialized sum.

Do not replace those checks with a successful compile. The wrapper enforces the input
contract, and the role bodies implement it; the layer does neither on your behalf.

## Check the kernel before tuning

Compare with a reference using nonuniform inputs and an explicit numerical tolerance.
For this GEMM, tests compare FP32-reference error against the BF16 PyTorch baseline. A different
reduction or approximation needs its own acceptance criterion.

Choose cases that exercise your actual lifetimes:

- One work item, then enough epochs to wrap every ring's stages and phases.
- Ragged tails and any supported empty work/reduction.
- Multiple items per unit if persistent; uneven work across units.
- Repeated launches and changed runtime shapes through the same compiled entry point.
- Every advertised cluster/schedule path; disjoint workspaces for concurrent launches.

Read `op.describe()` before compiling for declaration errors, then inspect it after tracing
for TMA/alias findings. Resolve each `unverified` site with explicit reasoning and tests;
`no hazard found` is not a proof. For a new raw async handoff or storage overlap, validate
completion and ordering independently, including sanitizer checks appropriate to that path.
A timeout is a failure to investigate, not a reason to extend the run until it passes.

Run the [repository testing tiers](../../../../AGENTS.md#warpdrive-testing-tiers). Only then
measure and tune stages, warps, register budgets, layouts or schedules. Keep the workload and
numerical checks fixed across comparisons. More stages or fewer instructions are not a
performance result by themselves.

## Where to go next

- [Persistent GEMM](../examples/persistent_gemm.py): multiple work items, pairs and multicast.
- [FA4 forward](../examples/fa4_fwd.py): multi-consumer handoffs, slots and overlapping storage.
- [Swap-AB decode](../examples/fa4_decode_swapab.py): round-robin instances and named channels;
  `SharedKVDecode` (`shared_kv=True` in the wrappers) adds exclusive consumers with FA4's
  rendezvous. Currently OneShot-only, not a persistent template.
- [Packed work](../examples/packed_work.py): a device-built list and readiness before decoding.
- [Dropping down](dropping_down.md): raw barriers and targeted opt-outs when a managed pattern
  does not express the algorithm. Explain the replacement ordering; don't suppress a check
  merely to make compilation pass.

For agent-assisted authoring, ask the agent to read this guide and the safety contract, then
show the work item, role/channel diagram and per-channel epoch counts before implementing.
The repository-local [authoring skill](../SKILL.md) follows that procedure and routes to the
same references. Finish with the runnable kernel, its tests and any remaining raw/unverified
sites—not just a declaration sketch or a profiler screenshot.
