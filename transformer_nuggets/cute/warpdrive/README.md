# warpdrive

Write the load, MMA and epilogue of a warp-specialized CuTeDSL kernel as separate methods.
Connect them with channels:

```python
self.ab = Ring(3, Tma((128 + 256) * 64 * 2), Umma)
self.acc = Ring(1, Umma, Threads)
```

You still choose the tiles, layouts, warps and register budgets. warpdrive handles the barrier
counts and phases, role dispatch and TMEM allocation/free. The math is ordinary CuTeDSL.

Experimental. Requires `nvidia-cutlass-dsl` 4.8 and a Blackwell SM100/SM103 GPU. A kernel that
compiles can still deadlock if its roles disagree about the work. See [GUARANTEES.md](GUARANTEES.md)
for what is checked and what you have to get right.

## A GEMM

[gemm.py](examples/gemm.py) computes BF16 `D = A @ B.T`. Warp 4 loads A/B, warp 5 issues MMA,
and warps 0–3 store D. Here is the declaration and load role; the linked file has the MMA,
epilogue, tensor partitions and launch:

```python
import cutlass
from cutlass import cute
from transformer_nuggets.cute.warpdrive import (
    OneShot,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    Tma,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from transformer_nuggets.cute.warpdrive.examples.gemm import GemmParams


class Gemm(WarpSpecialized):  # excerpt; other roles and the kernel prologue are in gemm.py
    def __init__(self):
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", cutlass.BFloat16, align=1024)
        b = self.smem_plan.buffer("B", cutlass.BFloat16, align=1024)
        self.ab = Ring(3, Tma((128 + 256) * 64 * 2), Umma, buffer={"A": a, "B": b})
        self.acc = Ring(1, Umma, Threads)
        self.tmem_plan = TmemPlan().region("acc", 256)
        self.work = WorkStream(schedule=OneShot())

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: GemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        while work.valid:
            m, n, _ = work.item
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as h:
                    cute.copy(
                        params.tma_a,
                        params.tAgA[None, m, k],
                        h.view("A"),
                        tma_bar_ptr=h.barrier,
                    )
                    cute.copy(
                        params.tma_b,
                        params.tBgB[None, n, k],
                        h.view("B"),
                        tma_bar_ptr=h.barrier,
                    )
            work.advance()
        return ab, work
```

`ab.produce()` waits for a free stage; TMA fills it and completes its full barrier. The MMA
role waits with `ab.consume()`, reads that stage and releases it for the next load.
`warps=4` means warp index 4, not four warps.

Run the complete example through its wrapper:

```python
import torch
from transformer_nuggets.cute.warpdrive.examples.gemm import gemm

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
out = gemm(a, b)
```

## How it fits together

A role gets its input values from `params` and its channel ends from annotated arguments.
The argument `ab: Producer` binds to `self.ab`. Use a frozen dataclass for `params`; return
every endpoint, work cursor and turnstile seat you take as an argument. Helpers must return
updated state too. See [roles](docs/roles.md) for placement and binding.

Channels describe handoffs, not the math. A `Ring` cycles through stages; `Slots` lets you name
the slot. `Tma`, `Umma` and `Threads` say how each side signals. For `Threads`, all threads of
the bound role participate, even when only some lanes have data to write.

`WorkStream` decides which work each CTA runs. `mapping=` turns a linear index into an item;
`schedule=` chooses the indices. Change `OneShot()` to `StaticPersistent()`,
`AtomicPersistent()` or `ClusterLaunchControl()` without changing the role loop. The host
setup must also supply the [schedule's launch arguments](docs/scheduling.md#bind-and-launch).

In the kernel prologue, `begin_roles()` initializes the barriers and returns a `SmemAllocator`.
Allocate your tensors there, or declare a `SmemPlan` and bind its buffers to channels with
`buffer=`. `run_roles(params)` synchronizes, runs the roles and frees TMEM after its users
finish. [Storage](docs/storage.md) shows how a handle selects its own buffer stage.

## More examples and details

- [Roles](docs/roles.md): placement, params, returned state and register budgets.
- [Scheduling](docs/scheduling.md): launch setup, custom work lists and CUTLASS/FA4 mappings.
- [Channels](docs/channels.md): rings, slots, overlapping handles, TMEM and turnstiles.
- [Storage](docs/storage.md): stage views and checking declared aliases.
- [CTA pairs](docs/cta_pairs.md): two-CTA MMA and cross-CTA handoffs.
- [Dropping down](docs/dropping_down.md): raw barriers, count overrides and unchecked sites.
- [Tracing](docs/tracing.md): inspect the kernel, capture IKET ranges and use the profiler.
- [Examples](docs/examples.md): persistent GEMM, FA4, input limits and historical timings.

## Tests

From the repository root:

```bash
pip install -e ".[cute,dev]" pytest-xdist
pytest -n 12 test/test_cute_warpdrive*.py
pytest -n 12 -m fuzz test/test_cute_warpdrive*.py
```

The first pytest command runs the normal suite; fuzz tests are skipped by default. The second
opts into quick fuzzing. FA4 tests skip without `flash_attn.cute`. Long campaigns, sanitizer and
CUTracer runs are described in [GUARANTEES.md](GUARANTEES.md#fuzzing-and-evidence).

The suite covers numerical correctness and determinism, shape reuse, schedules and CTA pairs,
protocol errors, raw/managed synchronization, TMA and alias diagnostics, and the
IKET-to-profiler path.
