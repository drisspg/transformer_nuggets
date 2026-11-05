# Roles

[Back to warpdrive](../README.md)

```python
@role(warps=4, regs=40)
@cute.jit
def load(self, params: GemmParams, *, ab: Producer):
    for k in cutlass.range(params.num_k_tiles, unroll=1):
        with ab.produce() as h:
            ...  # load A/B for this stage
    return ab
```

Put `@role` directly above `@cute.jit`. `warps=4` means warp index 4;
`warps=range(4)` means warps 0–3. `warps=[range(4), range(4, 8)]` creates two instances of
the same method. The ranges for each instance must be contiguous and cannot overlap other roles.
Each instance runs on exactly those warps; holes run no user role.

`regs=40` sets that warpgroup's `setmaxnreg` budget. The default `None` leaves it unchanged.
All roles in a warpgroup must agree, including `None` versus a number. Numeric budgets must
be multiples of 8 from 24 to 256. Any numeric budget pads the launch to full warpgroups.
Padding warps get the group's budget too; entirely idle groups get 24. The adjustment
happens in a warpgroup-uniform branch before role dispatch, including in a pair's peer CTA.

The budgets have to fit in the registers the launch starts with. Every warp starts with
65536 / threads, rounded down to a multiple of 8 and capped at 256 (168 for 12 warps), and
`setmaxnreg` only moves registers between warps. So 12 warps can split 232/232/40 but not
232/232/48: that totals 65536, more than 168 x 384, and the increase would wait forever. It
raises `ValueError`. `None` counts as the starting budget.

`cta="leader"` restricts the role to the leader of each MMA pair of a
[cluster](clusters.md), rank 0 of a CTA pair. `cta="rank0"` runs it once per cluster. The default
is `"all"`. They are equivalent for one CTA; without MMA pairs, every CTA is a leader.

Call `begin_roles()` and `run_roles()` from converged CTA/cluster code using the declared
launch shape. Elected barrier initialization and the common initialization fence/sync
precede dispatch. Every participating thread must reach initialization and exit, including
threads in warp holes. For manually initialized barriers, see
[raw barriers and roles](dropping_down.md#raw-barriers-and-roles).

## Arguments

The dispatcher binds arguments by annotation and name. `*` makes the dependencies
keyword-only; positional signatures also work. Declare each argument explicitly, without
`*args` or `**kwargs`.

| Parameter | What it receives |
|---|---|
| `ab: Producer` / `ab: Consumer` | An end of `self.ab` |
| `s: Producer[Instance]` / `s: Consumer[Instance]` | An end of `self.s`, bound to this instance's slot |
| `stage: Instance` | The instance index, a Python int |
| `tmem: Tmem` | The kernel's TMEM view |
| `work: WorkCursor` | The cursor for `self.work` |
| `mufu: Turn` | A seat in `self.mufu`, a `Turnstile` |
| `bars: Barriers` | The base pointer of `self.bars` |
| `params: GemmParams` | The object passed to `run_roles(params)` |

Names matter for channels, streams, turnstiles and raw barriers. `ab: Consumer` requires a
channel named `ab` on the kernel. An instance-bound channel must be `Slots`, not `Ring`.
You still pass the slot explicitly, for example `s.wait(stage)`.

Use a frozen dataclass for typed params:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmParams:
    tma_a: cute.CopyAtom
    tAgA: cute.Tensor
    num_k_tiles: cutlass.Int32
```

Build it in the kernel prologue from the atoms, partitions, layouts and loop counts the roles
need. Frozen fields cannot be rebound, but the tensors they point to can still be written.
This is shallow immutability: referenced lists and other objects remain mutable too.
A `SimpleNamespace` also works; leave the role's one context parameter unannotated. The layer
does not make the namespace immutable.

An annotation matching the params type, or a base class of it, receives that object. An
unresolvable annotation, a missing dependency, two context parameters or incompatible context
types raises `ProtocolError`. `run_roles(params)` checks the object against the annotated
type. Mutable dataclasses are rejected, even without a type annotation.

## Return the updated state

```python
@role(warps=5, regs=40)
@cute.jit
def mma(self, params: GemmParams, *, ab: Consumer, acc: Producer, work: WorkCursor, tmem: Tmem):
    while work.valid:
        ...  # use the channel ends and the current item
        work.advance()
    return ab, acc, work
```

Return every endpoint, work cursor and turnstile seat you take as an argument. Return one
object or a tuple; tuple order doesn't matter. Missing, duplicate or foreign state is an
error. This applies with tracing on or off, and with one or two CTAs.

Do not return `self`, `params`, the instance index, TMEM view or raw barrier pointers. Close
stage handles inside the role; a handle is not endpoint state to hand back at exit.

Helpers need the same state carry:

```python
ab = load_next(ab, params)
```

Return the current value, not a snapshot saved before a loop. The checks detect ownership
and missing returns, not stale snapshots. Returned ends supply the final phases for producer
tails; returned cursors also let tracing close any remaining work range.

## Choose placement in the constructor

```python
from dataclasses import dataclass
from transformer_nuggets.cute.warpdrive import Instance, RoleConfig


@dataclass(frozen=True)
class AttentionRoles:
    softmax: RoleConfig
    mma: RoleConfig


class Attention(WarpSpecialized):
    def __init__(self, registers):
        self.roles = AttentionRoles(
            softmax=RoleConfig([range(4), range(4, 8)], regs=registers.softmax),
            mma=RoleConfig(12, regs=registers.other, cta="leader"),
        )
        ...  # channels and storage

    @role(config=lambda kernel: kernel.roles.softmax)
    @cute.jit
    def softmax(self, params, *, stage: Instance): ...
```

The selector is a host callable returning a `RoleConfig`. A named function works too.
`describe()`, tracing or a compile-cache lookup can trigger the first collection; the result
is then fixed for that kernel instance. Construct another instance to change it; mutating
or recollecting declarations after the first collection is unsupported. Selectors are host
code and must not depend on runtime tensor contents.

`RoleConfig` is frozen and copies lists of ranges into tuples. It has the same `warps`,
`regs`, `cta` and `raw` fields as the decorator. Don't mix `config=` with explicit placement
arguments. Changing placement does not change your tensor partitions; update those yourself.

The FA4 example accepts a `Fa4Roles` configuration:

```python
from dataclasses import replace
from transformer_nuggets.cute.warpdrive.examples.fa4_fwd import Fa4Roles, fa4_fwd

roles = Fa4Roles()
roles = replace(
    roles,
    softmax=replace(roles.softmax, regs=184),
    correction=replace(roles.correction, regs=96),
)
out, lse = fa4_fwd(q, k, v, roles=roles)
```

## Compile keys

The examples compile through [`jit_cache`](../../jit_cache/README.md), which keeps compiled
kernels on disk across processes. Make your compile helper a module-level `@jit_cache`
function of the kernel:

```python
@jit_cache(uncached=lambda op: op.trace_protocol)  # a protocol trace needs a real trace
def compile_kernel(op: MyKernel):
    return compile_tvm_ffi(op, *fake_args, name=op.get_name())
```

The kernel's key is `op.__cute_cache_key__()`: its constructor call (defaults filled in,
frozen when the constructor returns), the cluster, each work stream,
`iket`/`iket_waits`/`trace_protocol`/`alias_check`, the resolved roles, and the source of the
modules defining the kernel's classes. So `Gemm()` and
`Gemm(ab_stages=3)` share an entry, a changed budget or placement does not, and editing a
kernel subclass in your own file recompiles it. Pass static choices to the constructor:
assigning another public attribute after construction (`op.tile_n = 128`) makes the key raise
`TypeError`, because the constructor call no longer describes the kernel. The cluster, work
streams and those four settings may be reassigned. Changes inside objects the constructor
built (`op.ab.n = 4`) are not seen at all, so don't make them.

A cached kernel was not traced, but `describe()` still shows the trace's findings: `jit_cache`
restores the TMA, SMEM and alias reports and the opt-outs saved with the build, and repeats
their warnings. Kernels pickle as their constructor call, so `compile_many` can build several
in parallel.

Define kernels in source files: CuTeDSL reads the source again for AST preprocessing.
