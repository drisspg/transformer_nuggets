---
name: authoring-warpdrive
description: Authors and ports warp-specialized CuTeDSL kernels using warpdrive. Use when building a kernel from roles, channels and work streams, or adapting an existing kernel to that API.
---

# Authoring warpdrive kernels

This skill lives with the library. Resolve its relative paths from this directory, using the
same checkout for docs, examples and imported code. The repository docs and implementation
are the API authority; do not substitute another checkout's newer API.

Use this for a new warpdrive kernel or a port to managed roles. A standalone CuTeDSL kernel
that doesn't use warpdrive needs the usual CuTeDSL guidance, not this workflow. Layer-internal
changes also need the maintainer contract and the repository's layer validation tiers.

## Read the authoring path

Read [Write a kernel](docs/writing-kernels.md) and the [safety contract](GUARANTEES.md).
Then read the closest complete example, including its host wrapper, launch and compile helper:

- [GEMM](examples/gemm.py): single-CTA TMA → MMA → epilogue.
- [Persistent GEMM](examples/persistent_gemm.py): repeated work, schedules and clusters.
- [FA4 forward](examples/fa4_fwd.py): slots, fan-out and overlapping lifetimes.
- [Swap-AB decode](examples/fa4_decode_swapab.py): round-robin and named channels; not a
  persistent template.
- [Packed work](examples/packed_work.py): mapping a device-built list, not a complete kernel.

Read the linked topic guide when using that feature. The
[numbered catalog](docs/contract-reference.md) is for a specific rejection or layer change,
not a checklist of error codes to reproduce.

## Establish the protocol before implementing

State the input/output contract, numerical reference and supported shapes. For each work
item, record which CTA/cluster owns the result, the roles and their warps, and each channel's
producer, consumers, epoch count and storage lifetime. A short diagram or table is enough.
Include feedback edges: identify the initial available value or capacity that allows progress.

Keep role math and tensor layouts explicit. Choose managed channels when they express the
handoff, not an additional abstraction around the CuTeDSL math. Begin without storage aliases
unless the algorithm requires them. Preserve the ability to use raw code where necessary.

When adapting an existing kernel, compare both protocols before replacing synchronization:
which operation completes the access, which threads signal, which CTA's barrier is targeted,
and when the stage may be overwritten. Similar-looking acquire/commit calls are not evidence
of equivalent semantics.

## Implement against the contract

Follow the guide's lifecycle, prologue and launch sections. In particular:

- Return current endpoints, cursors and turnstile seats; helpers return and callers reassign
  state. Do not restore a saved pre-loop snapshot.
- Keep the declared participants in each handoff; predicate data accesses, not lane-dependent
  participation. Account for epochs across work items, including odd trip counts.
- Bind views to the buffers they actually partition. Handle stage projection is not a proof
  of storage identity, layout validity or safe aliasing.
- Use the current example's compile helper and fake runtime ABI. Put static choices in the
  cache key, validate runtime alignment/compactness before relying on them, and keep scheduler
  workspaces disjoint across concurrent launches.
- For raw synchronization, overrides or opt-outs, explain the exact ordering supplied instead.
  Never suppress a check just to obtain a successful compile. Inspect unverified TMA/alias
  sites rather than interpreting silence as proof.

## Validate and hand back

Use the repository's [testing tiers](../../../AGENTS.md#warpdrive-testing-tiers) and its GPU
reservation rules. Verify imports point at this checkout before running tests. Keep launches
bounded by a process timeout; a timeout is a failure requiring diagnosis, not a performance
measurement or permission to extend the run.

Test the operation's numerical and lifecycle contract: one item, stage/phase wraparound,
supported empty/ragged work, multiple persistent items, repeated launches, and the advertised
schedule/cluster variants. For newly introduced async or alias lifetimes, add targeted
ordering tests and appropriate sanitizer checks. Don't run long campaigns merely because the
maintainer catalog lists them; follow the repository's scope and authorization rules.

Establish correctness before benchmarking. Compare a fixed workload and reference, keep
buffers alive throughout timed/replayed work, and report codegen/resource changes when they
are relevant. A compile-only check or a profiler plan is not a correctness or speed result.

Deliver the runnable kernel and wrapper, tests and reproduction commands, the role/channel
sketch, and any remaining limitations or raw/unverified sites. Report tests that couldn't run
as validation gaps. Finish the authorized implementation and checks; don't stop at a skeleton
or silently broaden the supported input domain.
