"""IKET ranges named after roles and channels, and the matching ``PipelinePlan``.

``kernel.enable_iket()`` (before compiling) makes the layer emit IKET ranges; with it off the
layer emits nothing, so the generated code is unchanged. Range names are stable:

* ``<role>``: one range per work item of a role that loops over a ``WorkStream`` (from the
  cursor's start to its ``advance()``), else one range over the whole role. Role instances keep
  their suffix (``softmax[0]``). The CTA of a pair kernel is not in the name: IKET records the
  CTA of every range, and ``plan_from_warpdrive(kernel, cta=...)`` builds the plan of one CTA.
* ``<channel>.acquire`` / ``<channel>.wait`` / ``<channel>.peek`` / ``<channel>.tail``: the
  blocking mbarrier wait of a producer acquire, a consumer wait or preview, and a producer
  tail, nested inside the role range, so waiting is separated from working. ``<turnstile>.turn`` and ``<named barrier>.sync`` likewise.
* user regions: ``with self.trace_region("name"):`` inside a role.

``pipeline_plan.plan_from_warpdrive`` exports the layer's structure in the profiler's neutral model
(``transformer_nuggets.cute.profiler.pipeline``): one ``Role`` and one ``Region`` per role
instance, both named like the role range, one ``Resource`` per channel, and per channel a data
dependency producer -> consumer plus a reuse dependency back. The plan's iteration is the work
item, so ``load_iket_capture``, ``analyze_pipeline``, and ``write_pipeline_perfetto`` consume an
IKET capture of the kernel with no hand annotations.
"""

from __future__ import annotations

from contextlib import contextmanager

import cutlass
from cutlass import cute

from transformer_nuggets.cute.profiler.pipeline.annotations import load_iket


def push(name: str) -> None:
    load_iket().range_push(name)


def pop() -> None:
    load_iket().range_pop()


@cute.jit
def push_if(valid, name: cutlass.Constexpr):
    """Open the next work item's range only when the cursor holds one."""
    if valid:
        push(name)


@cute.jit
def pop_if(still_open):
    """Close a work item's range that a loop exit other than ``valid`` left open."""
    if still_open:
        pop()


@contextmanager
def wait_range(trace, name: str):
    """IKET range around one blocking wait (nothing when IKET is off)."""
    if trace is None or not trace.iket_waits:
        yield
        return
    push(name)
    yield
    pop()
