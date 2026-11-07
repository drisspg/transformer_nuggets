"""Host side of a ``"globaltimer"`` trace (``tracing.py``): ``profile_session`` over a warpdrive
kernel's profile buffer.

Not imported by the kernel modules or the package ``__init__``: a compile key hashes every
module a kernel imports, and decoding has nothing to do with the compiled code.
"""

from __future__ import annotations

import warnings
from collections import Counter

from transformer_nuggets.cute.profiler.host import profile_session
from transformer_nuggets.cute.profiler.postprocessors import compose, rename_threads
from transformer_nuggets.cute.warpdrive.protocol import ProtocolError


def trace_session(
    kernel, num_ctas: int, *, post_process_events=None, post_process_trace=None, **kwargs
):
    """``profile_session`` for one launch of a compiled ``enable_trace("globaltimer")`` kernel
    with ``num_ctas`` CTAs. Pass ``session.prof.tensor`` as the kernel's trace buffer
    (``run_roles(..., trace_buffer=...)``). After the block, ``session.events`` holds the
    decoded ranges (``tag_name`` = range name, ``unit_id`` = CTA, ``tid`` = the role's first
    warp, ``payload`` = a work item's index or a ``trace_region``'s payload) and
    ``session.trace`` the Perfetto trace, one process per CTA and one thread per role.

    ``post_process_events``/``post_process_trace`` run after that grouping; ``kwargs`` go to
    ``profile_session`` (``trace_path``, ``device``, ...). Warns when a role filled its
    ``events_per_role`` records in some CTA (later ranges were dropped)."""
    if kernel.trace_backend != "globaltimer":
        raise ProtocolError(
            f"trace_session needs enable_trace('globaltimer'), not {kernel.trace_backend!r}"
        )
    if not getattr(kernel, "_ws_traced", False):
        raise ProtocolError(
            "trace_session needs a compiled kernel: range names come from its trace"
        )
    roles = kernel._ws_roles
    lanes = {r.first_warp: r.name for r in roles}
    capacity = kernel.trace_events_per_role

    def group(events, ctx):
        counts = Counter((e.unit_id, e.tid) for e in events)
        full = sorted({lanes[tid] for (_, tid), n in counts.items() if n >= capacity})
        if full:
            warnings.warn(
                f"warpdrive trace: role(s) {full} filled their {capacity} records in some CTA; "
                "later ranges were dropped (raise enable_trace(events_per_role=...))",
                stacklevel=2,
            )
        for e in events:
            e.pid = e.unit_id
        return events if post_process_events is None else post_process_events(events, ctx)

    names = rename_threads(lanes)
    return profile_session(
        max_events_per_unit=len(roles) * capacity,
        num_units=(num_ctas, "CTA"),
        tag_names=kernel._ws_trace_tags,
        post_process_events=group,
        post_process_trace=names
        if post_process_trace is None
        else compose(names, post_process_trace),
        **kwargs,
    )
