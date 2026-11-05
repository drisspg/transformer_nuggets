"""Declared CTA named barriers: one hardware barrier ID and one thread count per declaration.

``NamedBarrier(threads=N)`` is a kernel attribute, declared in ``__init__`` like a channel. The
kernel allocates its ID together with the named-barrier channels' IDs (``kernel.py``); role
bodies call ``self.<name>.sync()`` or ``.arrive()``. Each call records a trace event, so the
alias diagnostic sees the barrier, the kernel rejects hand-picked ``alias.named_barrier`` IDs
that collide with it, and ``describe()`` lists the roles that used it.

The resource supplies the ID and a consistent count, nothing more: which threads arrive in
which generation, and why a fast participant cannot arrive twice before slow ones finish a
generation, remain the author's protocol (docs/channels.md, "Named barriers").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cutlass import cute

from transformer_nuggets.cute.warpdrive.channels import _named_arrive, _named_sync
from transformer_nuggets.cute.warpdrive.protocol import ProtocolError, current_trace

WARP_SIZE = 32
MAX_THREADS = 1024  # CUDA's block limit


@dataclass(eq=False)
class NamedBarrier:
    """One CTA named barrier whose ID the kernel allocates; ``threads`` arrivals complete it.

    ``sync()`` arrives and waits for the generation to complete (``barrier.cta.sync``),
    ``arrive()`` arrives without waiting (``barrier.cta.arrive``). Both are unaligned, so lanes
    and warps may reach them from different code sites. ``aligned=True`` emits the aligned
    forms (``bar.sync``/``bar.arrive``), which require every thread of each participating warp
    to execute the same instruction convergently; use it only where that holds and it matters.

    ``id`` is the allocated barrier ID for exceptional native calls (a library that takes an
    ID). Such calls are invisible to the layer: no diagnostic event and no collision check.
    """

    threads: int
    name: str = field(default="", init=False)
    _id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.threads) is not int
            or not WARP_SIZE <= self.threads <= MAX_THREADS
            or self.threads % WARP_SIZE
        ):
            raise ValueError(
                f"NamedBarrier threads must be an int multiple of {WARP_SIZE} in "
                f"[{WARP_SIZE}, {MAX_THREADS}] (bar.sync counts whole warps), got {self.threads!r}"
            )

    @property
    def id(self) -> int:
        if self._id is None:
            raise ProtocolError(
                f"NamedBarrier {self.name or '(unnamed)'} has no ID yet: declare it as a kernel "
                "attribute; the kernel allocates IDs when it collects its declarations "
                "(describe(), compile)"
            )
        return self._id

    def sync(self, *, aligned: bool = False):
        """Arrive, then wait until ``threads`` threads have arrived in this generation."""
        if aligned:
            cute.arch.barrier(barrier_id=self.id, number_of_threads=self.threads)
        else:
            _named_sync(self.id, self.threads)
        self._record("barrier")

    def arrive(self, *, aligned: bool = False):
        """Arrive without waiting; the generation completes when ``threads`` have arrived."""
        if aligned:
            cute.arch.barrier_arrive(barrier_id=self.id, number_of_threads=self.threads)
        else:
            _named_arrive(self.id, self.threads)
        self._record("barrier_arrive")

    def _record(self, kind: str):
        # "barrier" is the event alias.named_barrier records; an arrive orders nothing for the
        # arriving thread, so it gets its own kind the alias diagnostic ignores.
        trace = current_trace()
        if trace is not None:
            trace.event(kind, threads=self.threads, id=self.id, barrier=self)
