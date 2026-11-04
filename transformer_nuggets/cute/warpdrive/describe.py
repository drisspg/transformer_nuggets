"""Plain-text summary of a ``WarpSpecialized`` kernel: what the layer derived from it.

``describe_kernel`` reads only host-side state collected by ``WarpSpecialized._collect`` (and,
after a compile, the TMA check reports, check opt-outs, and the traced protocol), so the output
is stable for a given declaration. Sections: header, roles, channels, turnstiles, raw barriers,
streams, TMEM, opt-outs, traced protocol.
"""

from __future__ import annotations

import textwrap

from transformer_nuggets.cute.warpdrive.channels import Threads, Tma, Umma
from transformer_nuggets.cute.warpdrive.scheduling import _tile_grid, scheduler_role

_RANK = {0: "leader", 1: "peer"}
WIDTH = 100  # wrap channel flows past this many columns
_LABEL = 8  # width of the channel detail labels ("override")


def describe_kernel(kernel) -> str:
    sections = [
        _header(kernel),
        _roles(kernel),
        _channels(kernel),
        _turnstiles(kernel),
        _raw_barriers(kernel),
        _streams(kernel),
        _tmem(kernel),
        _opt_outs(kernel),
        _protocol(kernel),
    ]
    return "\n\n".join(s for s in sections if s)


# --------------------------------------------------------------------------- #
# Value formatting
# --------------------------------------------------------------------------- #


def fmt_counts(counts) -> str:
    """Per-slot arrival counts: ``1 x6`` when uniform, ``[1, 4]`` otherwise."""
    counts = list(counts)
    if len(set(counts)) == 1:
        return str(counts[0]) if len(counts) == 1 else f"{counts[0]} x{len(counts)}"
    return f"[{', '.join(str(c) for c in counts)}]"


def fmt_per_cta(per_rank: dict, ranks, by_rank: bool = False) -> str:
    """Counts of the CTAs ``ranks`` (``{rank: per-slot list}``), labelled by rank when
    ``by_rank`` or when they differ."""
    ranks = list(ranks)
    values = [fmt_counts(per_rank[r]) for r in ranks]
    if not by_rank and len(set(values)) == 1:
        return values[0]
    return ", ".join(f"{_RANK.get(r, f'rank {r}')} {v}" for r, v in zip(ranks, values))


def _fmt_value(key: str, value, cluster_size: int) -> str:
    if key in ("full_counts", "empty_counts"):
        return fmt_per_cta(value, range(cluster_size))
    if key in ("full_mask", "empty_mask"):
        return f"{value:#04b}"
    return str(value)


def _sites(sites) -> str:
    """``a.py:1, a.py:7, b.py:3`` -> ``a.py:1,7, b.py:3``."""
    by_file = {}
    for site in sorted(sites, key=_site_key):
        path, _, line = site.rpartition(":")
        by_file.setdefault(path, []).append(line)
    return ", ".join(f"{path}:{','.join(lines)}" for path, lines in by_file.items())


def _site_key(site: str):
    path, _, line = site.rpartition(":")
    return (path, int(line) if line.isdigit() else 0)


def _table(rows: list[list[str]], right: set[int] = frozenset(), indent: str = "  ") -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [
            c.rjust(w) if i in right else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))
        ]
        lines.append((indent + "  ".join(cells)).rstrip())
    return lines


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _header(kernel) -> str:
    lines = [f"{type(kernel).__name__}: {kernel.num_warps} warps, {kernel.num_threads} threads"]
    ctx = kernel._ws_ctx_type
    rows = [
        ["cluster", kernel.cluster.describe()],
        ["params", ctx.__name__ if ctx is not None else "unannotated (any object)"],
    ]
    return "\n".join(lines + _table(rows))


def _roles(kernel) -> str:
    pair = kernel.cluster.size > 1
    rows = [["warps", "role", "regs", "cta", "raw"]]
    entries = [(r.first_warp, r.warps, r.name, r.regs, r.cta, r.raw) for r in kernel._ws_roles]
    entries += [(w.start, w, "(padding)", regs, "", False) for w, regs in kernel._ws_padding]
    for _, warps, name, regs, cta, raw in sorted(entries, key=lambda e: e[0]):
        span = str(warps.start) if len(warps) == 1 else f"{warps.start}-{warps.stop - 1}"
        rows.append(
            [
                span,
                name,
                "-" if regs is None else str(regs),
                cta if pair else ("" if name == "(padding)" else "all"),
                "raw" if raw else "",
            ]
        )
    return "\n".join(["Roles", *_table(rows, right={2})])


def _mech(mech) -> str:
    if isinstance(mech, Tma):
        return f"Tma({mech.tx_bytes} B)"
    return getattr(mech, "__name__", repr(mech))


def _bound(bound) -> str:
    return ", ".join(r.name + ("" if s is None else f"@{s}") for r, s in bound)


def _channels(kernel) -> str:
    chans = kernel._ws_chans
    if not chans:
        return ""
    cs = kernel.cluster.size
    width = max(len(n) for n in chans)
    pad = " " * (2 + width + 2)
    lines = ["Channels"]
    for name, ch in chans.items():
        kind = f"Slots({ch.n})" if ch.fixed else f"Ring({ch.n})"
        head = f"  {name:<{width}}  {kind:<9}"
        flow = f"{_bound(ch.producers)} -> {_bound(ch.consumers)}"
        lines += textwrap.wrap(
            flow,
            WIDTH,
            initial_indent=head,
            subsequent_indent=" " * (len(head) + 4),
            break_on_hyphens=False,
        )
        details = []
        sc = ch.scope
        full_ranks = sc.full_ctas if sc.pair else range(cs)
        empty_ranks = sc.empty_ctas if sc.pair else range(cs)
        full = fmt_per_cta(sc.full_counts, full_ranks, by_rank=sc.pair)
        if sc.tx is not None:
            full += f", tx {sc.tx} B" + (" (leader-armed)" if sc.pair else "")
        details.append(("signal", f"{_mech(ch.producer)} -> {_mech(ch.consumer)}"))
        details.append(("full", full))
        details.append(("empty", fmt_per_cta(sc.empty_counts, empty_ranks, by_rank=sc.pair)))
        if cs > 1:
            scope = ["pair-scoped" if sc.pair else "CTA-local"]
            if sc.pair:
                if ch.producer is Umma:
                    scope.append(f"commit_mask {sc.full_mask:#04b}")
                if ch.consumer is Umma:
                    scope.append(f"release_mask {sc.empty_mask:#04b}")
                if Threads in (ch.producer, ch.consumer):
                    scope.append(f"remote_scope {ch.remote_scope}")
                if sc.tail:
                    scope.append("tail")
            details.append(("scope", ", ".join(scope)))
        extra = []
        if ch.payload is not None:
            dtype, shape = ch.payload
            extra.append(f"payload {getattr(dtype, '__name__', dtype)}{tuple(shape)}")
        if ch.commit_fence:
            extra.append(f"commit_fence {ch.commit_fence}")
        if ch.release_fence:
            extra.append(f"release_fence {ch.release_fence}")
        if extra:
            details.append(("storage" if ch.payload is not None else "fences", ", ".join(extra)))
        details += [("override", o) for o in _overrides(ch, cs)]
        tma = _tma_status(kernel, name, ch)
        if tma:
            details.append(("tma", tma))
        lines += [f"{pad}{k:<{_LABEL}}  {v}" for k, v in details]
    return "\n".join(lines)


def _overrides(ch, cluster_size: int) -> list[str]:
    """``key derived -> effective`` for every overridden value."""
    out = []
    derived = ch.derived_scope
    for key in ch.override or {}:
        ours, theirs = getattr(derived, key), getattr(ch.scope, key)
        note = "  (differs from derived)" if ours != theirs else ""
        out.append(
            f"{key} {_fmt_value(key, ours, cluster_size)} -> "
            f"{_fmt_value(key, theirs, cluster_size)}{note}"
        )
    if ch.remote_scope_overridden:
        ours, theirs = ch.derived_remote_scope, ch.remote_scope
        note = "  (differs from derived)" if ours != theirs else ""
        out.append(f"remote_scope {ours} -> {theirs}{note}")
    return out


def _tma_status(kernel, name: str, ch) -> str:
    report = kernel._ws_tma.get(name)
    if report is not None:
        return report.status(_sites)
    if isinstance(ch.producer, Tma) and not ch.internal:
        return "not checked yet (compile to check the copies)"
    return ""


def _turnstiles(kernel) -> str:
    if not kernel._ws_turns:
        return ""
    rows = [
        [name, " -> ".join(r.name for r in ts.parties), "" if ts.enabled else "(disabled)"]
        for name, ts in kernel._ws_turns.items()
    ]
    return "\n".join(["Turnstiles", *_table(rows)])


def _raw_barriers(kernel) -> str:
    if not kernel._ws_raw_bars:
        return ""
    rows = []
    for name, b in kernel._ws_raw_bars.items():
        init = "user-initialized" if b.counts is None else f"layer-initialized {b.counts}"
        rows.append([name, f"{b.n} mbarriers", init, "user-driven (unchecked)"])
    return "\n".join(["Raw barriers", *_table(rows)])


def _streams(kernel) -> str:
    if not kernel._ws_streams:
        return ""
    lines = ["Streams"]
    for name, st in kernel._ws_streams.items():
        pad = " " * (2 + len(name) + 2)
        lines.append(f"  {name}  {st.describe()}")
        sched = next((r for r in kernel._ws_roles if r.fn is scheduler_role), None)
        if sched is not None:
            lines.append(f"{pad}scheduler warp {sched.first_warp} broadcasts over {name}_ring")
        else:
            grid = (
                "; launch grid = tile space (1-D past 65535 n_tiles or l)"
                if _tile_grid(st)
                else ""
            )
            lines.append(f"{pad}decoded locally in every role{grid}")
        users = [r for r in st.roles if sched is None or r != sched.name]
        lines.append(f"{pad}roles: {', '.join(users)}")
    return "\n".join(lines)


def _tmem(kernel) -> str:
    plan = kernel.tmem_plan
    if plan is None:
        return ""
    head = f"TMEM: {plan.used_cols} columns used, {plan.alloc_cols} allocated"
    return f"{head}\n{plan.describe()}"


def _opt_outs(kernel) -> str:
    if not kernel._ws_opt_outs:
        return ""
    return "\n".join(["Unchecked opt-outs", *(f"  {o.describe()}" for o in kernel._ws_opt_outs)])


def _protocol(kernel) -> str:
    log = kernel._ws_log
    if log is None or not log.ops:
        return ""
    body = "\n".join(f"  {line}" for line in log.describe().splitlines())
    return f"Traced protocol\n{body}"
