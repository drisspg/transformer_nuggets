"""Compare the warpdrive swap-AB decode port with FA4's SM100 swap-AB decode at matched splits.

For each shape FA4 runs through its interface (``FLASH_ATTENTION_SM100_BACKEND=swapab``); its split
partials, split count and per-sequence split table are captured at the combine call. The port then
runs with the same split count and table. Reported per shape:

* partial O / LSE: max |port - FA4| over the splits each sequence owns, and the bitwise-equal
  fraction; final O against an fp32 reference for both;
* CUDA-graph replay, per kernel (CUPTI durations, median over replays): FA4's decode kernel vs the
  port's, plus the whole call (FA4: prepare + decode + combine; port: decode + combine) as graph
  replay time.

Run under ``gpu-run``: ``python scripts/bench_warpdrive_decode.py [--reps 200]``.
"""

import argparse
import math
import os
import statistics

import torch

os.environ.setdefault("FLASH_ATTENTION_SM100_BACKEND", "swapab")

from flash_attn.cute import interface  # noqa: E402

from transformer_nuggets.cute.warpdrive.examples.fa4_decode_swapab import (  # noqa: E402
    decode_partials,
    fa4_decode,
)

# (batch, kv capacity, heads, kv heads, seqused pattern): "full" or "ragged" (lengths cycle
# through 1/8, 1/2, 1 and 3/4 of the capacity, so per-sequence split counts differ).
SHAPES = [
    (1, 8192, 8, 1, "full"),
    (4, 16384, 32, 8, "full"),
    (8, 32768, 32, 8, "full"),
    (8, 65536, 8, 1, "ragged"),
    (16, 32768, 64, 8, "ragged"),
    (1, 32768, 32, 1, "full"),
    (2, 131072, 16, 1, "ragged"),
]


def lengths(batch, capacity, pattern):
    if pattern == "full":
        return None
    fractions = (0.125, 0.5, 1.0, 0.75)
    used = [max(1, int(capacity * fractions[i % 4])) for i in range(batch)]
    return torch.tensor(used, device="cuda", dtype=torch.int32)


def fa4_call(q, k, v, used):
    captured = {}
    combine = interface._flash_attn_fwd_combine

    def capture(out_partial, lse_partial, *args, **kwargs):
        table = kwargs.get("num_splits_dynamic_ptr")
        captured.update(o=out_partial.clone(), lse=lse_partial.clone(), table=table)
        return combine(out_partial, lse_partial, *args, **kwargs)

    capture.compile_cache = combine.compile_cache  # the combine reads it by global name
    interface._flash_attn_fwd_combine = capture
    try:
        out, lse, *_ = interface._flash_attn_fwd(
            q, k, v, seqused_k=used, max_seqlen_k=k.shape[1], num_splits=0, return_lse=True
        )
    finally:
        interface._flash_attn_fwd_combine = combine
    return out, lse, captured


def reference(q, k, v, used):
    b, _, h, _ = q.shape
    qpk = h // k.shape[2]
    outs = []
    for i in range(b):
        n = k.shape[1] if used is None else int(used[i])
        kk = k[i, :n].float().repeat_interleave(qpk, 1)
        vv = v[i, :n].float().repeat_interleave(qpk, 1)
        s = torch.einsum("hd,thd->ht", q[i, 0].float(), kk) / math.sqrt(q.shape[-1])
        outs.append(torch.einsum("ht,thd->hd", torch.softmax(s, -1), vv))
    return torch.stack(outs)[:, None]


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def replay_times(graph, reps):
    """(median graph replay us, {kernel name: median us})."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(10):
        graph.replay()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for s, e in zip(starts, ends):
            s.record()
            graph.replay()
            e.record()
        torch.cuda.synchronize()
    replay = statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    kernels = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.device_time > 0:
            kernels.setdefault(ev.name, []).append(ev.device_time)
    return replay, {name: statistics.median(t) for name, t in kernels.items()}


def decode_kernel(kernels, needle):
    names = [n for n in kernels if needle in n]
    return kernels[names[0]] if names else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--shapes", type=int, nargs="*", help="indices into SHAPES")
    args = parser.parse_args()
    shapes = [SHAPES[i] for i in args.shapes] if args.shapes else SHAPES
    for b, s, h, hkv, pattern in shapes:
        torch.manual_seed(0)
        q = torch.randn(b, 1, h, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, s, hkv, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, s, hkv, 128, device="cuda", dtype=torch.bfloat16)
        used = lengths(b, s, pattern)
        out_fa4, _, cap = fa4_call(q, k, v, used)
        if "o" not in cap:
            print(f"b{b} s{s} h{h}/{hkv} {pattern}: FA4 chose one split; skipped")
            continue
        splits, table = cap["o"].shape[0], cap["table"]
        o, lse = decode_partials(q, k, v, splits, seqused_k=used, splits_per_seq=table)
        out, _ = fa4_decode(q, k, v, splits, seqused_k=used, splits_per_seq=table)
        counts = table.tolist() if table is not None else [splits] * b
        owned = torch.tensor(
            [[i < counts[j] for j in range(b)] for i in range(splits)], device="cuda"
        )
        o_fa4, lse_fa4 = cap["o"], cap["lse"][:, :, 0]  # captured transposed: (ns, b, 1, h)
        lse = lse[..., 0]
        do = (o - o_fa4)[owned].abs().max().item()
        same = torch.equal(o[owned], o_fa4[owned])
        fin = owned[..., None] & torch.isfinite(lse_fa4)
        dl = (lse - lse_fa4)[fin].abs().max().item() if fin.any() else 0.0
        ref = reference(q, k, v, used)
        err_port = (out.float() - ref).abs().max().item()
        err_fa4 = (out_fa4.float() - ref).abs().max().item()
        g_fa4 = graph_of(lambda: interface._flash_attn_fwd(
            q, k, v, seqused_k=used, max_seqlen_k=s, num_splits=0, return_lse=True
        ))  # fmt: skip
        # The port's wrapper fills seqused_k when it is None; build it outside the graph.
        used_port = used if used is not None else torch.full((b,), s, device="cuda").int()
        g_port = graph_of(
            lambda: fa4_decode(q, k, v, splits, seqused_k=used_port, splits_per_seq=table)
        )
        t_fa4, k_fa4 = replay_times(g_fa4, args.reps)
        t_port, k_port = replay_times(g_port, args.reps)
        # Interleave a second round so drift affects both.
        t_fa4b, k_fa4b = replay_times(g_fa4, args.reps)
        t_portb, k_portb = replay_times(g_port, args.reps)
        dec_fa4 = min(decode_kernel(k_fa4, "attn_kernel"), decode_kernel(k_fa4b, "attn_kernel"))
        dec_port = min(decode_kernel(k_port, "warpdrive"), decode_kernel(k_portb, "warpdrive"))
        print(
            f"b{b} s{s} h{h}/{hkv} {pattern} splits {splits} "
            f"{'per-seq ' + str(counts) if table is not None else 'static'}\n"
            f"  partials: max|dO| {do:.1e} bitwise {same}, max|dLSE| {dl:.1e}; "
            f"final vs fp32: port {err_port:.1e}, FA4 {err_fa4:.1e}\n"
            f"  decode kernel us: FA4 {dec_fa4:.2f}, port {dec_port:.2f} "
            f"({dec_port / dec_fa4:.3f}x); graph us: FA4 {min(t_fa4, t_fa4b):.2f}, "
            f"port {min(t_port, t_portb):.2f}",
            flush=True,
        )
    print(sorted(k_fa4), sorted(k_port))


if __name__ == "__main__":
    main()
