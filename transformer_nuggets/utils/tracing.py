import itertools
import weakref
from functools import partial

import torch
import torch.overrides
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils.weak import WeakIdRef

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}


class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


class LoggingMode(TorchDispatchMode):
    """Example usage:
    with LoggingMode():
        torch.nn.functional.linear(torch.randn(3, 4), torch.randn(5, 4))
    """

    next_id: int

    def __init__(self, with_type: bool = True, collect_logs=False):
        self.memo = {}
        self.next_id = 0
        self.with_type = with_type
        self.collect_logs = collect_logs
        self.logs = []

    def _shortid(self, t: torch.Tensor) -> int:
        o = WeakIdRef(t)
        weak_self = weakref.ref(self)

        def del_memo():
            self = weak_self()
            if self is None:
                return
            self.memo.pop(o, None)

        weakref.finalize(t, del_memo)
        if o not in self.memo:
            self.memo[o] = self.next_id
            self.next_id += 1
        return self.memo[o]

    def _fmt(self, a: object, with_type: bool = False) -> str:
        if isinstance(a, torch.Tensor):
            maybe_type = ""
            if with_type and self.with_type:
                maybe_type = f": {dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]"
            return Lit(f"${self._shortid(a)}{maybe_type}")
        else:
            return a

    def str_logs(self):
        return "\n".join(self.logs)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        fmt_args = ", ".join(
            itertools.chain(
                (repr(tree_map(self._fmt, a)) for a in args),
                (f"{k}={tree_map(self._fmt, v)}" for k, v in kwargs.items()),
            )
        )
        fmt_rets = repr(tree_map(partial(self._fmt, with_type=True), rs))
        log_msg = f"{fmt_rets} = {torch.overrides.resolve_name(func)}({fmt_args})"
        if self.collect_logs:
            self.logs.append(log_msg)
        else:
            print(log_msg)
        return rs
