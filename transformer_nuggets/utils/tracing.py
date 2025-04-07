import itertools
import weakref
from functools import partial

import torch
import torch.overrides
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_map_only
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
    torch.float8_e4m3fn: "f8e4m3fn",
    torch.float8_e5m2: "f8e5m2",
}

abbr_to_dtype = {v: k for k, v in dtype_abbrs.items()}


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


def get_error_string(func, types, args, kwargs, output_has_nan, output_has_inf):
    error_string = f"Function {func}(*{args}, **{kwargs}) returned "
    if output_has_nan:
        error_string += "a NaN"
    if output_has_inf:
        if output_has_nan:
            error_string += " and an Inf"
        else:
            error_string += "an Inf"
    return error_string


def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


class NanInfDetect(TorchDispatchMode):
    """This mode can be helpful for debugging NaNs or Infs in your code.
    Example usage:
    ```Python
        >>> a = torch.tensor([0.,])
        >>> with NanDetect():
        >>>    print(torch.div(a, a)
        RuntimeError: Function aten.div.Tensor(*(tensor([0.]), tensor([0.])), **{}) returned a NaN
    ```
    Args:
        do_breakpoint: If True, will call `breakpoint()` when a NaN or Inf is detected.
        distributed: use torch.distributed.breakpoint() instead of breakpoint()
    """

    def __init__(self, do_breakpoint: bool = False, distributed: bool = False):
        super().__init__()
        self.do_breakpoint = do_breakpoint
        self.distributed = distributed

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        res = func(*args, **kwargs)
        if not isinstance(res, (torch.Tensor, list, tuple)):
            return res
        try:
            output_has_nan = any(
                ensure_list(tree_map_only(torch.Tensor, lambda x: torch.any(torch.isnan(x)), res))
            )
            output_has_inf = any(
                ensure_list(tree_map_only(torch.Tensor, lambda x: torch.any(torch.isinf(x)), res))
            )
        except:  # noqa: E722
            return res
        if output_has_nan or output_has_inf:
            if self.do_breakpoint:
                if self.distributed:
                    torch.distributed.breakpoint()
                else:
                    breakpoint()
            raise RuntimeError(
                get_error_string(func, types, args, kwargs, output_has_nan, output_has_inf)
            )
        return res
