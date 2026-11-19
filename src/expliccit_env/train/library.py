"""The grounding library's features, and fitted formulas over them.

A base variable is a stored label or a composite (lcfs.py), on a signed
log scale (``scaled``). A feature is a base variable at the step, or under
an operator over the steps before it in its own episode: a mean, minimum
or maximum over 240 or 960 ms, or the difference over 60 ms. A feature's
value in physical units (``physical``: ms, Mbit/s, kB or none) is what a
fitted formula reads. One definition serves the formula search
(experiments/ground.py), the formulas net1 learns (an LCF such as
``4.67 - log([max_960ms(buffer_fill)] + 78.0)``, lcfs.py), and the
features a campaign stores beside strided steps (dataset.ShardWriter).

A formula is PySR's, printed by SymPy: + - * /, powers, log, numbers and
features in brackets. ``evaluate`` reads only those, so a formula is data
and never code.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Callable, Iterable
from typing import Final

import numpy as np
import numpy.typing as npt

from expliccit_env.labels import WINDOWS_MS
from expliccit_env.train.lcfs import LCF_UNITS, is_formula

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

# The operators' windows and the difference's lag, in 10 ms steps.
WINDOWS: Final = (24, 96)
CHANGE: Final = 6
OPERATORS: Final = (
    f"change_{CHANGE * 10}ms",
    *(f"{op}_{w * 10}ms" for w in WINDOWS for op in ("mean", "min", "max")),
)
_SCALE: Final = {"_bps": 1e-6, "_ns": 1e-6, "_bytes": 1e-3}
_FEATURE: Final = re.compile(r"^(\w+?_\d+ms)\((.+)\)$")
_BRACKETED: Final = re.compile(r"\[([^\[\]]+)\]")


def label(name: str) -> str:
    """A label column's name without its window."""
    return next(
        (name.removesuffix(f"_{w}ms") for w in WINDOWS_MS if name.endswith(f"_{w}ms")), name
    )


def factor(name: str) -> float:
    """A base variable's factor to its physical units."""
    if name in LCF_UNITS:
        return LCF_UNITS[name][0]
    return next((v for suffix, v in _SCALE.items() if label(name).endswith(suffix)), 1.0)


def scaled(name: str, values: npt.NDArray[np.floating]) -> F64:
    """A base variable on the library's scale: the signed log1p of its
    physical value; NaN where it is not finite."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = values.astype(np.float64) * factor(name)
        v = np.sign(v) * np.log1p(np.abs(v))
    v[~np.isfinite(v)] = np.nan
    return v


def physical(x: F64) -> F64:
    """Values on the library's scale back in physical units."""
    return np.sign(x) * np.expm1(np.abs(x))


def history(v: F64, episode: I64, step: I64, idx: I64) -> F64:
    """[anchor, age in steps, variable]: the variables (rows by episode,
    then step) at each anchor row and the steps before it in its own
    episode; NaN beyond the episode's start and where a step is missing."""
    age = np.arange(max(WINDOWS))
    past = idx[:, None] - age[None]
    row = np.maximum(past, 0)
    ok = (past >= 0) & (episode[row] == episode[idx][:, None])
    ok &= step[idx][:, None] - step[row] == age[None]
    h = v[row]
    h[~ok] = np.nan
    return h


def operators(h: F64) -> list[tuple[str, F64]]:
    """Every operator on a history (``history``), the variable itself
    first under the name "", in the order of OPERATORS."""
    out = [("", h[:, 0]), (OPERATORS[0], h[:, 0] - h[:, CHANGE])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a window with no defined row
        for w in WINDOWS:
            for op, f in (("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)):
                out.append((f"{op}_{w * 10}ms", f(h[:, :w], axis=1)))
    return out


def parse(feature: str) -> tuple[str, str]:
    """(operator, base variable) of a feature's name; operator "" for the
    variable at the step."""
    m = _FEATURE.match(feature)
    if m is None:
        return "", feature
    if m[1] not in OPERATORS:
        raise ValueError(f"{feature}: unknown operator {m[1]}; one of {OPERATORS}")
    return m[1], m[2]


def needs_history(feature: str) -> bool:
    return parse(feature)[0] != ""


def features_of(formula: str) -> list[str]:
    """The features a formula reads, in order of first use."""
    return list(dict.fromkeys(_BRACKETED.findall(formula)))


def stored(names: Iterable[str]) -> tuple[str, ...]:
    """The features to store with strided steps for these features or
    formulas: those over the steps before, which a strided table cannot
    compute."""
    out: list[str] = []
    for n in names:
        for f in features_of(n) if is_formula(n) else [n]:
            if needs_history(f) and f not in out:
                out.append(f)
    return tuple(out)


def values(
    features: Iterable[str], get: Callable[[str], npt.NDArray[np.floating]], episode: I64, step: I64
) -> dict[str, F64]:
    """Each feature in physical units at every row; ``get`` gives a base
    variable's values on rows ordered by episode, then step."""
    names = list(dict.fromkeys(features))
    parsed = [parse(f) for f in names]
    variables = list(dict.fromkeys(v for _, v in parsed))
    v = np.stack([scaled(n, get(n)) for n in variables], axis=1)
    out = {f: np.empty(len(episode)) for f in names}
    # A few thousand anchors at a time: a history is anchors x 96 x variables.
    for idx in np.array_split(np.arange(len(episode)), len(episode) // 4096 + 1):
        ops = dict(operators(history(v, episode, step, idx)))
        for f, (op, var) in zip(names, parsed, strict=True):
            out[f][idx] = physical(ops[op][:, variables.index(var)])
    return out


_BINARY: Final[dict[type[ast.operator], Callable[[F64, F64], F64]]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


def evaluate(formula: str, get: Callable[[str], F64], log: Callable[[F64], F64] = np.log) -> F64:
    """A formula's value, each bracketed feature from ``get``. NaN where a
    feature is NaN or the formula is undefined (a log of a negative)."""
    names = features_of(formula)
    text = _BRACKETED.sub(lambda m: f"x{names.index(m[1])}", formula)
    xs = [get(n) for n in names]

    def walk(node: ast.AST) -> F64:
        match node:
            case ast.BinOp(left=a, op=op, right=b) if type(op) in _BINARY:
                return _BINARY[type(op)](walk(a), walk(b))
            case ast.UnaryOp(op=ast.USub() | ast.UAdd() as op, operand=a):
                return -walk(a) if isinstance(op, ast.USub) else walk(a)
            case ast.Call(func=ast.Name(id="log"), args=[a], keywords=[]):
                return log(walk(a))
            case ast.Constant(value=float() | int() as c) if not isinstance(c, bool):
                return np.asarray(c, np.float64)  # broadcasts against the features
            case ast.Name(id=n) if re.fullmatch(r"x\d+", n):
                return xs[int(n[1:])]
            case _:
                raise ValueError(f"{formula}: {ast.dump(node)} is not + - * / ** log or a number")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return walk(ast.parse(text, mode="eval").body)
