import cmath
import math
from typing import Any, Dict, Tuple


def _split_params(spec: Tuple[Any, ...]) -> Tuple[str, Dict[str, Any]]:
    name = spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict):
        params = dict(spec[1])
    elif len(spec) > 1:
        params = {"positional": spec[1:]}
    else:
        params = {}
    return name, params


def _pop_positional(params: Dict[str, Any], defaults: Tuple[Tuple[str, float], ...]):
    """Populate named parameters from positional fallbacks."""
    if "positional" not in params:
        return {k: params.get(k, default) for k, default in defaults}
    pos = params["positional"]
    resolved = {}
    for idx, (key, default) in enumerate(defaults):
        resolved[key] = float(pos[idx]) if idx < len(pos) else default
    return resolved


def get_decay(spec):
    if spec is None:
        return lambda d: 1.0
    if not isinstance(spec, tuple):
        raise ValueError(f"Unsupported decay spec: {spec}")

    name, params = _split_params(spec)
    if name in ("const",):
        return lambda d: 1.0
    if name in ("lin", "inv"):
        p = float(params.get("p", params.get("positional", (1.0,))[0] if params.get("positional") else 1.0))
        return lambda d: 1.0 / (d ** p)
    if name == "exp":
        lam = float(params.get("lambda", params.get("positional", (1.0,))[0] if params.get("positional") else 1.0))
        return lambda d: math.exp(-lam * (d - 1))
    if name in ("gauss", "gaussian"):
        values = _pop_positional(params, (("mu", 1.0), ("sigma", 1.0)))
        mu = values["mu"]
        sigma = max(values["sigma"], 1e-9)

        def _gauss(d: int) -> float:
            return math.exp(-((d - mu) ** 2) / (2.0 * sigma * sigma))

        return _gauss
    if name in ("morlet", "morl"):
        values = _pop_positional(
            params,
            (("omega", 5.0), ("sigma", 1.0), ("scale", 1.0), ("shift", 0.0)),
        )
        omega = values["omega"]
        sigma = max(values["sigma"], 1e-9)
        scale = max(values["scale"], 1e-9)
        shift = values["shift"]

        def _morlet(d: int):
            t = ((d - 1) - shift) / scale
            envelope = math.exp(-0.5 * (t / sigma) ** 2)
            return envelope * cmath.exp(1j * omega * t)

        return _morlet
    if name in ("mexican", "mexh"):
        values = _pop_positional(
            params,
            (("sigma", 1.0), ("scale", 1.0), ("shift", 0.0)),
        )
        sigma = max(values["sigma"], 1e-9)
        scale = max(values["scale"], 1e-9)
        shift = values["shift"]

        def _mexican(d: int) -> float:
            t = ((d - 1) - shift) / scale
            u = t / sigma
            return (1.0 - u * u) * math.exp(-0.5 * u * u)

        return _mexican

    raise ValueError(f"Unsupported decay spec: {spec}")
