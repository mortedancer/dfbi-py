import math
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from .alphabet import Alphabet

Numeric = Union[int, float]
PhaseSpec = Union[
    None,
    Numeric,
    Sequence[Any],
    dict,
]


def _entropy_theta(
    seq: Sequence[str],
    alphabet: Alphabet,
    theta_min: float = 0.0,
    theta_max: float = math.pi,
    eps: float = 1e-12,
) -> float:
    """Map Shannon entropy of the sequence onto an angular range."""
    counts = np.zeros(len(alphabet.symbols), dtype=np.float64)
    for ch in seq:
        counts[alphabet.index[ch]] += 1.0
    total = counts.sum()
    if total <= 0:
        return theta_min
    probs = counts[counts > 0] / total
    entropy = float(-(probs * np.log(probs + eps)).sum())
    h_max = math.log(len(alphabet.symbols)) if alphabet.symbols else 1.0
    ratio = entropy / h_max if h_max > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return theta_min + ratio * (theta_max - theta_min)


def _parse_phase_tuple(
    spec: Sequence[Any],
    seq: Sequence[str],
    alphabet: Alphabet,
) -> Tuple[str, Tuple[float, ...]]:
    if not spec:
        raise ValueError("Empty phase specification tuple")
    mode = spec[0]
    if isinstance(mode, str):
        mode = mode.lower()
    params = spec[1:]
    if mode in ("theta", "angle", None):
        if not params:
            raise ValueError("Phase specification ('theta', value) requires a value")
        return "theta", (float(params[0]),)
    if mode == "entropy":
        theta_min = float(params[0]) if len(params) > 0 else 0.0
        theta_max = float(params[1]) if len(params) > 1 else math.pi
        return "entropy", (theta_min, theta_max)
    if mode == "linear":
        if len(params) < 2:
            raise ValueError("Phase specification ('linear', start, stop) requires two values")
        return "linear", (float(params[0]), float(params[1]))
    raise ValueError(f"Unsupported phase specification tuple: {spec}")


def _parse_phase_dict(
    spec: dict,
    seq: Sequence[str],
    alphabet: Alphabet,
) -> Tuple[str, Tuple[float, ...]]:
    mode = spec.get("mode", "theta")
    if isinstance(mode, str):
        mode = mode.lower()
    if mode == "theta":
        if "value" not in spec:
            raise ValueError("Phase dict with mode='theta' requires 'value'")
        return "theta", (float(spec["value"]),)
    if mode == "entropy":
        theta_min = float(spec.get("theta_min", 0.0))
        theta_max = float(spec.get("theta_max", math.pi))
        return "entropy", (theta_min, theta_max)
    if mode == "linear":
        start = float(spec.get("start", 0.0))
        stop = float(spec.get("stop", start))
        return "linear", (start, stop)
    raise ValueError(f"Unsupported phase specification dict: {spec}")


def phase_vector(
    spec: PhaseSpec,
    seq: Sequence[str],
    alphabet: Alphabet,
    horizon: int,
) -> Optional[np.ndarray]:
    """Construct a complex phase vector exp(i * phi(d))."""
    if spec is None:
        return None

    if isinstance(spec, (int, float)):
        mode, params = "theta", (float(spec),)
    elif isinstance(spec, dict):
        mode, params = _parse_phase_dict(spec, seq, alphabet)
    elif isinstance(spec, (list, tuple)):
        mode, params = _parse_phase_tuple(spec, seq, alphabet)
    elif isinstance(spec, str):
        if spec.lower() == "entropy":
            mode, params = "entropy", (0.0, math.pi)
        else:
            try:
                theta = float(spec)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported phase specification string: {spec}") from exc
            mode, params = "theta", (theta,)
    else:
        raise ValueError(f"Unsupported phase specification: {spec}")

    d = np.arange(1, horizon + 1, dtype=np.float64)

    if mode == "theta":
        theta = params[0]
        return np.exp(1j * theta * d)

    if mode == "entropy":
        theta = _entropy_theta(seq, alphabet, params[0], params[1])
        return np.exp(1j * theta * d)

    if mode == "linear":
        start, stop = params
        thetas = np.linspace(start, stop, horizon)
        return np.exp(1j * thetas * d)

    raise ValueError(f"Unexpected phase mode: {mode}")
