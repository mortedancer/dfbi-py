
import numpy as np


def _flatten_complex(a: np.ndarray) -> np.ndarray:
    if not np.iscomplexobj(a):
        return a.reshape(-1)
    stacked = np.stack([a.real, a.imag], axis=-1)
    return stacked.reshape(-1)


def dist_l2(A, B):
    D = A - B
    return float(np.sqrt(np.sum(np.abs(D) ** 2)))


def dist_l1(A, B):
    return float(np.sum(np.abs(A - B)))


def dist_chi2(A, B, eps=1e-12):
    if np.iscomplexobj(A) or np.iscomplexobj(B):
        A = np.abs(A)
        B = np.abs(B)
    if np.any(A < 0) or np.any(B < 0):
        raise ValueError("chi2 requires non-negative inputs")
    denom = A + B + eps
    return float(np.sum(((A - B) ** 2) / denom))


def dist_l2_multi(A, B):
    a = _flatten_complex(np.asarray(A))
    b = _flatten_complex(np.asarray(B))
    d = a - b
    return float(np.sqrt(np.dot(d, d)))


def dist_cosine(A, B, eps=1e-12):
    a = A.reshape(-1)
    b = B.reshape(-1)
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        num = np.vdot(a, b)
        na = np.sqrt(float(np.real(np.vdot(a, a)))) + eps
        nb = np.sqrt(float(np.real(np.vdot(b, b)))) + eps
        cos_sim = float(np.real(num) / (na * nb))
    else:
        num = np.dot(a, b)
        na = np.sqrt(np.dot(a, a)) + eps
        nb = np.sqrt(np.dot(b, b)) + eps
        cos_sim = float(num / (na * nb))
    return float(1.0 - cos_sim)
