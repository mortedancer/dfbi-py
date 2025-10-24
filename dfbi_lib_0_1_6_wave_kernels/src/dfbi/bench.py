
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np

from . import fingerprints as fp
from . import metrics as mt

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_folder_corpus(root: Path) -> Dict[str, List[Path]]:
    data: Dict[str, List[Path]] = {}
    for adir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = sorted([p for p in adir.glob("*.txt") if p.is_file()])
        if files:
            data[adir.name] = files
    return data

def build_profiles(corpus: Dict[str, List[Path]], **fp_kwargs):
    per_file: Dict[str, Tuple[str, np.ndarray]] = {}
    t0 = time.time()
    bytes_total = 0
    for author, files in corpus.items():
        for p in files:
            txt = _read_text(p)
            bytes_total += len(txt.encode("utf-8", errors="ignore"))
            M = fp.fingerprint(txt, **fp_kwargs)
            per_file[str(p)] = (author, M)
    dt = time.time() - t0
    mbps = (bytes_total/1_000_000.0)/dt if dt>0 else float("inf")
    centroids: Dict[str, np.ndarray] = {}
    for author, files in corpus.items():
        Ms = [per_file[str(p)][1] for p in files]
        centroids[author] = np.mean(Ms, axis=0)
    return per_file, centroids, mbps

def _metric_fn(name: str):
    return {
        "l1": mt.dist_l1,
        "l2": mt.dist_l2,
        "chi2": mt.dist_chi2,
        "l2_multi": mt.dist_l2_multi,
        "cosine": mt.dist_cosine,
    }[name]

def nearest_author(M, centroids, metric_name: str):
    f = _metric_fn(metric_name)
    best, best_d = None, 1e300
    for a, C in centroids.items():
        d = f(M, C)
        if d < best_d:
            best, best_d = a, d
    return best

def loocv_accuracy(per_file: Dict[str, Tuple[str, np.ndarray]], metric_name: str) -> float:
    items = list(per_file.items())
    correct = 0
    for path, (author, M) in items:
        by_author = {}
        for path2, (a2, M2) in items:
            if path2 == path: continue
            by_author.setdefault(a2, []).append(M2)
        centroids_tmp = {a: np.mean(Ms, axis=0) for a, Ms in by_author.items()}
        pred = nearest_author(M, centroids_tmp, metric_name)
        if pred == author:
            correct += 1
    return correct/len(items) if items else 0.0

def run_grid(corpus: Dict[str, List[Path]],
             horizons: Iterable[int],
             metrics: Iterable[str],
             masks: Iterable[str] = ("letters",),
             normalize: str = "global",
             decay = ('exp', 0.7),
             kernel: str = "",
             bank: str = "",
             aggregate: str = "",
             phase=None):
    results = []
    for h in horizons:
        for mask in masks:
            per_file, centroids, mbps = build_profiles(
                corpus, horizon=h, mask=mask, normalize=normalize,
                decay=decay, kernel=kernel, bank=bank, aggregate=aggregate,
                phase=phase
            )
            for m in metrics:
                acc = loocv_accuracy(per_file, m)
                results.append({
                    "horizon": h,
                    "mask": mask,
                    "metric": m,
                    "throughput_MBps": mbps,
                    "loocv_acc": acc,
                    "kernel": kernel or "decay",
                    "bank": bank or "none",
                    "aggregate": aggregate or "none",
                    "phase": "none" if phase in (None, "None") else repr(phase),
                })
    return results
