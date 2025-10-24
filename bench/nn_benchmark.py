
# -*- coding: utf-8 -*-

"""
DFBI mini-benchmark & smoke test on a folder corpus.

Folder layout:
  corpus/
    author_a/
      a1.txt
      a2.txt
      ...
    author_b/
      b1.txt
      ...

We compute DFBI fingerprints for all files, then run 1-NN author attribution
in leave-one-out manner. We report micro-accuracy and throughput (MB/s).
"""

import argparse, os, time, glob, math, statistics
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from dfbi import fingerprints as fp
from dfbi import metrics as mt

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_corpus(root: Path) -> Dict[str, List[Path]]:
    data = {}
    for author_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted([p for p in author_dir.glob("*.txt") if p.is_file()])
        if files:
            data[author_dir.name] = files
    return data

def build_profiles(corpus: Dict[str, List[Path]], **fp_kwargs):
    # returns: dict[path]->(author, matrix), and per-author centroid matrix
    per_file = {}
    t0 = time.time()
    bytes_total = 0
    for author, files in corpus.items():
        for p in files:
            txt = read_text(p)
            bytes_total += len(txt.encode("utf-8", errors="ignore"))
            M = fp.fingerprint(txt, **fp_kwargs)
            per_file[str(p)] = (author, M)
    dt = time.time() - t0
    mbps = (bytes_total/1_000_000.0)/dt if dt>0 else float("inf")
    # centroids
    per_author = {}
    for author, files in corpus.items():
        Ms = [per_file[str(p)][1] for p in files]
        per_author[author] = np.mean(Ms, axis=0)
    return per_file, per_author, mbps

def nearest_author(M, centroids, metric):
    best, best_d = None, 1e300
    for a, C in centroids.items():
        d = metric(M, C)
        if d < best_d:
            best, best_d = a, d
    return best

def run_loocv(per_file, metric):
    # leave-one-out over author's files using centroids of others
    items = list(per_file.items())
    correct = 0
    for path, (author, M) in items:
        # recompute centroid for each author leaving this file out if belongs
        # to that author
        per_author_tmp = {}
        # gather Ms per author
        by_author = {}
        for path2,(a2,M2) in items:
            if path2 == path: continue
            by_author.setdefault(a2, []).append(M2)
        for a, Ms in by_author.items():
            per_author_tmp[a] = np.mean(Ms, axis=0)
        pred = nearest_author(M, per_author_tmp, metric)
        if pred == author:
            correct += 1
    acc = correct/len(items) if items else 0.0
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_root", help="Path to folder with author subfolders")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--decay", default="('exp',0.7)")
    ap.add_argument("--mask", choices=["none","letters","punct"], default="letters")
    ap.add_argument("--normalize", choices=["global","row"], default="global")
    ap.add_argument("--metric", choices=["l1","l2","chi2"], default="l1")
    args = ap.parse_args()

    root = Path(args.corpus_root)
    corpus = load_corpus(root)
    if not corpus:
        raise SystemExit(f"No authors with .txt files found under: {root}")

    per_file, per_author, mbps = build_profiles(
        corpus,
        horizon=args.horizon,
        decay=eval(args.decay),
        mask=args.mask,
        normalize=args.normalize,
    )

    metric = {"l1":mt.dist_l1,"l2":mt.dist_l2,"chi2":mt.dist_chi2}[args.metric]
    acc = run_loocv(per_file, metric)

    print(f"# files: {len(per_file)} | authors: {len(per_author)}")
    print(f"Throughput: {mbps:.2f} MB/s")
    print(f"1-NN LOOCV accuracy (author): {acc*100:.2f}%")

if __name__ == "__main__":
    main()
