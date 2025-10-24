"""
DFBI Evaluation with Global-Centroid Scoring
===========================================

This script extends the DFBI parameter sweep by adopting alternative
classification strategies.  In addition to the standard nearest
centroid rule, candidates are ranked using a composite score that
takes into account the distance between an author’s centroid and the
global centroid of all training data.  The idea is to prefer authors
whose style is not only close to the test sample but also aligns well
with the general language patterns of the corpus.  Two scoring
functions are considered:

* **Rule A (centroid + global)**:
    score(a) = dist(sample, centroid[a]) + dist(centroid[a], global)

* **Rule B (centroid – global)**:
    score(a) = dist(sample, centroid[a]) – dist(centroid[a], global)

For each rule we record the fraction of cases where the true author is
ranked 1st (top‑1), within the top‑3, and within the top‑5.

The script reuses the optimisation from ``dfbi_analysis_optimized.py``:
counts tensors are computed once per horizon, then different decay
functions, masks and phases are applied in O(horizon) time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

# Add DFBI library to path
ROOT = Path(__file__).resolve().parents[0]
candidate_dirs = [
    ROOT / 'dfbi_lib_0_1_6_wave_kernels' / 'src',
    ROOT / 'project' / 'dfbi_lib_0_1_6_wave_kernels' / 'src',
]
for cand in candidate_dirs:
    if cand.exists():
        sys.path.insert(0, str(cand))
        break

from dfbi.alphabet import EN34
from dfbi.decay import get_decay
from dfbi.phase import phase_vector
from dfbi.utils import normalize_char
from dfbi.metrics import dist_l2


def load_corpus(corpus_dir: Path, max_chars: int | None = None) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for author_dir in sorted(corpus_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        for txt_file in sorted(author_dir.glob('*.txt')):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if max_chars:
                    text = text[:max_chars]
                texts.append(text)
                labels.append(author)
    return texts, labels


def compute_counts_tensor(seq_idx: list[int], horizon: int, alphabet_size: int) -> np.ndarray:
    h = horizon
    tensor = np.zeros((h, alphabet_size, alphabet_size), dtype=np.float64)
    n = len(seq_idx)
    for i in range(n):
        a = seq_idx[i]
        for d in range(1, h + 1):
            j = i + d
            if j >= n:
                break
            b = seq_idx[j]
            tensor[d - 1, a, b] += 1.0
    return tensor


def apply_decay_and_mask(counts: np.ndarray, decay_spec: tuple, mask: str | None,
                         phase_spec: tuple | None, seq: list[str]) -> np.ndarray:
    h = counts.shape[0]
    decay_fn = get_decay(decay_spec)
    weights = np.array([decay_fn(d) for d in range(1, h + 1)], dtype=np.complex128)
    if phase_spec is not None:
        pvec = phase_vector(phase_spec, seq, EN34, horizon=h)
        weights = weights * pvec
    weighted = np.tensordot(weights, counts, axes=([0], [0]))
    # apply mask
    if mask == 'letters':
        letters = set('abcdefghijklmnopqrstuvwxyz')
        M = np.ones((len(EN34.symbols), len(EN34.symbols)), dtype=np.float64)
        for i, a in enumerate(EN34.symbols):
            for j, b in enumerate(EN34.symbols):
                if not (a in letters and b in letters):
                    M[i, j] = 0.0
        M = M.astype(weighted.dtype)
        weighted = weighted * M
    # row normalise
    row_sums = weighted.real.sum(axis=1)
    row_sums[row_sums == 0.0] = 1.0
    weighted = weighted / row_sums[:, None]
    return weighted


def precompute_counts(texts: list[str], horizon: int) -> tuple[list[np.ndarray], list[list[str]]]:
    counts_list: list[np.ndarray] = []
    seq_chars_list: list[list[str]] = []
    for text in texts:
        seq_chars = [normalize_char(c, EN34) for c in text]
        seq_idx = [EN34.index[ch] for ch in seq_chars if ch in EN34.index]
        seq_chars_list.append(seq_chars)
        counts = compute_counts_tensor(seq_idx, horizon, len(EN34.symbols))
        counts_list.append(counts)
    return counts_list, seq_chars_list


def classify_with_global(fps: list[np.ndarray], labels: list[str], rule: str) -> tuple[float, float, float]:
    """
    Perform leave‑one‑out classification with alternative scoring rules.

    Returns:
        top1_acc, top3_acc, top5_acc
    """
    n = len(fps)
    authors = sorted(set(labels))
    idxs_by_author = {a: [] for a in authors}
    for i, lab in enumerate(labels):
        idxs_by_author[lab].append(i)
    hits1 = hits3 = hits5 = 0
    # Precompute global centroid for each leave‑one‑out split? We need global per test; we compute inside loop.
    for i in range(n):
        test_fp = fps[i]
        test_label = labels[i]
        # Build training set (exclude i)
        train_indices = [j for j in range(n) if j != i]
        # Compute global centroid of all training samples
        global_centroid = np.mean([fps[j] for j in train_indices], axis=0)
        # Compute centroids for each author excluding i
        centroids = {}
        for author in authors:
            sel = [j for j in idxs_by_author[author] if j != i]
            if sel:
                centroids[author] = np.mean([fps[j] for j in sel], axis=0)
        # Score each author according to rule
        scores = []
        for author, centroid in centroids.items():
            d_sample = dist_l2(test_fp, centroid)
            d_global = dist_l2(centroid, global_centroid)
            if rule == 'A':
                score = d_sample + d_global
            elif rule == 'B':
                score = d_sample - d_global
            else:
                raise ValueError(f"Unknown rule: {rule}")
            scores.append((score, author))
        scores.sort(key=lambda x: x[0])
        # Determine ranks
        predicted_order = [a for _, a in scores]
        if predicted_order[0] == test_label:
            hits1 += 1
        if test_label in predicted_order[:3]:
            hits3 += 1
        if test_label in predicted_order[:5]:
            hits5 += 1
    return hits1 / n, hits3 / n, hits5 / n


def main():
    parser = argparse.ArgumentParser(description="DFBI evaluation with global centroid scoring")
    parser.add_argument('--corpus-dir', type=str,
                        default=str(ROOT / 'project' / 'examples' / 'dfbi_ccat50' / 'corpus_authors'),
                        help='Path to CCAT50 corpus')
    parser.add_argument('--max-chars', type=int, default=3000,
                        help='Truncate texts to this many characters')
    parser.add_argument('--output', type=str, default='analysis_results_global.csv',
                        help='CSV file to write results')
    parser.add_argument('--plots-dir', type=str, default='analysis_plots_global',
                        help='Directory to store plots')
    args = parser.parse_args()

    texts, labels = load_corpus(Path(args.corpus_dir), args.max_chars if args.max_chars>0 else None)
    horizons = [4, 6, 8, 10, 12]
    decays = [
        ('exp', 0.5),
        ('gauss', {'mu':3.0,'sigma':2.5}),
        ('morlet', {'omega':2.5,'sigma':1.0}),
    ]
    masks = [None, 'letters']
    # Only consider no phase and theta=0.5 for morlet
    phase_opts = [None, ('theta',0.5)]
    results = []
    horizon_cache = {}
    for horizon in horizons:
        print(f"Precomputing counts for horizon {horizon}")
        counts_list, seq_chars_list = precompute_counts(texts, horizon)
        horizon_cache[horizon] = (counts_list, seq_chars_list)
        for decay in decays:
            decay_name = decay[0]
            decay_param = decay[1]
            # Determine phase list
            phases = phase_opts if decay_name == 'morlet' else [None]
            for phase in phases:
                for mask in masks:
                    # Build fingerprints
                    fps=[]
                    fp_time_start = time.perf_counter()
                    for counts, seq_chars in zip(horizon_cache[horizon][0], horizon_cache[horizon][1]):
                        fp = apply_decay_and_mask(counts, decay, mask, phase, seq_chars)
                        fps.append(fp)
                    fp_time = time.perf_counter() - fp_time_start
                    # Evaluate classification for rule A and B
                    cls_start = time.perf_counter()
                    acc1_A, acc3_A, acc5_A = classify_with_global(fps, labels, 'A')
                    cls_time_A = time.perf_counter() - cls_start
                    # Evaluate rule B
                    cls_start_B = time.perf_counter()
                    acc1_B, acc3_B, acc5_B = classify_with_global(fps, labels, 'B')
                    cls_time_B = time.perf_counter() - cls_start_B
                    results.append({
                        'horizon': horizon,
                        'decay': decay_name,
                        'decay_param': json.dumps(decay_param) if not isinstance(decay_param,(int,float)) else decay_param,
                        'mask': mask or 'none',
                        'phase': json.dumps(phase) if phase is not None else 'none',
                        'rule': 'A',
                        'top1': acc1_A,
                        'top3': acc3_A,
                        'top5': acc5_A,
                        'fp_time': fp_time,
                        'cls_time': cls_time_A,
                    })
                    results.append({
                        'horizon': horizon,
                        'decay': decay_name,
                        'decay_param': json.dumps(decay_param) if not isinstance(decay_param,(int,float)) else decay_param,
                        'mask': mask or 'none',
                        'phase': json.dumps(phase) if phase is not None else 'none',
                        'rule': 'B',
                        'top1': acc1_B,
                        'top3': acc3_B,
                        'top5': acc5_B,
                        'fp_time': fp_time,
                        'cls_time': cls_time_B,
                    })
                    print(f"h={horizon} decay={decay_name} mask={mask or 'none'} phase={phase or 'none'}"
                          f" rule=A -> top1={acc1_A:.3f} top3={acc3_A:.3f} top5={acc5_A:.3f}" )
                    print(f"                              rule=B -> top1={acc1_B:.3f} top3={acc3_B:.3f} top5={acc5_B:.3f}")
    df=pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    # Plotting summarised performance (optional)
    plots_dir=Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style='whitegrid')
        # Aggregate by decay and rule (top1)
        g=df.groupby(['decay','rule'])['top1'].max().reset_index()
        plt.figure()
        sns.barplot(data=g, x='decay', y='top1', hue='rule')
        plt.ylim(0,1)
        plt.title('Максимальная точность top‑1 по функциям и правилам')
        plt.ylabel('Точность')
        plt.savefig(plots_dir/'accuracy_bar_rule.png', dpi=120)
        plt.close()
    except ImportError:
        pass

if __name__ == '__main__':
    main()