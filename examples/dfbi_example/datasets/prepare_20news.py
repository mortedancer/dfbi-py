
import argparse, re
from pathlib import Path

def prepare(out_dir: Path, subset: int = 5, per_class: int = 300, remove_headers=True):
    try:
        from sklearn.datasets import fetch_20newsgroups
    except Exception as e:
        raise SystemExit("scikit-learn not available or fetch failed. Install scikit-learn and ensure internet access.") from e

    data = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes') if remove_headers else ())
    # choose first N categories alphabetically for determinism
    cats = sorted(set(data.target_names))[:subset]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {c:0 for c in cats}
    for text, label in zip(data.data, data.target):
        name = data.target_names[label]
        if name not in cats: continue
        if counts[name] >= per_class: continue
        counts[name]+=1
        safe = re.sub(r'[^0-9A-Za-z_-]+','_', name)
        d = out_dir / safe; d.mkdir(parents=True, exist_ok=True)
        (d / f"doc_{counts[name]:05d}.txt").write_text(text, encoding='utf-8')
    print("Prepared:", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--keep-headers", action="store_true")
    args = ap.parse_args()
    prepare(Path(args.out), subset=args.subset, per_class=args.per_class, remove_headers=not args.keep_headers)
