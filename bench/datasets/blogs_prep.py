
from __future__ import annotations
from pathlib import Path
import pandas as pd
import re, os

def _safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _slug(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z_\-]+', '_', str(s))
    s = re.sub(r'__+', '_', s).strip('_')
    return s or 'cls'

def download_blogs_via_kagglehub():
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub") from e
    path = kagglehub.dataset_download("rtatman/blog-authorship-corpus")
    csv_path = Path(path) / "blogtext.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"blogtext.csv not found under {path}")
    return csv_path

def prepare_blogs(out_dir: str,
                  group_by: str = "topic",
                  top_k: int = 10,
                  per_class: int = 500,
                  min_chars: int = 300,
                  random_state: int = 42,
                  csv_path: str | None = None) -> Path:
    out = Path(out_dir); _safe_mkdir(out)
    if csv_path:
        csv = Path(csv_path)
    else:
        csv = download_blogs_via_kagglehub()
    df = pd.read_csv(csv)
    if group_by not in df.columns:
        raise ValueError(f"group_by='{group_by}' not in columns: {df.columns.tolist()}")
    if 'text' not in df.columns:
        raise ValueError("column 'text' not found in CSV")
    # basic filter
    df = df[df['text'].astype(str).str.len() >= min_chars].copy()
    # top-k classes by count
    top = df[group_by].value_counts().head(top_k).index.tolist()
    df = df[df[group_by].isin(top)].copy()
    # per-class sample (balanced)
    parts=[]
    for cls in top:
        d = df[df[group_by]==cls]
        if len(d) > per_class:
            d = d.sample(n=per_class, random_state=random_state)
        parts.append(d)
    dfb = pd.concat(parts, axis=0).reset_index(drop=True)
    # write author-folder layout
    for cls, g in dfb.groupby(group_by):
        cls_dir = out / _slug(cls)
        _safe_mkdir(cls_dir)
        for i, row in enumerate(g.itertuples(), start=1):
            (cls_dir / f"{_slug(cls)}_{i:06d}.txt").write_text(str(row.text), encoding="utf-8")
    return out
