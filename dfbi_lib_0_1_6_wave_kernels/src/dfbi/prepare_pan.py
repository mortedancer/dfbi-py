# -*- coding: utf-8 -*-
import csv, re
from pathlib import Path
from typing import Dict, List, Optional

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_doc(out_dir: Path, author: str, doc_id: str, text: str):
    safe_author = re.sub(r"[^0-9A-Za-zА-Яа-я_\-]+", "_", author.strip())
    _ensure_dir(out_dir / safe_author)
    fname = f"{doc_id or 'doc'}".strip() or "doc"
    (out_dir / safe_author / f"{fname}.txt").write_text(text, encoding="utf-8")

def _guess_sep(line: str) -> str:
    if "\t" in line: return "\t"
    if "," in line: return ","
    return None

def prepare_from_csv(csv_path: Path, out_root: Path,
                     author_col: str = "author", text_col: str = "text", id_col: Optional[str] = None) -> int:
    rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            author = row.get(author_col) or row.get("label") or row.get("topic") or row.get("gender")
            text = row.get(text_col) or row.get("content") or row.get("body") or row.get("text")
            doc_id = (row.get(id_col) if id_col else None) or row.get("id") or f"doc_{i:06d}"
            if not author or not text:
                continue
            _write_doc(out_root, author, str(doc_id), text)
            rows += 1
    return rows

def prepare_from_truth_dir(in_root: Path, out_root: Path) -> int:
    truth_file = in_root / "truth.txt"
    if not truth_file.exists():
        raise FileNotFoundError(f"No truth.txt in {in_root}")
    mapping: Dict[str, str] = {}
    with truth_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if "\t" in line: parts = [p.strip() for p in line.split("\t")]
            elif "," in line: parts = [p.strip() for p in line.split(",")]
            else: parts = line.split()
            if len(parts) < 2: continue
            doc_id, author = parts[0], parts[1]
            mapping[doc_id] = author
    moved = 0
    for doc_id, author in mapping.items():
        src_txt = None
        if (in_root / f"{doc_id}.txt").exists():
            src_txt = in_root / f"{doc_id}.txt"
        else:
            p = in_root / doc_id
            if p.is_dir():
                cand = p / "unknown.txt"
                if cand.exists():
                    src_txt = cand
        if not src_txt:
            continue
        text = src_txt.read_text(encoding="utf-8", errors="ignore")
        _write_doc(out_root, author, doc_id, text)
        moved += 1
    return moved

def is_author_folder_layout(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    for d in p.iterdir():
        if d.is_dir() and list(d.glob("*.txt")):
            return True
    return False

def mirror_author_folder_layout(in_root: Path, out_root: Path) -> int:
    count = 0
    for author_dir in sorted([d for d in in_root.iterdir() if d.is_dir()]):
        for f in author_dir.glob("*.txt"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            _write_doc(out_root, author_dir.name, f.stem, text)
            count += 1
    return count

def prepare_pan(input_path: str, output_corpus_root: str) -> int:
    in_path = Path(input_path)
    out_root = Path(output_corpus_root)
    _ensure_dir(out_root)

    if in_path.is_file() and in_path.suffix.lower() == ".csv":
        return prepare_from_csv(in_path, out_root)

    if in_path.is_dir():
        if is_author_folder_layout(in_path):
            return mirror_author_folder_layout(in_path, out_root)
        truth = in_path / "truth.txt"
        if truth.exists():
            return prepare_from_truth_dir(in_path, out_root)

    raise ValueError("Unsupported input format: provide CSV, folder with truth.txt, or author/*/*.txt")
