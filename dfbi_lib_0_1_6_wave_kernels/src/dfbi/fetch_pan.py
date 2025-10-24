from __future__ import annotations
from pathlib import Path
import tempfile, zipfile, tarfile, urllib.request

def _safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _extract_any(archive_path: Path, dest: Path):
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z: z.extractall(dest); return
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as t: t.extractall(dest); return
    if name.endswith(".tar.bz2") or name.endswith(".tbz2"):
        with tarfile.open(archive_path, "r:bz2") as t: t.extractall(dest); return
    raise ValueError(f"Unsupported archive: {archive_path}")

def fetch_kaggle(dataset_ref: str, download_dir: Path) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    _safe_mkdir(download_dir)
    api.dataset_download_files(dataset_ref, path=str(download_dir), quiet=False, unzip=False)
    archives = list(download_dir.glob("*.zip")) + list(download_dir.glob("*.tar.gz")) + list(download_dir.glob("*.tgz")) + list(download_dir.glob("*.tar.bz2"))
    if not archives:
        raise RuntimeError("No archives found after Kaggle download.")
    extracted_root = download_dir / "extracted"; _safe_mkdir(extracted_root)
    for a in archives: _extract_any(a, extracted_root)
    return extracted_root

def fetch_url(url: str, download_dir: Path) -> Path:
    _safe_mkdir(download_dir)
    local = download_dir / Path(url).name
    with urllib.request.urlopen(url) as resp, open(local, "wb") as f: f.write(resp.read())
    out = download_dir / "extracted"; _safe_mkdir(out); _extract_any(local, out); return out

def fetch_and_prepare_pan(out_corpus: Path,
                          kaggle_ref: str | None = None,
                          url: str | None = None,
                          temp_root: Path | None = None) -> Path:
    if not kaggle_ref and not url:
        raise ValueError("Provide either kaggle_ref or url.")
    temp_root = temp_root or Path(tempfile.mkdtemp(prefix="dfbi_pan_"))
    if kaggle_ref: extracted = fetch_kaggle(kaggle_ref, temp_root / "download")
    else: extracted = fetch_url(url, temp_root / "download")
    from .prepare_pan import prepare_pan
    out_corpus = Path(out_corpus); _safe_mkdir(out_corpus)
    written = prepare_pan(str(extracted), str(out_corpus))
    if written <= 0:
        raise RuntimeError("No documents were prepared; check dataset format.")
    return out_corpus
