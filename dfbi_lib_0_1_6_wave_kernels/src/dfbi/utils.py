
import unicodedata
from .alphabet import Alphabet, RU41

def normalize_char(c: str, alpha: Alphabet = RU41) -> str:
    c = c.lower()
    if c in alpha.index: return c
    nf = unicodedata.normalize('NFKD', c)
    base = ''.join(ch for ch in nf if not unicodedata.combining(ch))
    if base and base[0] in alpha.index:
        return base[0]
    if c.isdigit(): return '0'
    if c in (';',): return ':'
    if c in ("«","»","'","`"): return '"'
    if c in ("–","—"): return '-'
    if c in ("!"): return '?'
    return '?'
