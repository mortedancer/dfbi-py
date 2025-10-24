
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class Alphabet:
    symbols: List[str]
    index: Dict[str, int]

def build_alphabet(symbols: List[str]) -> Alphabet:
    return Alphabet(symbols=list(symbols), index={c:i for i,c in enumerate(symbols)})

RUS_LETTERS = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
EXTRA = [' ', ',', '.', '-', ':', '"', '0', '?']
RU41 = build_alphabet(RUS_LETTERS + EXTRA)

EN_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
EN34 = build_alphabet(EN_LETTERS + EXTRA)
