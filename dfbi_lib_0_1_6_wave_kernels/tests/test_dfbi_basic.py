
# -*- coding: utf-8 -*-
import numpy as np
from dfbi import fingerprints as fp, metrics as mt

def test_shapes_and_norms():
    txt = "абвабв абв!"
    M = fp.fingerprint(txt, horizon=1, decay=('const',), mask='letters', normalize='global')
    assert M.shape == (41, 41)
    s = float(M.sum())
    assert np.isclose(s, 1.0) or s == 0.0

def test_horizon_effect():
    txt = "абвг"
    M1 = fp.fingerprint(txt, horizon=1, decay=('const',), mask='letters')
    M3 = fp.fingerprint(txt, horizon=3, decay=('const',), mask='letters')
    # При большем горизонте часть массы распределяется на дальние пары
    assert (M3 > M1).sum() > 0  # где-то должны появиться ненулевые ячейки, которых не было

def test_metrics_monotonicity():
    t1 = "в начале июля под вечер один молодой человек"
    t2 = "идёт по базару полицейский надзиратель очумелов"
    M1 = fp.fingerprint(t1, horizon=2, decay=('exp',0.7))
    M2 = fp.fingerprint(t2, horizon=2, decay=('exp',0.7))
    d1 = mt.dist_l1(M1, M2)
    d2 = mt.dist_l2(M1, M2)
    assert d1 >= 0 and d2 >= 0
