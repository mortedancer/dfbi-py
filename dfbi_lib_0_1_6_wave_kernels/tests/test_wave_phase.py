import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OLD_SRC = pathlib.Path(__file__).resolve().parents[2] / "dfbi_lib_0_1_0" / "src"

# Ensure we import the wave-kernel build rather than legacy editions.
sys.path = [p for p in sys.path if pathlib.Path(p).resolve() != OLD_SRC.resolve()]
sys.path.insert(0, str(SRC))

for mod in list(sys.modules):
    if mod.startswith("dfbi"):
        sys.modules.pop(mod)

from dfbi.fingerprints import fingerprint  # noqa: E402
from dfbi.alphabet import RU41  # noqa: E402
from dfbi import metrics as mt  # noqa: E402


def test_morlet_entropy_signature_is_complex_and_normalized():
    text = "абракадабра"
    M = fingerprint(
        text,
        alphabet=RU41,
        horizon=4,
        decay=("morlet", {"omega": 2.5, "sigma": 1.0, "scale": 1.0}),
        phase=("entropy", 0.0, np.pi),
    )
    assert M.shape == (len(RU41.symbols),) * 2
    assert np.iscomplexobj(M)
    assert np.isclose(np.abs(M).sum(), 1.0, atol=1e-9)

    idx_a = RU41.index["а"]
    idx_b = RU41.index["б"]
    assert np.isclose(M[idx_a, idx_b].real, 0.001649168074340559, atol=1e-12)
    assert np.isclose(M[idx_a, idx_b].imag, 0.005993740761930198, atol=1e-12)
    assert np.isclose(M[idx_a, idx_a].real, 0.00012014843299498369, atol=1e-12)
    assert np.isclose(M[idx_a, idx_a].imag, 0.0, atol=1e-12)


def test_row_normalization_handles_complex_values():
    M = fingerprint(
        "абракадабра",
        alphabet=RU41,
        horizon=4,
        decay=("morlet", {"omega": 2.5, "sigma": 1.0}),
        phase=("theta", 0.5),
        normalize="row",
    )
    row_sums = np.abs(M).sum(axis=1)
    nz = row_sums[row_sums > 1e-12]
    assert np.allclose(nz, 1.0, atol=1e-9)


def test_metrics_accept_complex_inputs():
    kwargs = dict(
        alphabet=RU41,
        horizon=5,
        decay=("morlet", {"omega": 3.1, "sigma": 1.2}),
        phase=("theta", 0.75),
    )
    M1 = fingerprint("волны текста", **kwargs)
    M2 = fingerprint("квантовая поэзия", **kwargs)

    assert np.iscomplexobj(M1) and np.iscomplexobj(M2)

    d_l1 = mt.dist_l1(M1, M2)
    d_l2 = mt.dist_l2(M1, M2)
    d_l2_multi = mt.dist_l2_multi(M1, M2)
    d_cos = mt.dist_cosine(M1, M2)
    d_chi2 = mt.dist_chi2(M1, M2)

    assert np.isclose(d_l1, 0.7804668388420433, atol=1e-12)
    assert np.isclose(d_l2, 0.044031915463228405, atol=1e-12)
    assert np.isclose(d_l2_multi, d_l2, atol=1e-12)
    assert np.isclose(d_cos, 0.6669224098517963, atol=1e-12)
    assert np.isclose(d_chi2, 0.5771673145134163, atol=1e-12)
