"""
Integration tests for the trident integrands and ``TridentProcess``.

These tests exercise the full Vegas integration through the T/L
integrands. They depend on the .dat exports produced by
``mathematica/export_TL_pieces.wl``; if any are missing the whole module
is skipped.
"""

import numpy as np
import pytest

from neptune import _tl_amplitude as tl
from neptune.integrands import (
    CoherentTridentIntegrand,
    DiffractiveTridentIntegrand,
)
from neptune.model import TridentSMModel
from neptune.processes import TridentProcess


def _exports_present():
    return all(tl._resource_path(fn).is_file() for fn in tl._FILES.values())


pytestmark = pytest.mark.skipif(
    not _exports_present(),
    reason="T/L .dat exports not generated; run mathematica/export_TL_pieces.wl",
)


def _proc(mode="full"):
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    return TridentProcess(m, Z=18, A=40, Enu=10.0,
                          mode=mode, nitn=4, neval=4_000)


# ---------------------------------------------------------------------------
# Vegas-level: each mode produces a finite, positive cross section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["full", "improved-epa", "epa"])
def test_coherent_modes_finite_and_positive(mode):
    mean, sdev = _proc(mode).sigma_coherent()
    assert np.isfinite(mean) and np.isfinite(sdev)
    assert mean > 0


@pytest.mark.parametrize("mode", ["full", "improved-epa", "epa"])
def test_diffractive_modes_finite_and_positive(mode):
    mean, sdev = _proc(mode).sigma_diffractive()
    assert np.isfinite(mean) and np.isfinite(sdev)
    assert mean > 0


# ---------------------------------------------------------------------------
# Mode dispatch and validation
# ---------------------------------------------------------------------------

def test_default_mode_is_full():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0)
    assert proc.mode == "full"
    assert proc._make_coherent_integrand().mode == "full"
    assert proc._make_diffractive_integrand().mode == "full"


def test_per_call_mode_override():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, mode="full")
    assert proc._make_coherent_integrand(mode="epa").mode == "epa"
    assert proc._make_diffractive_integrand(mode="improved-epa").mode == "improved-epa"


def test_integrand_classes_returned():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0)
    assert isinstance(proc._make_coherent_integrand(), CoherentTridentIntegrand)
    assert isinstance(proc._make_diffractive_integrand(), DiffractiveTridentIntegrand)


def test_invalid_mode_rejected():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    with pytest.raises(ValueError, match="mode"):
        TridentProcess(m, Z=18, A=40, Enu=10.0, mode="nonsense")


def test_mode_aliases_normalised():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    assert TridentProcess(m, Z=18, A=40, Enu=10.0, mode="iepa").mode == "improved-epa"
    assert TridentProcess(m, Z=18, A=40, Enu=10.0, mode="improved_epa").mode == "improved-epa"


# ---------------------------------------------------------------------------
# Sanity: σ_total combines diffractive (× Z) and coherent
# ---------------------------------------------------------------------------

def test_sigma_total_combines_components():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, nitn=4, neval=4_000)
    out = proc.sigma_total()
    assert set(out) == {"diffractive", "coherent", "total_per_nucleus"}
    dif_mean, _ = out["diffractive"]
    coh_mean, _ = out["coherent"]
    tot_mean, _ = out["total_per_nucleus"]
    assert tot_mean == pytest.approx(proc.Z * dif_mean + coh_mean, rel=1e-9)
