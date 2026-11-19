"""
Tests for neutrino-trident production on atomic electrons.

Two layers:

1. Loader / dispatch checks for ``regime='electron'`` in the unified
   ``differential_cross_section_tl``.
2. Vegas-level integration via ``ElectronTridentIntegrand`` and
   ``TridentProcess.sigma_electron``.

Like the other Vegas-level test modules, this one skips when the .dat
exports are missing.
"""

import numpy as np
import pytest

from neptune import _tl_amplitude as tl
from neptune import const
from neptune.integrands import ElectronTridentIntegrand
from neptune.model import TridentSMModel
from neptune.phase_space import trident_threshold
from neptune.processes import TridentProcess


# ---------------------------------------------------------------------------
# Loader / dispatch
# ---------------------------------------------------------------------------

def test_h_ele_files_loadable():
    """Both flux loaders compile to a callable expression."""
    assert np.isfinite(tl.h_T_ele(0.001, 0.05, 1.0, const.m_e))
    assert np.isfinite(tl.h_L_ele(0.001, 0.05, 1.0, const.m_e))


def test_h_T_ele_finite_and_real():
    out = tl.h_T_ele(np.array([1e-4, 1e-3, 1e-2]), 0.05, 1.0, const.m_e)
    assert np.all(np.isfinite(out))


def test_electron_regime_requires_me():
    """differential_cross_section_tl should reject regime='electron' with no me."""
    with pytest.raises(ValueError, match="electron regime requires me"):
        tl.differential_cross_section_tl(
            0.001, 0.05, 0.01, 0.01, 0.02, 0.02,
            Enu=1.0, Mn=const.m_e, ml1=0.1057, ml2=0.1057,
            V2=1.0, A2=1.0, VA=1.0,
            regime="electron",
        )


def test_electron_regime_dispatch_returns_finite():
    out = tl.differential_cross_section_tl(
        0.001, 0.05, 0.01, 0.01, 0.02, 0.02,
        Enu=1.0, Mn=const.m_e, ml1=0.1057, ml2=0.1057,
        V2=1.0, A2=1.0, VA=1.0,
        regime="electron", me=const.m_e,
    )
    assert np.isfinite(out)


# ---------------------------------------------------------------------------
# Threshold sanity
# ---------------------------------------------------------------------------

def test_eee_threshold_low():
    """ν + e → ν + e + e⁺ + e⁻ has threshold ≈ 4 m_e (a few MeV)."""
    th = trident_threshold(const.m_e, const.m_e, const.m_e)
    assert th < 0.01    # GeV — well below 10 MeV


def test_emumu_threshold_high():
    """ν + e → ν + e + μ⁺ + μ⁻ has threshold ≈ m_μ²/m_e ~ 22 GeV."""
    th = trident_threshold(0.1057, 0.1057, const.m_e)
    assert 10 < th < 100  # GeV — heavy-pair-on-electron threshold is high


# ---------------------------------------------------------------------------
# Vegas-level
# ---------------------------------------------------------------------------

def _exports_present():
    return all(tl._resource_path(fn).is_file() for fn in tl._FILES.values())


pytestmark_vegas = pytest.mark.skipif(
    not _exports_present(),
    reason="T/L .dat exports not generated; evaluate mathematica/BSM_trident.nb",
)


@pytestmark_vegas
def test_electron_integrand_finite_and_nonneg():
    """ElectronTridentIntegrand is finite and ≥ 0 on random points."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    f = ElectronTridentIntegrand(
        m.nu_flavor, m.l1_flavor, m.l2_flavor, m, m.ml1, m.ml2, Enu=1.0,
    )
    np.random.seed(3)
    out = f(np.random.rand(64, f.ndim))
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


@pytestmark_vegas
def test_sigma_electron_finite_and_positive_eee():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    proc = TridentProcess(m, Z=18, A=40, Enu=1.0, nitn=4, neval=4_000)
    mean, sdev = proc.sigma_electron()
    assert np.isfinite(mean) and np.isfinite(sdev)
    assert mean > 0


@pytestmark_vegas
def test_sigma_electron_scales_linearly_with_Z():
    """Per-atom σ should be Z × per-electron σ exactly (incoherent sum)."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    proc = TridentProcess(m, Z=18, A=40, Enu=2.0, nitn=4, neval=4_000)
    s_per_e, _ = proc.sigma_electron_per_e()
    s_atom,  _ = proc.sigma_electron()
    # Two independent Vegas runs — compare to within MC noise.
    assert s_atom == pytest.approx(18.0 * s_per_e, rel=0.1, abs=0)


@pytestmark_vegas
def test_sigma_electron_below_threshold_is_zero_for_emumu():
    """E_ν below the eμμ threshold should give exactly 0."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    th = trident_threshold(0.1057, 0.1057, const.m_e)
    proc = TridentProcess(m, Z=18, A=40, Enu=th * 0.5,
                           nitn=2, neval=1_000)
    mean, sdev = proc.sigma_electron()
    assert mean == 0.0
    # Vegas reports a numerically-zero sdev (~1e-100); allow for floating-point dust.
    assert sdev < 1e-50


@pytestmark_vegas
def test_sigma_total_includes_electron_when_requested():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    proc = TridentProcess(m, Z=18, A=40, Enu=2.0, nitn=4, neval=2_000)
    out = proc.sigma_total(include_electron=True)
    assert "electron" in out
    coh, _ = out["coherent"]
    dif, _ = out["diffractive"]
    ele, _ = out["electron"]
    tot, _ = out["total_per_nucleus"]
    assert tot == pytest.approx(coh + dif + ele, rel=1e-9, abs=0)


@pytestmark_vegas
def test_eee_electron_below_coherent_on_argon():
    """On Ar-40 at GeV-scale Eν, σ_electron ≪ σ_coherent for e+e-."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    proc = TridentProcess(m, Z=18, A=40, Enu=2.0, nitn=4, neval=4_000)
    coh, _ = proc.sigma_coherent()
    ele, _ = proc.sigma_electron()
    assert 0 < ele < 0.05 * coh    # electron piece below 5% of coherent
