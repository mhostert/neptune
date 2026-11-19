"""
Tests for the Δ(1232)-resonance trident path.

Two layers are tested:

1. ``neptune.resonance``: Lalakulich–Paschos form factors, Breit–Wigner
   spectral function, and the per-nucleus
   ``DeltaResonanceStructureFunctions`` callable.

2. ``neptune.integrands.ResonantTridentIntegrand`` and
   ``TridentProcess.sigma_resonance`` end-to-end.

Like the other TL tests, the Vegas-level checks skip when the .dat
exports are missing.
"""

import numpy as np
import pytest

from neptune import _tl_amplitude as tl
from neptune import const
from neptune.integrands import ResonantTridentIntegrand
from neptune.model import TridentSMModel
from neptune.processes import TridentProcess
from neptune.resonance import (
    DeltaResonanceStructureFunctions,
    GAMMA0_DEFAULT,
    M_PI,
    M_R_DEFAULT,
    SumStructureFunctions,
    breit_wigner,
    C3V,
    C4V,
    delta_running_width,
    pion_cm_momentum,
)


# ---------------------------------------------------------------------------
# Form factors
# ---------------------------------------------------------------------------

def test_C3V_at_Q2_zero():
    """C_3^V(0) = C_3^V(0) constant (default 1.95)."""
    assert C3V(0.0) == pytest.approx(1.95)


def test_C3V_high_Q2_falloff():
    """Modified-dipole falloff: at Q² = 1 GeV² the form factor is ~Q⁻⁶."""
    # (1 + 1/0.84²)² × (1 + 1/(4·0.84²)) ~ 7.9
    val = C3V(1.0)
    assert 0.05 < val < 0.5
    assert val < C3V(0.0)


def test_C4V_LP_relation_at_M_R():
    """C_4^V(Q²=0, W=M_R) = -m_N/M_R · C_3^V(0)."""
    expected = -const.m_AVG / M_R_DEFAULT * C3V(0.0)
    assert C4V(0.0, M_R_DEFAULT) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Pion CM momentum and running width
# ---------------------------------------------------------------------------

def test_pion_cm_momentum_at_pole():
    """p_π at the Δ pole ~0.227 GeV (analytic)."""
    p = pion_cm_momentum(M_R_DEFAULT)
    assert p == pytest.approx(0.227, abs=2e-3)


def test_pion_cm_momentum_at_threshold_is_zero():
    assert pion_cm_momentum(const.m_AVG + M_PI) == pytest.approx(0.0)


def test_pion_cm_momentum_below_threshold_is_zero():
    assert pion_cm_momentum(0.5) == 0.0


def test_running_width_at_pole_equals_Gamma0():
    """Γ_R(M_R) = Γ_0 by construction."""
    assert delta_running_width(M_R_DEFAULT) == pytest.approx(GAMMA0_DEFAULT)


def test_running_width_below_threshold_is_zero():
    assert delta_running_width(0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Breit-Wigner
# ---------------------------------------------------------------------------

def test_BW_below_threshold_is_zero():
    W2_below = (const.m_AVG + M_PI) ** 2 - 0.01
    assert breit_wigner(W2_below) == pytest.approx(0.0)


def test_BW_peaks_near_pole():
    """BW maximum sits within Γ₀ of M_R² (slight downward shift from the
    running width is expected)."""
    W2_grid = np.linspace(1.05, 2.5, 5001)
    rho = breit_wigner(W2_grid)
    W2_at_max = W2_grid[rho.argmax()]
    assert abs(W2_at_max - M_R_DEFAULT**2) < (M_R_DEFAULT * GAMMA0_DEFAULT) * 1.5


def test_BW_normalization_within_window():
    """∫ ρ_BW dW² over a wide window approaches ~1."""
    W2_grid = np.linspace(0.5, 6.0, 20_001)
    rho = breit_wigner(W2_grid)
    integral = np.trapezoid(rho, W2_grid)
    assert 0.85 < integral < 1.05


# ---------------------------------------------------------------------------
# Structure functions
# ---------------------------------------------------------------------------

def test_F1_F2_below_threshold_are_zero():
    """At W² < (m_N + m_π)², both structure functions vanish."""
    sf = DeltaResonanceStructureFunctions(A=1)
    Q2 = 0.001
    # W² = m_N² + Q²(1/xB - 1). Below the Nπ threshold W² < (m_N + m_π)²
    # we need xB > Q² / [m_π(2 m_N + m_π) + Q²] ≈ 3.7e-3 for Q² = 0.001.
    # Pick xB = 0.5 ⇒ W² ≈ m_N² < (m_N + m_π)², well below threshold.
    xB_below = 0.5
    W2 = sf.W2(xB_below, Q2)
    assert W2 < (const.m_AVG + M_PI) ** 2          # double-check kinematics
    assert sf.F1(xB_below, Q2) == pytest.approx(0.0)
    assert sf.F2(xB_below, Q2) == pytest.approx(0.0)


def test_F1_F2_peak_at_resonance_xB():
    """At fixed Q², F1 peaks at xB such that W² ≈ M_R²."""
    sf = DeltaResonanceStructureFunctions(A=1)
    Q2 = 0.5
    xB_grid = np.linspace(0.05, 0.95, 500)
    F1 = sf.F1(xB_grid, Q2)
    xB_peak = xB_grid[F1.argmax()]
    W2_peak = sf.W2(xB_peak, Q2)
    # BW peak shifts slightly below M_R² because of the running width;
    # check we are within ~1.5 widths
    assert abs(W2_peak - M_R_DEFAULT**2) < 3.0 * M_R_DEFAULT * GAMMA0_DEFAULT


def test_F1_F2_scale_linearly_with_A():
    sf1 = DeltaResonanceStructureFunctions(A=1)
    sf40 = DeltaResonanceStructureFunctions(A=40)
    xB, Q2 = 0.45, 0.5
    assert sf40.F1(xB, Q2) == pytest.approx(40.0 * sf1.F1(xB, Q2), abs=0)
    assert sf40.F2(xB, Q2) == pytest.approx(40.0 * sf1.F2(xB, Q2), abs=0)


def test_F1_F2_isospin_factor_default_is_one_third():
    sf = DeltaResonanceStructureFunctions(A=1)
    assert sf.iso == pytest.approx(1.0 / 3.0)


def test_structure_functions_handle_arrays():
    """F1/F2 must be vectorised for use in the Vegas integrand."""
    sf = DeltaResonanceStructureFunctions(A=1)
    xB = np.linspace(0.1, 0.8, 50)
    Q2 = np.linspace(0.05, 1.0, 50)
    F1 = sf.F1(xB, Q2)
    F2 = sf.F2(xB, Q2)
    assert F1.shape == xB.shape
    assert F2.shape == xB.shape
    assert np.all(np.isfinite(F1))
    assert np.all(np.isfinite(F2))


# ---------------------------------------------------------------------------
# Vegas-level integrand and TridentProcess wiring
# ---------------------------------------------------------------------------

def _exports_present():
    return all(tl._resource_path(fn).is_file() for fn in tl._FILES.values())


pytestmark_vegas = pytest.mark.skipif(
    not _exports_present(),
    reason="T/L .dat exports not generated; run mathematica/export_TL_pieces.wl",
)


# ---------------------------------------------------------------------------
# SumStructureFunctions aggregator
# ---------------------------------------------------------------------------

def test_sum_structure_functions_adds_F1_F2():
    """Sum aggregator must return component-wise F1+F2 sums."""
    sf1 = DeltaResonanceStructureFunctions(A=1, M_R=1.232)
    sf2 = DeltaResonanceStructureFunctions(A=1, M_R=1.520)   # mock 2nd peak
    s   = SumStructureFunctions(sf1, sf2)

    xB = np.array([0.3, 0.4, 0.5, 0.6])
    Q2 = 0.5
    np.testing.assert_allclose(s.F1(xB, Q2), sf1.F1(xB, Q2) + sf2.F1(xB, Q2))
    np.testing.assert_allclose(s.F2(xB, Q2), sf1.F2(xB, Q2) + sf2.F2(xB, Q2))


def test_sum_structure_functions_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        SumStructureFunctions()


def test_sum_structure_functions_aggregates_kinematic_limits():
    class _A:
        q2_min, q2_max, x_min = 0.1, 5.0, 0.01
        def F1(self, x, Q2): return 1.0
        def F2(self, x, Q2): return 1.0

    class _B:
        q2_min, q2_max, x_min = 0.5, 3.0, 0.001
        def F1(self, x, Q2): return 2.0
        def F2(self, x, Q2): return 2.0

    s = SumStructureFunctions(_A(), _B())
    # Union of component domains: min(q2_min), max(q2_max), min(x_min)
    assert s.q2_min == 0.1
    assert s.q2_max == 5.0
    assert s.x_min == 0.001
    assert s.F1(0.5, 1.0) == 3.0


@pytestmark_vegas
def test_resonance_integrand_smoke():
    """ResonantTridentIntegrand is finite and non-negative on random points."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    f = ResonantTridentIntegrand(
        m.nu_flavor, m.l1_flavor, m.l2_flavor, m, m.ml1, m.ml2,
        Enu=10.0,
    )
    np.random.seed(2)
    out = f(np.random.rand(64, f.ndim))
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


@pytestmark_vegas
def test_sigma_resonance_finite_and_positive():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, nitn=4, neval=4_000)
    mean, sdev = proc.sigma_resonance()
    assert np.isfinite(mean)
    assert np.isfinite(sdev)
    assert mean > 0


@pytestmark_vegas
def test_sigma_resonance_scales_linearly_with_A():
    """Per-nucleus σ_res should scale linearly with A (impulse approximation)."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc1 = TridentProcess(m, Z=1, A=1, Enu=10.0, nitn=4, neval=4_000)
    proc12 = TridentProcess(m, Z=6, A=12, Enu=10.0, nitn=4, neval=4_000)
    s1, e1 = proc1.sigma_resonance()
    s12, e12 = proc12.sigma_resonance()
    ratio = s12 / s1
    # A=12/A=1 = 12 expected; allow 3σ Monte-Carlo wobble plus 5%.
    rel_err = np.hypot(e1 / s1, e12 / s12)
    assert abs(ratio - 12.0) / 12.0 < 0.05 + 3.0 * rel_err


@pytestmark_vegas
def test_sigma_resonance_smaller_than_coherent_at_high_E():
    """At 10 GeV the Δ contribution is several orders below σ_coherent."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, nitn=4, neval=4_000)
    res, _ = proc.sigma_resonance()
    coh, _ = proc.sigma_coherent()
    assert res > 0 and coh > 0
    assert res < coh   # coherent dominates by 100×–1000× on heavy nuclei


@pytestmark_vegas
def test_sigma_total_includes_resonance_when_requested():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, nitn=4, neval=4_000)
    out = proc.sigma_total(include_resonance=True)
    assert "resonance" in out
    res_mean, _ = out["resonance"]
    dif_mean, _ = out["diffractive"]
    coh_mean, _ = out["coherent"]
    tot_mean, _ = out["total_per_nucleus"]
    assert tot_mean == pytest.approx(dif_mean + coh_mean + res_mean, rel=1e-9, abs=0)


@pytestmark_vegas
def test_resonance_integrand_uses_resonance_form_factors():
    """Constructor default puts a DeltaResonanceStructureFunctions on the
    integrand with A baked in."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0)
    f = proc._make_resonance_integrand()
    assert isinstance(f.structure_functions, DeltaResonanceStructureFunctions)
    assert f.structure_functions.A == 40.0
