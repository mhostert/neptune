"""Neutrino-electron elastic scattering: kinematics, closed forms, generator."""

import numpy as np
import pytest

import neptune as nep
from neptune import nu_electron as ne


@pytest.mark.parametrize("Enu", [1e-3, 1e-2, 1.0, 10.0])
@pytest.mark.parametrize("flavor", ["e", "mu"])
def test_quadrature_matches_analytic_sm(Enu, flavor):
    """The BSM quadrature path with g' → 0 must reproduce the SM closed form,
    including at E_nu ~ m_e where T_max = 2E²/(m_e + 2E) differs from E - m_e."""
    sm = ne.total_xsec(Enu, flavor)
    quad = ne.total_xsec(Enu, flavor, mzprime=0.1, gprime=1e-12, n_grid=4000)
    assert quad == pytest.approx(sm, rel=1e-5, abs=0)


def test_dsigma_vanishes_outside_kinematic_range():
    Enu = 2e-3
    Tmax = ne.Te_max(Enu)
    assert ne.dsigma_dTe(Enu, 0.999 * Tmax, "e") > 0
    assert ne.dsigma_dTe(Enu, 1.001 * Tmax, "e") == 0
    assert ne.dsigma_dTe(Enu, -1e-6, "e") == 0


def test_sm_cross_section_scale_and_flavor_ordering():
    """σ(ν_e e) ≃ 9.3e-42 cm² (E/GeV) at high energy, and σ(ν_e) > σ(ν_μ)
    thanks to the CC contribution."""
    s_e = ne.total_xsec(1.0, "e")
    s_mu = ne.total_xsec(1.0, "mu")
    assert 8e-42 < s_e < 1.1e-41
    assert s_e > s_mu
    # Linear in E far above m_e.
    assert ne.total_xsec(10.0, "e") / s_e == pytest.approx(10.0, rel=1e-3)


def test_generator_momentum_conservation_and_weights():
    model = nep.NuElectronSMModel(nu_flavor="mu")
    df = nep.NuElectronGenerator(model, Enu=1.0, n_events=2000, seed=1).generate()
    P_in = df["P_projectile"].to_numpy() + df["P_target"].to_numpy()
    P_out = df["P_decay_ell_minus"].to_numpy() + df["P_decay_N_daughter"].to_numpy()
    assert np.allclose(P_in, P_out, atol=1e-12)

    p_nu = df["P_decay_N_daughter"].to_numpy()
    m2 = p_nu[:, 0] ** 2 - np.sum(p_nu[:, 1:] ** 2, axis=1)
    assert np.max(np.abs(m2)) < 1e-8

    # Mean weight estimates the total cross section at fixed Enu.
    w = df["w_event_rate"].to_numpy()
    assert w.mean() == pytest.approx(model.total_xsec(1.0), rel=0.05, abs=0)
    assert np.all(df["Te"] <= ne.Te_max(1.0) + 1e-15)
    assert np.all(df["Te"] >= 0)


def test_bsm_light_zprime_enhances_low_recoil():
    sm = nep.NuElectronSMModel(nu_flavor="mu")
    bsm = nep.NuElectronBSMModel(nu_flavor="mu", mzprime=1e-3, gprime=1e-4)
    # The Z' propagator 1/(M² + 2 m_e T) makes the BSM shift largest at low T.
    r_low = bsm.dsigma_dTe(1.0, 1e-3) / sm.dsigma_dTe(1.0, 1e-3)
    r_high = bsm.dsigma_dTe(1.0, 0.5) / sm.dsigma_dTe(1.0, 0.5)
    assert abs(r_low - 1) > 10 * abs(r_high - 1)
