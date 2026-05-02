import numpy as np

import neptune as nep
from neptune import const
from neptune.dis import (
    dis_partonic_x_min,
    photon_pdf_inelastic,
    sigma_dis,
    sigma_dis_x_grid,
    sigma_dis_lab_threshold,
)
from neptune.phase_space import (
    dis_trident_phase_space_density,
    map_dis_unit_to_physical,
)


def _dot(a, b):
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def test_dis_kinematics_conserve_momentum_and_on_shell():
    rng = np.random.default_rng(12345)
    x = rng.random((128, 5))
    ps = map_dis_unit_to_physical(
        x,
        s_hat=3.0,
        ml1=const.m_mu,
        ml2=const.m_mu,
    )

    residual = ps["p1"] + ps["q"] - ps["p2"] - ps["km"] - ps["kp"]
    assert np.max(np.abs(residual)) < 1e-12
    assert np.max(np.abs(_dot(ps["p1"], ps["p1"]))) < 1e-12
    assert np.max(np.abs(_dot(ps["q"], ps["q"]))) < 1e-12
    assert np.max(np.abs(_dot(ps["p2"], ps["p2"]))) < 1e-12
    assert np.max(np.abs(_dot(ps["km"], ps["km"]) - const.m_mu**2)) < 1e-12
    assert np.max(np.abs(_dot(ps["kp"], ps["kp"]) - const.m_mu**2)) < 1e-12
    assert np.all(ps["valid"])


def test_dis_phase_space_massless_volume():
    s_hat = 5.0
    Mll2 = np.linspace(0.0, s_hat, 20_001)
    density = dis_trident_phase_space_density(s_hat, Mll2, 0.0, 0.0)
    volume = np.trapezoid(density * (4.0 * np.pi) ** 2, Mll2)
    expected = s_hat / (256.0 * np.pi**3)
    assert np.isclose(volume, expected, rtol=5e-5)


def test_inelastic_photon_pdf_uses_single_collinear_splitting_x():
    x = np.array([0.1, 0.3, 0.7])
    mu2 = 4.0
    log_mu = np.log(mu2 / const.m_proton**2)
    expected = (
        const.alphaQED
        / (2.0 * np.pi)
        * (4.0 / 9.0)
        * (1.0 + (1.0 - x) ** 2)
        / x
        * log_mu
    )
    assert np.allclose(photon_pdf_inelastic(x, mu2), expected)


def test_sigma_dis_enforces_inelastic_lab_threshold_before_partonic_fold():
    model = nep.TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    threshold = sigma_dis_lab_threshold(model.ml1, model.ml2)
    assert threshold > 0.1
    assert sigma_dis(model, Enu=0.1, photon_pdf=photon_pdf_inelastic) == 0.0

    inclusive_threshold = sigma_dis_lab_threshold(
        model.ml1,
        model.ml2,
        hadronic_mass_min=const.m_proton,
    )
    assert inclusive_threshold < threshold
    assert inclusive_threshold > 0.1


def test_sigma_dis_empty_grid_below_lab_threshold():
    model = nep.TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    grid = sigma_dis_x_grid(model, Enu=0.1, photon_pdf=photon_pdf_inelastic)
    assert grid["sigma"] == 0.0
    assert grid["x"].size == 0
    assert dis_partonic_x_min(0.1, model.ml1, model.ml2) < 1.0
