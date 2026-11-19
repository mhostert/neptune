"""Trident couplings for ν and ν̄, and CP relations of the integrated pieces.

At one-photon exchange with an unpolarized hadron, CP gives
σ(ν̄_α → ℓ1⁻ ℓ2⁺) = σ(ν_α → ℓ2⁻ ℓ1⁺). In terms of the coupling structures,
the V² and A² integrals are symmetric under m1 ↔ m2 and the VA integral is
antisymmetric. These tests also guard the phase-space map, which once cut
the ℓ2 propagator peak (q·p_ℓ2 < 0) and broke both relations.
"""

import numpy as np
import pytest

import neptune as nep
from neptune import const


GV, GA = const.gV_lep, const.gA_lep


@pytest.mark.parametrize(
    "nu, l1, l2, nubar, V, A",
    [
        ("mu", "mu", "mu", False, 1 + GV, 1 + GA),
        ("mu", "mu", "mu", True, 1 + GV, 1 + GA),
        ("mu", "mu", "e", False, 1.0, 1.0),
        ("mu", "e", "mu", True, 1.0, 1.0),
        ("mu", "e", "e", False, GV, GA),
        ("mu", "e", "e", True, GV, GA),
        ("e", "tau", "e", True, 1.0, 1.0),
    ],
)
def test_vertex_couplings(nu, l1, l2, nubar, V, A):
    assert const.get_Vijk(nu, l1, l2, nubar) == pytest.approx(V)
    assert const.get_Aijk(nu, l1, l2, nubar) == pytest.approx(A)


@pytest.mark.parametrize(
    "nu, l1, l2, nubar",
    [("mu", "e", "mu", False), ("mu", "mu", "e", True), ("mu", "e", "tau", False)],
)
def test_flavor_violating_channels_raise(nu, l1, l2, nubar):
    with pytest.raises(ValueError):
        nep.TridentSMModel(nu_flavor=nu, l1_flavor=l1, l2_flavor=l2, is_nubar=nubar)


def test_nubar_flips_only_va():
    nu = nep.TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    nb = nep.TridentSMModel(
        nu_flavor="mu", l1_flavor="mu", l2_flavor="mu", is_nubar=True
    )
    c_nu, c_nb = nu.get_coupling_terms(), nb.get_coupling_terms()
    assert c_nb["V2"] == c_nu["V2"] and c_nb["A2"] == c_nu["A2"]
    assert c_nb["VA"] == -c_nu["VA"]


def test_nubar_flip_acts_on_zprime_axial_piece():
    kw = dict(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu", mzprime=0.05,
              gV_prime=1e-2, gA_prime=5e-3)
    nu = nep.TridentBSMModel(**kw)
    nb = nep.TridentBSMModel(is_nubar=True, **kw)
    m6 = np.array([1e-4, 1e-2])
    assert np.allclose(nb.get_coupling_terms(m6)["VA"], -nu.get_coupling_terms(m6)["VA"])


def test_zprime_absent_in_flavor_changing_channel():
    sm = nep.TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="e")
    bsm = nep.TridentBSMModel(
        nu_flavor="mu", l1_flavor="mu", l2_flavor="e", mzprime=0.01, gV_prime=0.1
    )
    c_sm, c_bsm = sm.get_coupling_terms(), bsm.get_coupling_terms(np.array([1e-3]))
    for k in ("V2", "A2", "VA"):
        assert np.allclose(c_bsm[k], c_sm[k])


def _piece(l1, l2, which, regime, neval=20_000):
    # Couplings are overridden below; nu_flavor = l1 keeps the channel allowed.
    model = nep.TridentSMModel(nu_flavor=l1, l1_flavor=l1, l2_flavor=l2)
    coup = {"V2": 0.0, "A2": 0.0, "VA": 0.0}
    coup[which] = 1.0
    model.get_coupling_terms = lambda m6=None: dict(coup)
    proc = nep.TridentProcess(model, Z=18, A=40, Enu=10.0)
    if regime == "diffractive":
        return proc.sigma_diffractive_nucleon("proton", nitn=10, neval=neval)[0]
    return proc.sigma_coherent(nitn=10, neval=neval)[0]


@pytest.mark.parametrize("regime", ["coherent", "diffractive"])
def test_vv_symmetric_under_lepton_mass_swap(regime):
    a = _piece("mu", "e", "V2", regime)
    b = _piece("e", "mu", "V2", regime)
    assert a == pytest.approx(b, rel=0.03, abs=0)


@pytest.mark.parametrize("regime", ["coherent", "diffractive"])
def test_va_antisymmetric_under_lepton_mass_swap(regime):
    a = _piece("mu", "e", "VA", regime)
    b = _piece("e", "mu", "VA", regime)
    assert a == pytest.approx(-b, rel=0.03, abs=0)


@pytest.mark.parametrize("regime", ["coherent", "diffractive"])
def test_va_vanishes_for_equal_masses(regime):
    va = _piece("mu", "mu", "VA", regime)
    vv = _piece("mu", "mu", "V2", regime)
    assert abs(va) < 0.01 * vv


def test_nubar_cross_section_equals_cp_conjugate():
    nu = nep.TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="e")
    nb = nep.TridentSMModel(
        nu_flavor="mu", l1_flavor="e", l2_flavor="mu", is_nubar=True
    )
    s_nu = nep.TridentProcess(nu, Z=18, A=40, Enu=10.0).sigma_diffractive_nucleon(
        "proton", nitn=10, neval=20_000
    )[0]
    s_nb = nep.TridentProcess(nb, Z=18, A=40, Enu=10.0).sigma_diffractive_nucleon(
        "proton", nitn=10, neval=20_000
    )[0]
    assert s_nb == pytest.approx(s_nu, rel=0.02, abs=0)
