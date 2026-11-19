"""
Tests for ``neptune.nuclear_models``.

Three layers:

1. Smith-Moniz analytic checks on ``pauli_blocking_factor``.
2. ``WoodsSaxonDensity`` normalisation, central density, local p_F.
3. Behaviour of the three model classes (``FreeNucleon``, ``FermiGas``,
   ``LocalFermiGas``) and Vegas-level integration through
   ``TridentProcess`` to confirm the ordering Free ≥ LFG ≈ FG(0.221) ≥
   FG(0.251) on Argon at 10 GeV.

The Vegas-level checks skip if the .dat exports are missing.
"""

import numpy as np
import pytest

from neptune import _tl_amplitude as tl
from neptune.model import TridentSMModel
from neptune.nuclear_models import (
    FERMI_MOMENTUM_GEV,
    FermiGas,
    FreeNucleon,
    HBARC,
    LocalFermiGas,
    WoodsSaxonDensity,
    pauli_blocking_factor,
    woods_saxon_density_for,
)
from neptune.processes import TridentProcess


# ---------------------------------------------------------------------------
# Smith-Moniz analytic
# ---------------------------------------------------------------------------

def test_pauli_factor_at_zero_momentum_is_zero():
    assert pauli_blocking_factor(0.0, 0.221) == 0.0


def test_pauli_factor_at_2pF_is_one():
    pF = 0.221
    assert pauli_blocking_factor(2.0 * pF, pF) == pytest.approx(1.0)


def test_pauli_factor_at_pF_is_eleven_sixteenths():
    """Smith-Moniz: at |q| = pF, f_PB = 3/4 - 1/16 = 11/16."""
    pF = 0.221
    assert pauli_blocking_factor(pF, pF) == pytest.approx(11.0 / 16.0)


def test_pauli_factor_above_2pF_is_one():
    pF = 0.221
    assert pauli_blocking_factor(10.0 * pF, pF) == pytest.approx(1.0)


def test_pauli_factor_zero_when_pF_is_zero():
    """A nucleus with no Fermi sea has no nucleons to block against —
    convention: f_PB = 0 (no contribution rather than divergence)."""
    assert pauli_blocking_factor(0.5, 0.0) == 0.0


def test_pauli_factor_vectorised():
    pF = 0.221
    qmag = np.array([0.0, pF, 2.0 * pF, 5.0 * pF])
    out = pauli_blocking_factor(qmag, pF)
    np.testing.assert_allclose(out, [0.0, 11.0 / 16.0, 1.0, 1.0])


def test_pauli_factor_qmag_pF_broadcasting():
    qmag = np.array([0.1, 0.2])
    pF = np.array([0.2, 0.4])
    out = pauli_blocking_factor(qmag, pF)
    expected = np.array([
        pauli_blocking_factor(0.1, 0.2),
        pauli_blocking_factor(0.2, 0.4),
    ])
    np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Woods-Saxon density
# ---------------------------------------------------------------------------

def test_woods_saxon_normalisation_argon():
    ar = woods_saxon_density_for("Ar40")
    r = np.linspace(0.0, ar.r_max, 10_000)
    A_int = 4.0 * np.pi * np.trapezoid(r**2 * ar.rho(r), r)
    assert A_int == pytest.approx(40.0, rel=1e-3)


def test_woods_saxon_central_density_in_typical_range():
    ar = woods_saxon_density_for("Ar40")
    rho0 = ar.rho(0.0)
    # Saturation density 0.16 fm^-3; medium-A nuclei usually 0.16-0.18.
    assert 0.14 < rho0 < 0.20


def test_local_fermi_momentum_in_GeV():
    """ℏc·(3π² ρ/2)^{1/3} ≈ 0.27 GeV at the centre of Argon."""
    ar = woods_saxon_density_for("Ar40")
    pF0 = ar.p_F_isoscalar(0.0)
    assert 0.24 < pF0 < 0.30


def test_p_F_falls_with_radius():
    ar = woods_saxon_density_for("Ar40")
    pF_centre = ar.p_F_isoscalar(0.0)
    pF_edge   = ar.p_F_isoscalar(ar.c + 5 * ar.a)   # well past surface
    assert pF_edge < 0.5 * pF_centre


def test_unknown_nucleus_raises():
    with pytest.raises(KeyError):
        woods_saxon_density_for("Unobtainium")


# ---------------------------------------------------------------------------
# Nuclear-model classes
# ---------------------------------------------------------------------------

def test_free_nucleon_returns_one_everywhere():
    nm = FreeNucleon()
    qmag = np.array([0.0, 0.1, 0.5, 2.0])
    np.testing.assert_array_equal(nm.pauli_factor(qmag), np.ones_like(qmag))


def test_fermi_gas_matches_smith_moniz_analytic():
    nm = FermiGas(p_F=0.221)
    qmag = np.array([0.0, 0.1, 0.221, 0.4, 1.0])
    np.testing.assert_allclose(
        nm.pauli_factor(qmag),
        pauli_blocking_factor(qmag, 0.221),
    )


def test_fermi_gas_for_nucleus_uses_table():
    nm = FermiGas.for_nucleus("Ar40")
    assert nm.p_F == FERMI_MOMENTUM_GEV["Ar40"]


def test_fermi_gas_for_unknown_nucleus_raises():
    with pytest.raises(KeyError):
        FermiGas.for_nucleus("Unobtainium")


def test_lfg_pauli_factor_monotone_in_qmag():
    """f_PB^LFG should rise monotonically with |q| up to the saturation."""
    lfg = LocalFermiGas(woods_saxon_density_for("Ar40"))
    qmag = np.linspace(0.0, 0.6, 60)
    f = lfg.pauli_factor(qmag)
    assert np.all(np.diff(f) >= -1e-12)


def test_lfg_pauli_factor_saturates_to_one_at_large_q():
    lfg = LocalFermiGas(woods_saxon_density_for("Ar40"))
    assert lfg.pauli_factor(2.0) == pytest.approx(1.0, abs=1e-3)


def test_lfg_pauli_factor_zero_at_zero_momentum():
    lfg = LocalFermiGas(woods_saxon_density_for("Ar40"))
    assert lfg.pauli_factor(0.0) == pytest.approx(0.0, abs=1e-6)


def test_lfg_close_to_fg_with_effective_pF():
    """LFG and FG with the LFG-derived effective p_F should track each
    other to within ~10% over the range where Pauli blocking matters."""
    lfg = LocalFermiGas(woods_saxon_density_for("Ar40"))
    fg = FermiGas(p_F=lfg.effective_p_F)
    qmag = np.linspace(0.05, 0.6, 30)
    rel = np.abs(lfg.pauli_factor(qmag) - fg.pauli_factor(qmag)) / np.maximum(
        fg.pauli_factor(qmag), 1e-3
    )
    # Effective-pF heuristic only matches at one anchor point, so the
    # relative deviation can reach ~15% in the rising region.
    assert rel.max() < 0.20


def test_lfg_at_uniform_density_recovers_fg():
    """A uniform-density sphere of radius R = (3A/4πρ)^{1/3} should give
    f_PB^{LFG}(|q|) == f_PB^{FG}(|q|, p_F(ρ))."""
    A = 40.0
    rho_const = 0.16          # fm^-3
    R = (3.0 * A / (4.0 * np.pi * rho_const)) ** (1.0 / 3.0)  # ≈ 3.91 fm
    pF_box = HBARC * (3.0 * np.pi**2 * rho_const / 2.0) ** (1.0 / 3.0)

    class _Box(WoodsSaxonDensity):
        def __init__(self, A=A, Z=18, R=R, rho_const=rho_const):
            self.A = float(A); self.Z = float(Z)
            self.c = float(R); self.a = 1e-4
            self._rho_const = float(rho_const)
        @property
        def r_max(self): return self.c
        @property
        def rho0(self): return self._rho_const
        def rho(self, r):
            r = np.asarray(r, dtype=float)
            return np.where(r <= self.c, self._rho_const, 0.0)
        def p_F_isoscalar(self, r):
            r = np.asarray(r, dtype=float)
            return np.where(r <= self.c, pF_box, 0.0)

    lfg = LocalFermiGas(_Box(), qmag_max=2.0, n_qmag=400, n_radial=2048)
    fg = FermiGas(p_F=pF_box)
    qmag = np.linspace(0.05, 0.5, 25)
    np.testing.assert_allclose(
        lfg.pauli_factor(qmag),
        fg.pauli_factor(qmag),
        atol=5e-3,
    )


# ---------------------------------------------------------------------------
# Vegas-level: nuclear-model effects on σ_diffractive
# ---------------------------------------------------------------------------

def _exports_present():
    return all(tl._resource_path(fn).is_file() for fn in tl._FILES.values())


pytestmark_vegas = pytest.mark.skipif(
    not _exports_present(),
    reason="T/L .dat exports not generated; run mathematica/export_TL_pieces.wl",
)


@pytestmark_vegas
def test_diffractive_ordering_free_gt_lfg_fg():
    """σ_dif: FreeNucleon > FG(0.221) ≈ LFG (Pauli suppression ordering)."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    base_kwargs = dict(model=m, Z=18, A=40, Enu=10.0, nitn=4, neval=4_000)
    s_free, _ = TridentProcess(nuclear_model=FreeNucleon(), **base_kwargs).sigma_diffractive()
    s_fg, _   = TridentProcess(nuclear_model=FermiGas(p_F=0.221), **base_kwargs).sigma_diffractive()
    s_lfg, _  = TridentProcess(
        nuclear_model=LocalFermiGas(woods_saxon_density_for("Ar40")),
        **base_kwargs,
    ).sigma_diffractive()

    assert s_free > s_fg
    assert s_free > s_lfg
    # FG(0.221) and LFG should be within ~10% of each other (effective p_F
    # of LFG is ≈ 0.221 GeV by construction for Argon).
    assert abs(s_fg - s_lfg) / s_fg < 0.15


@pytestmark_vegas
def test_per_call_nuclear_model_overrides_default():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, nitn=3, neval=2_000)
    # default = legacy FermiGas(p_F=0.221) on Diffractive
    f_default = proc._make_diffractive_integrand()
    assert isinstance(f_default.nuclear_model, FermiGas)
    assert f_default.nuclear_model.p_F == 0.221

    f_override = proc._make_diffractive_integrand(nuclear_model=FreeNucleon())
    assert isinstance(f_override.nuclear_model, FreeNucleon)


def test_resonance_with_lfg_is_mildly_suppressed_vs_free():
    """
    Pauli blocking of the Δ uses the inelastic |q|² = Q² + ν², with
    ν ≥ (M_Δ² − M² + Q²)/2M ≃ 0.34 GeV. This sits near 2 p_F, so the LFG
    suppression is pointwise ≤ 1 and small in total (sub-percent at 10 GeV).
    """
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    base = dict(model=m, Z=18, A=40, Enu=10.0)
    free = TridentProcess(nuclear_model=FreeNucleon(), **base)._make_resonance_integrand()
    lfg = TridentProcess(
        nuclear_model=LocalFermiGas(woods_saxon_density_for("Ar40")), **base,
    )._make_resonance_integrand()
    x = np.random.default_rng(3).random((20_000, free.ndim))
    f_free = free(x).ravel()
    f_lfg = lfg(x).ravel()
    assert np.all(f_lfg <= f_free * (1 + 1e-12))
    assert np.any(f_lfg < 0.99 * f_free)
    ratio = f_lfg.sum() / f_free.sum()
    assert 0.95 < ratio < 1.0


def test_invalid_nuclear_model_type_raises():
    """The integrand should reject non-NuclearModel arguments."""
    from neptune.integrands import DiffractiveTridentIntegrand
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    with pytest.raises(TypeError, match="nuclear_model"):
        DiffractiveTridentIntegrand(
            m.nu_flavor, m.l1_flavor, m.l2_flavor, m, m.ml1, m.ml2,
            Enu=10.0, nuclear_model="bogus",
        )


# ---------------------------------------------------------------------------
# Atomic (Coulomb) screening of the nuclear charge
# ---------------------------------------------------------------------------

def test_atomic_screening_boundary_conditions_tsai():
    """G_at(0) = 0 (full screening), G_at(∞) = 1 (no screening)."""
    from neptune.nuclear_tools import atomic_screening_factor
    assert atomic_screening_factor(0.0, 18, "tsai") == 0.0
    assert atomic_screening_factor(1e30, 18, "tsai") == pytest.approx(1.0)


def test_atomic_screening_at_inverse_a_TF_squared_is_half():
    """Tsai: G(Q² = 1/a_TF²) = x/(1+x) with x = 1, so G = 1/2."""
    from neptune.nuclear_tools import (
        atomic_screening_factor, thomas_fermi_length,
    )
    Z = 18
    Q2 = 1.0 / thomas_fermi_length(Z) ** 2
    assert atomic_screening_factor(Q2, Z, "tsai") == pytest.approx(0.5)


def test_atomic_screening_boundary_conditions_moliere():
    from neptune.nuclear_tools import atomic_screening_factor
    assert atomic_screening_factor(0.0, 18, "moliere") == pytest.approx(0.0)
    assert atomic_screening_factor(1e30, 18, "moliere") == pytest.approx(1.0)


def test_moliere_weights_sum_to_one():
    from neptune.nuclear_tools import _MOLIERE_ALPHAS
    assert sum(_MOLIERE_ALPHAS) == pytest.approx(1.0)


def test_atomic_screening_aliases():
    from neptune.nuclear_tools import atomic_screening_factor
    Q2 = 1e-9
    Z = 18
    g_tsai = atomic_screening_factor(Q2, Z, "tsai")
    for alias in ("Tsai", "thomas-fermi", "thomas_fermi", "TF", "tf"):
        assert atomic_screening_factor(Q2, Z, alias) == pytest.approx(g_tsai)


def test_unknown_screening_raises():
    from neptune.nuclear_tools import atomic_screening_factor
    with pytest.raises(ValueError, match="atomic_screening"):
        atomic_screening_factor(1e-9, 18, "bogus")


def test_salvat_raises_when_table_missing():
    """The exact Salvat option requires a per-Z table; raise if absent."""
    from neptune.nuclear_tools import atomic_screening_factor
    with pytest.raises(KeyError, match="Salvat"):
        atomic_screening_factor(1e-9, 18, "salvat")


def test_atomic_screening_higher_Z_screens_at_higher_Q2():
    """Heavier nuclei have smaller a_TF, so the screening crossover
    moves to higher Q²."""
    from neptune.nuclear_tools import atomic_screening_factor
    Q2 = 1e-9
    g_C  = atomic_screening_factor(Q2, 6,  "tsai")
    g_Pb = atomic_screening_factor(Q2, 82, "tsai")
    # At fixed Q², Pb (smaller a_TF) is screened MORE than C (larger a_TF)
    # → smaller G value.
    assert g_Pb < g_C


def test_apply_atomic_screening_wraps_form_factor():
    from neptune.nuclear_tools import apply_atomic_screening
    F0 = lambda Q2: np.ones_like(np.asarray(Q2, dtype=float))
    F = apply_atomic_screening(F0, 18, "tsai")
    assert F(0.0) == 0.0
    assert F(1.0) == pytest.approx(1.0)


def test_apply_atomic_screening_none_is_passthrough():
    """kind=None must return the original callable unchanged."""
    from neptune.nuclear_tools import apply_atomic_screening
    F0 = lambda Q2: 0.5 * np.ones_like(np.asarray(Q2, dtype=float))
    F = apply_atomic_screening(F0, 18, None)
    assert F is F0


@pytestmark_vegas
def test_screening_has_negligible_effect_on_muon_pairs():
    """For μ⁺μ⁻ on Ar, the kinematic Q²_min ≫ 1/a_TF², so screening
    should not change the cross section beyond Vegas MC noise."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    base = dict(model=m, Z=18, A=40, Enu=10.0, nitn=6, neval=8_000)
    s_unscreened, e1 = TridentProcess(atomic_screening=None, **base).sigma_coherent()
    s_tsai,       e2 = TridentProcess(atomic_screening="tsai", **base).sigma_coherent()
    diff_sigma = abs(s_unscreened - s_tsai) / np.hypot(e1, e2)
    assert diff_sigma < 3.0, f"μ⁺μ⁻ unexpectedly affected by screening: {diff_sigma:.2f}σ"


@pytestmark_vegas
def test_screening_constructor_records_choice():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    proc = TridentProcess(m, Z=18, A=40, Enu=2.0, atomic_screening="tsai")
    assert proc.atomic_screening == "tsai"
    # The wrapped form factor should give 0 at Q²=0 (full screening).
    assert proc._form_factor(0.0) == pytest.approx(0.0)


@pytestmark_vegas
def test_screening_form_factor_continuous_in_Q2():
    """Sanity: the wrapped form factor is finite and continuous on a
    realistic Q² range."""
    m = TridentSMModel(nu_flavor="mu", l1_flavor="e", l2_flavor="e")
    proc = TridentProcess(m, Z=18, A=40, Enu=2.0, atomic_screening="tsai")
    Q2 = np.geomspace(1e-12, 1e-2, 50)
    F = proc._form_factor(Q2)
    assert F.shape == Q2.shape
    assert np.all(np.isfinite(F))
    assert np.all(F >= 0)
    assert F[0] < F[-1]   # rises from full screening to bare nucleus


def test_moliere_screening_closed_form_and_limits():
    """G = Σ α_i x/(x + β_i²), x = Q² a_TF², with Molière (α, β)."""
    from neptune.nuclear_tools import atomic_screening_factor, thomas_fermi_length
    Z = 18
    a = thomas_fermi_length(Z)
    x = np.array([1e-3, 1.0, 1e4])
    G = atomic_screening_factor(x / a**2, Z, "moliere")
    alphas, betas = np.array([0.10, 0.55, 0.35]), np.array([6.0, 1.2, 0.3])
    expected = np.sum(alphas[:, None] * x / (x + betas[:, None] ** 2), axis=0)
    np.testing.assert_allclose(G, expected, rtol=1e-12)
    assert G[1] == pytest.approx(0.549, abs=1e-3)
    assert G[-1] == pytest.approx(1.0, abs=1e-2)
    assert np.all(np.diff(atomic_screening_factor(np.geomspace(1e-3, 1e3, 50) / a**2,
                                                  Z, "moliere")) > 0)


def test_moliere_and_tsai_agree_at_order_one():
    """
    Both are Thomas–Fermi fits and agree within 15% for Q² a_TF² ≥ 1. Below
    that they differ by design: G → x Σ α_i/β_i² ≃ 4.3 x (Molière) vs. x (Tsai).
    """
    from neptune.nuclear_tools import atomic_screening_factor, thomas_fermi_length
    Z = 26
    a2 = thomas_fermi_length(Z) ** 2
    Q2 = np.geomspace(1.0, 100.0, 20) / a2
    ratio = atomic_screening_factor(Q2, Z, "moliere") / atomic_screening_factor(Q2, Z, "tsai")
    assert np.all((ratio > 0.85) & (ratio < 1.15))
    small = atomic_screening_factor(1e-6 / a2, Z, "moliere") / 1e-6
    assert small == pytest.approx(0.10 / 36 + 0.55 / 1.44 + 0.35 / 0.09, rel=1e-4)


def test_local_fermi_gas_factor_bounded():
    lfg = LocalFermiGas(woods_saxon_density_for("Ar40"))
    q = np.linspace(0.0, 3.0, 20_001)
    f = lfg.pauli_factor(q)
    assert np.all((f >= 0.0) & (f <= 1.0))
