"""
neptune.nuclear_tools
=====================
Nuclear target utilities for neutrino trident calculations.

Re-exports DarkNews NuclearTarget and related helpers, and adds:
  - FF_WS, FF_WS_Q2 : Woods-Saxon coherent form factor (BSMscan C++ definition)
  - H1_p, H2_p, H1_n, H2_n : nucleon elastic dipole form-factor combinations
  - atomic_screening_factor, apply_atomic_screening : atomic charge screening

All momenta / mass arguments are in GeV.
"""

import numpy as np

# ─── Re-export everything from DarkNews ───────────────────────────────────────
from DarkNews.nuclear_tools import (  # noqa: F401
    NuclearTarget,
    assign_form_factors,
    elements_dic,
    fourier_bessel_dic,
    nuclear_F1_fourier_bessel_EM,
    nuclear_F1_Fsym_EM,
    nucleon_F1_EM,
    nucleon_F2_EM,
    nucleon_F1_NC,
    nucleon_F2_NC,
    nucleon_F3_NC,
)

from neptune import const


def get_nuclear_target(name: str) -> NuclearTarget:
    """
    Convenience wrapper: return a DarkNews NuclearTarget by name.

    Parameters
    ----------
    name : str
        Element name in Nuclear Data Table format, e.g. 'Ar40', 'C12', 'Fe56'.
        Also accepts 'proton' or 'neutron' for free nucleons.

    Returns
    -------
    NuclearTarget
    """
    # map friendly aliases
    _alias = {
        "p": "H1",
        "proton": "H1",
        "n": "neutron",
        "neutron": "neutron",
    }
    return NuclearTarget(_alias.get(name, name))


def find_nucleus_name(Z, A):
    """
    Reverse-lookup the DarkNews element name (e.g. 'Ar40') from (Z, A).

    Returns ``None`` if no match is found.
    """
    Z = int(Z)
    A = int(A)
    for name, entry in elements_dic.items():
        if int(entry.get("Z", -1)) == Z and int(entry.get("A", -1)) == A:
            return name
    return None


def get_form_factor(spec, Z, A, nuclear_target=None):
    """
    Resolve a coherent nuclear form-factor specification to a callable
    ``f(Q²) -> F(Q²)`` with ``Q²`` in GeV² and ``F`` dimensionless
    (normalised to 1 at Q² = 0).

    Parameters
    ----------
    spec : str or callable
        - ``'woods-saxon'`` / ``'ws'`` (default): analytic, A-only Woods-Saxon
          form factor (`FF_WS`). Same definition as the C++ reference.
        - ``'darknews'`` / ``'dn'``: DarkNews's per-nucleus form factor — the
          Fourier-Bessel parametrisation with measured coefficients where
          tabulated, falling back to the symmetrised Fermi profile otherwise.
        - any callable ``f(Q²)`` returning the dimensionless form factor
          (used as-is; useful for user-defined parametrisations).
    Z, A : int
        Atomic and mass numbers (used by the ``'woods-saxon'`` path and for
        the (Z, A) → name reverse lookup in the ``'darknews'`` path).
    nuclear_target : str or NuclearTarget, optional
        Used only for ``spec='darknews'``. If a string, it is interpreted as
        the Nuclear Data Table name (e.g. ``'Ar40'``); if a ``NuclearTarget``
        instance, it is used directly. If omitted, neptune attempts to find
        a matching nucleus from ``(Z, A)``.

    Returns
    -------
    callable
        ``f(Q²) -> F(Q²)``, vectorised over numpy arrays.
    """
    if callable(spec):
        return spec
    if spec in ("ws", "woods-saxon", "woods_saxon"):
        A_local = A
        return lambda Q2: FF_WS_Q2(Q2, A_local)
    if spec in ("dn", "darknews"):
        if nuclear_target is None:
            name = find_nucleus_name(Z, A)
            if name is None:
                raise ValueError(
                    f"Could not find a DarkNews nucleus matching (Z={Z}, A={A}). "
                    "Pass nuclear_target='ElementName' (e.g. 'Ar40') explicitly."
                )
            target = NuclearTarget(name)
        elif isinstance(nuclear_target, str):
            target = NuclearTarget(nuclear_target)
        else:
            target = nuclear_target
        # NuclearTarget.__init__ already calls assign_form_factors(self),
        # which sets target.F1_EM to the appropriate parametrisation.
        return target.F1_EM
    raise ValueError(
        f"Unknown form_factor spec {spec!r}. "
        "Use 'woods-saxon', 'darknews', or pass a callable f(Q²)."
    )


# ─── Coherent (nuclear) form factor: Woods-Saxon ──────────────────────────────


def FF_WS(q, A):
    """
    Woods-Saxon nuclear electromagnetic form factor at ``|q| = sqrt(Q²)``.

    Parameters
    ----------
    q : array-like
        |q| = sqrt(Q²) in GeV.
    A : float
        Atomic mass number.
    """
    q = np.asarray(q, dtype=float)
    return FF_WS_Q2(q * q, A)


def FF_WS_Q2(Q2: np.ndarray, A: float) -> np.ndarray:
    """
    Vectorized Woods-Saxon form factor.

    Parameters
    ----------
    Q2 : array-like
        Squared momentum transfer Q² ≥ 0 (GeV²).  Can be array.
    A : float
        Atomic mass number.

    Returns
    -------
    np.ndarray
        Form factor values, same shape as Q2.
    """
    Q2 = np.asarray(Q2, dtype=float)
    q = np.sqrt(np.maximum(Q2, 0.0))  # safe sqrt

    r0 = 1.07 / 0.197 * A ** (1.0 / 3.0)
    a = 0.57 / 0.197

    # avoid division by zero at q=0: F → 1 as q → 0
    safe_q = np.where(q > 0, q, 1.0)
    num = (
        3.0
        * np.pi
        * safe_q
        * a
        * (
            np.pi * safe_q * a / np.tanh(np.pi * safe_q * a) * np.sin(safe_q * r0)
            - safe_q * r0 * np.cos(safe_q * r0)
        )
    )
    den = safe_q**3 * np.sinh(np.pi * safe_q * a) * (r0**2 + np.pi**2 * a**2) * r0
    result = np.where(q > 0, num / den, 1.0)
    return result


# ─── Nucleon form factors (dipole approximation) ─────────────────────────────

# Constants from cross_sections.h / cross_sections.cxx
_MAG_MOM_P = 2.793  # proton magnetic moment (in nuclear magnetons)
_MAG_MOM_N = -1.913  # neutron magnetic moment
_MV2 = 0.71  # vector form factor dipole mass² (GeV²)


def _G_dip(q: np.ndarray) -> np.ndarray:
    """Dipole form factor G(q) with q = |Q| = sqrt(Q²)."""
    Q2 = q * q
    return (1.0 + Q2 / _MV2) ** (-2.0)


def H1_p(q: np.ndarray) -> np.ndarray:
    """
    Proton Pauli form factor H1 (used for diffractive scattering).

    Parameters
    ----------
    q : array-like
        |q| = sqrt(Q²) in GeV.

    Returns
    -------
    np.ndarray
    """
    q = np.asarray(q, dtype=float)
    Q2 = q * q
    tau = -Q2 / (4.0 * const.m_AVG**2)  # tau = Q²/4M² (spacelike convention)
    Gd = _G_dip(q)
    return (Gd**2 - tau * (_MAG_MOM_P * Gd) ** 2) / (1.0 - tau)


def H2_p(q: np.ndarray) -> np.ndarray:
    """
    Proton Sachs form factor H2 (used for diffractive scattering).

    Parameters
    ----------
    q : array-like
        |q| = sqrt(Q²) in GeV.

    Returns
    -------
    np.ndarray
    """
    q = np.asarray(q, dtype=float)
    Gd = _G_dip(q)
    return (_MAG_MOM_P * Gd) ** 2


def H1_n(q: np.ndarray) -> np.ndarray:
    """Neutron H1 form factor."""
    q = np.asarray(q, dtype=float)
    Q2 = q * q
    tau = -Q2 / (4.0 * const.m_AVG**2)
    Gd = _G_dip(q)
    return -tau / (1.0 - tau) * (_MAG_MOM_N * Gd) ** 2


def H2_n(q: np.ndarray) -> np.ndarray:
    """Neutron H2 form factor."""
    q = np.asarray(q, dtype=float)
    Gd = _G_dip(q)
    return (_MAG_MOM_N * Gd) ** 2


# ─── Atomic (Coulomb) screening of the nuclear charge ───────────────────────
#
# Bound atomic electrons screen the nuclear charge at small Q^2. The effect
# is a multiplicative modulator G_at(Q^2) on the nuclear form factor:
#
#       F_nuclear(Q^2)  →  F_nuclear(Q^2) · G_at(Q^2),
#
# with G_at(0) = 0 (photon sees a neutral atom) and G_at(∞) = 1 (photon
# resolves the bare nucleus). The crossover scale is set by the Thomas–Fermi
# screening length  a_TF = 0.8853 · a_0 · Z^{-1/3}.
#
# Two parametrisations are exposed:
#
#   - 'tsai'    : single-pole Thomas–Fermi  G(Q²) = x/(1+x), x = Q² a_TF².
#                 Tsai, Rev. Mod. Phys. 46, 815 (1974).
#
#   - 'moliere' : 3-exponential Moliere fit to the Thomas–Fermi potential,
#                 a universal-coefficient stand-in for the Z-dependent
#                 Salvat–Martínez–Mayol–Parellada (Phys. Rev. A 36, 467
#                 (1987)) parametrisation. Single length scale ``a_TF``.
#
# The exact Salvat parameterisation requires the per-Z (α_i, λ_i) table from
# the original 1987 paper; if you have that table, drop it into
# ``_SALVAT_PARAMS`` and select ``'salvat'`` — the evaluator below will pick
# it up automatically.

# Bohr radius in GeV^-1: a_0 [m] / (ℏc [GeV·m])
BOHR_RADIUS_GEV_INV = 5.29177e-11 / (0.197327e-15)


def thomas_fermi_length(Z) -> np.ndarray:
    """
    Thomas–Fermi screening length :math:`a_{TF} = 0.8853 \\, a_0 \\, Z^{-1/3}`
    in GeV⁻¹.
    """
    Z = np.asarray(Z, dtype=float)
    return 0.8853 * BOHR_RADIUS_GEV_INV * np.power(Z, -1.0 / 3.0)


# Moliere coefficients (universal, not Z-dependent). The Moliere fit to the
# Thomas-Fermi screening function is φ(r) = Σ_i α_i exp(-β_i r / a_TF), whose
# Fourier transform gives
#     G(Q²) = 1 - F_atom(Q²) = Σ_i α_i x / (x + β_i²),   x = Q² a_TF².
# Larger β_i is a shorter-range (inner-shell) component that keeps screening
# up to larger Q².
_MOLIERE_ALPHAS = np.array([0.10, 0.55, 0.35])
_MOLIERE_BETAS  = np.array([6.0,  1.2,  0.3])


# Drop a per-Z table here to enable the exact Salvat parametrisation.
# Each entry: list of (alpha_i, lambda_i) tuples in *atomic* units, i.e.
# ``a = a_0 / Z^{1/3}`` (note: not a_TF). G(Q²) = sum_i α_i · Q²/(Q² + (λ_i/a)²).
_SALVAT_PARAMS: dict = {}


def _tsai_screening(Q2: np.ndarray, Z: float) -> np.ndarray:
    """Thomas–Fermi single-pole screening modulator (Tsai)."""
    a = thomas_fermi_length(Z)
    x = np.asarray(Q2, dtype=float) * a * a
    return x / (1.0 + x)


def _moliere_screening(Q2: np.ndarray, Z: float) -> np.ndarray:
    """3-exponential Moliere screening modulator (universal coefficients)."""
    a = thomas_fermi_length(Z)
    Q2 = np.asarray(Q2, dtype=float)
    x = Q2 * a * a
    # Σ_i α_i x / (x + β_i²); → Σ_i α_i = 1 as x → ∞.
    out = 0.0
    for alpha_i, beta_i in zip(_MOLIERE_ALPHAS, _MOLIERE_BETAS):
        out = out + alpha_i * x / (x + beta_i * beta_i)
    return out


def _salvat_screening(Q2: np.ndarray, Z: float) -> np.ndarray:
    """
    Salvat–Martínez–Mayol–Parellada three-exponential screening (1987).

    Requires per-Z (α_i, λ_i) coefficients in ``_SALVAT_PARAMS``. Raises
    ``KeyError`` if the requested ``Z`` is not tabulated.
    """
    Z_int = int(Z)
    if Z_int not in _SALVAT_PARAMS:
        raise KeyError(
            f"Salvat-MMP per-Z coefficients not tabulated for Z={Z_int}. "
            "Drop the (alpha_i, lambda_i) table into "
            "neptune.nuclear_tools._SALVAT_PARAMS to enable this option."
        )
    a = BOHR_RADIUS_GEV_INV * np.power(float(Z), -1.0 / 3.0)  # a = a_0/Z^{1/3}
    Q2 = np.asarray(Q2, dtype=float)
    out = 0.0
    for alpha_i, lambda_i in _SALVAT_PARAMS[Z_int]:
        ki2 = (lambda_i / a) ** 2
        out = out + alpha_i * Q2 / (Q2 + ki2)
    return out


_ATOMIC_SCREENING_FUNCS = {
    "tsai":            _tsai_screening,
    "thomas-fermi":    _tsai_screening,
    "thomas_fermi":    _tsai_screening,
    "tf":              _tsai_screening,
    "moliere":         _moliere_screening,
    "moliere-salvat":  _moliere_screening,
    "salvat":          _salvat_screening,
}


def atomic_screening_factor(Q2, Z, kind: str = "tsai"):
    """
    Atomic charge-screening modulator ``G_at(Q²)`` for kind ∈
    {'tsai', 'moliere', 'salvat'}.

    ``G_at(0) = 0`` (photon sees a neutral atom) and ``G_at(∞) = 1``
    (photon resolves the bare nucleus). Multiply ``F_nuclear(Q²)`` by
    this factor to obtain the Coulomb-screened effective charge form
    factor used in the coherent integrand.

    Parameters
    ----------
    Q2 : array_like
        Photon virtuality :math:`Q^2 = -q^2 > 0` in GeV².
    Z : int
        Atomic number of the target.
    kind : str
        Choice of screening parameterisation. ``'tsai'`` (default) is the
        Thomas–Fermi single-pole formula; ``'moliere'`` is a Z-independent
        3-exponential approximation to the Salvat function (used by
        GEANT4); ``'salvat'`` requires the per-Z (α_i, λ_i) table — see
        ``_SALVAT_PARAMS``.
    """
    key = kind.lower().replace(" ", "-")
    if key not in _ATOMIC_SCREENING_FUNCS:
        raise ValueError(
            f"Unknown atomic_screening kind {kind!r}. "
            f"Choose from {sorted(_ATOMIC_SCREENING_FUNCS)}."
        )
    return _ATOMIC_SCREENING_FUNCS[key](Q2, Z)


def make_atomic_screening(Z, kind: str = "tsai"):
    """Return a callable ``f(Q²) → G_at(Q²)`` for the given nucleus."""
    if kind is None:
        return lambda Q2: np.ones_like(np.asarray(Q2, dtype=float))
    key = kind.lower().replace(" ", "-")
    if key not in _ATOMIC_SCREENING_FUNCS:
        raise ValueError(
            f"Unknown atomic_screening kind {kind!r}. "
            f"Choose from {sorted(_ATOMIC_SCREENING_FUNCS)}."
        )
    fn = _ATOMIC_SCREENING_FUNCS[key]
    Z_val = float(Z)
    return lambda Q2: fn(Q2, Z_val)


def apply_atomic_screening(form_factor_callable, Z, kind: str = "tsai"):
    """
    Wrap ``form_factor(Q²) → F(Q²)`` so the returned callable evaluates the
    *screened* effective form factor ``F(Q²) · G_at(Q²)``.

    Returns the original callable unchanged if ``kind is None``.
    """
    if kind is None:
        return form_factor_callable
    G = make_atomic_screening(Z, kind)
    F = form_factor_callable

    def screened(Q2):
        return F(Q2) * G(Q2)

    return screened
