"""
neptune.nuclear_tools
=====================
Nuclear target utilities for neutrino trident calculations.

Re-exports DarkNews NuclearTarget and related helpers, and adds
form factors used in the BSMscan C++ code:
  - FF_WS : Woods-Saxon coherent form factor (used for coherent trident)
  - H1_p, H2_p, H1_n, H2_n : nucleon elastic dipole form factors
  - F_diffractive : proton Sachs form factor (same as H1_p; alias)
  - F_pauli_block : Pauli blocking suppression factor

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


# ─── Coherent (nuclear) form factor: Woods-Saxon ──────────────────────────────


def FF_WS(q: float, A: float) -> float:
    """
    Woods-Saxon nuclear electromagnetic form factor (scalar).

    Parameters
    ----------
    q : float
        |q| = sqrt(Q²) in GeV.
    A : float
        Atomic mass number.

    Returns
    -------
    float
        Dimensionless form factor F(q, A).
    """
    r0 = (
        1.07 / 0.197 * A ** (1.0 / 3.0)
    )  # GeV^-1, converted from fm via ℏc=0.197 GeV·fm
    a = 0.57 / 0.197  # GeV^-1

    num = (
        3.0
        * np.pi
        * q
        * a
        * (
            np.pi * q * a * (1.0 / np.tanh(np.pi * q * a)) * np.sin(q * r0)
            - q * r0 * np.cos(q * r0)
        )
    )
    den = q**3 * np.sinh(np.pi * q * a) * (r0**2 + np.pi**2 * a**2) * r0
    return num / den


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


# ─── Diffractive form factor (proton Sachs, same as F_diffractive in C++) ─────


def F_diffractive(Q2: np.ndarray) -> np.ndarray:
    """
    Proton Sachs-like form factor for diffractive (single-nucleon) scattering.

    F(Q²) = (G_dip + tau * qsi * G_dip) / (1 + tau)   with qsi = MAG_MOM_P - 1

    This matches the C++ F_diffractive function.
    """
    Q2 = np.asarray(Q2, dtype=float)
    q = np.sqrt(np.maximum(Q2, 0.0))
    tau = Q2 / (4.0 * const.m_AVG**2)
    qsi = _MAG_MOM_P - 1.0  # anomalous magnetic moment
    Gd = _G_dip(q)
    return (Gd + tau * qsi * Gd) / (1.0 + tau)


# ─── Pauli blocking factor (for diffractive scattering) ───────────────────────

_P_FERMI = 0.2210  # Fermi momentum in GeV (for Ar-40; typical value from C++ code)


def F_pauli_block(Q2: np.ndarray, p_fermi: float = _P_FERMI) -> np.ndarray:
    """
    Pauli blocking suppression factor for diffractive (quasi-elastic) scattering.

    F_PB(Q²) = (3/2)(|q|/2pF) - (1/2)(|q|/2pF)³   if |q| ≤ 2pF
             = 1                                      otherwise

    where |q| = sqrt(Q² * (1 + Q²/4M²)).

    Parameters
    ----------
    Q2 : array-like
        Squared momentum transfer Q² ≥ 0 (GeV²).
    p_fermi : float
        Fermi momentum in GeV.  Default is 0.221 GeV (Ar-40).

    Returns
    -------
    np.ndarray
    """
    Q2 = np.asarray(Q2, dtype=float)
    qvec = np.sqrt(Q2 * (1.0 + Q2 / (4.0 * const.m_AVG**2)))
    xi = qvec / (2.0 * p_fermi)
    return np.where(xi <= 1.0, 1.5 * xi - 0.5 * xi**3, 1.0)
