"""
neptune.const
=============
Physical constants and SM parameters for neutrino trident calculations.

Imports everything from DarkNews.const and adds/overrides trident-specific values.
Values match those in cross_sections.h of the original BSMscan C++ code.
"""

# Import all from DarkNews.const first
from DarkNews.const import *  # noqa: F401, F403

import numpy as np

# ─── Particle masses (GeV) ────────────────────────────────────────────────────
m_e = 0.511e-3  # electron mass
m_mu = 105.7e-3  # muon mass
m_tau = 1.777  # tau mass
m_proton = 0.93827  # proton mass
m_neutron = 0.93956  # neutron mass
m_AVG = 0.9389  # average nucleon mass (used in C++ code)

# ─── Fundamental constants ─────────────────────────────────────────────────────
Gf = 1.1663787e-5  # Fermi constant (GeV^-2)
alphaQED = 1.0 / 137.035  # fine structure constant
sw2 = 0.2223  # sin^2(theta_W)

# ─── Unit conversion ───────────────────────────────────────────────────────────
GeV2_to_cm2 = 3.9204e-28  # conversion: 1 GeV^-2 = 3.9204e-28 cm^2

# ─── SM Z couplings to charged leptons (NC) ───────────────────────────────────
# Convention from C++ BSMscan:
#   gL_lep = sw2 - 0.5   (left-handed lepton coupling to Z)
#   gR_lep = sw2         (right-handed lepton coupling to Z)
#   Vector coupling:  gV_lep = gL_lep + gR_lep = 2*sw2 - 0.5
#   Axial coupling:   gA_lep = gL_lep - gR_lep = -0.5
gL_lep = sw2 - 0.5
gR_lep = sw2
gV_lep = gL_lep + gR_lep  # = 2*sw2 - 0.5
gA_lep = gL_lep - gR_lep  # = -0.5

# ─── SM coupling combinations for trident (Vijk, Aijk) ────────────────────────
# These come from the W+Z interference structure.
# For nu_alpha → nu_alpha l^- l^+  (charged current + neutral current):
#
#   If nu_alpha == l1_flavor == l2_flavor  (e.g., nu_mu mu mu):
#       Vijk = 0.5 + 2*sw2,  Aijk = 0.5
#
#   If nu_alpha == l1_flavor != l2_flavor  (e.g., nu_e e mu):
#       Vijk = 1.0,  Aijk = 1.0
#
#   If nu_alpha != l1_flavor  (purely NC trident, e.g., nu_mu e e):
#       Vijk = -0.5 + 2*sw2, Aijk = -0.5
#
# Defined as functions because they depend on the channel:


def get_Vijk(
    nu_flavor: str, l1_flavor: str, l2_flavor: str, is_nubar: bool = False
) -> float:
    """
    SM vector coupling combination Vijk for the trident amplitude.

    Parameters
    ----------
    nu_flavor : str
        Incoming neutrino flavor ('e', 'mu', 'tau').
    l1_flavor : str
        Flavor of the negatively charged lepton.
    l2_flavor : str
        Flavor of the positively charged lepton.
    is_nubar : bool
        If True, flip the sign (antineutrino). Not needed for Vijk (Aijk only).

    Returns
    -------
    float
    """
    if nu_flavor == l1_flavor:
        if l1_flavor == l2_flavor:
            return 0.5 + 2.0 * sw2
        else:
            return 1.0
    else:
        return -0.5 + 2.0 * sw2


def get_Aijk(
    nu_flavor: str, l1_flavor: str, l2_flavor: str, is_nubar: bool = False
) -> float:
    """
    SM axial coupling combination Aijk for the trident amplitude.

    Parameters
    ----------
    nu_flavor : str
        Incoming neutrino flavor ('e', 'mu', 'tau').
    l1_flavor : str
        Flavor of the negatively charged lepton.
    l2_flavor : str
        Flavor of the positively charged lepton.
    is_nubar : bool
        If True, flip the axial sign for antineutrino.

    Returns
    -------
    float
    """
    if nu_flavor == l1_flavor:
        if l1_flavor == l2_flavor:
            Aijk = 0.5
        else:
            Aijk = 1.0
    else:
        Aijk = -0.5

    if is_nubar:
        Aijk = -Aijk
    return Aijk


# ─── Lepton flavor helpers ──────────────────────────────────────────────────────
LEPTON_MASSES = {"e": m_e, "mu": m_mu, "tau": m_tau}


def get_lepton_mass(flavor: str) -> float:
    """Return lepton mass in GeV for flavor 'e', 'mu', or 'tau'."""
    try:
        return LEPTON_MASSES[flavor]
    except KeyError:
        raise ValueError(f"Unknown lepton flavor '{flavor}'. Use 'e', 'mu', or 'tau'.")


# ─── Channel flag map (mirrors C++ enum in cross_sections.h) ──────────────────
CHANNEL_FLAGS = {
    ("e", "e", "e"): 0,  # eeee
    ("mu", "mu", "mu"): 1,  # mmmm
    ("e", "e", "mu"): 2,  # emme
    ("mu", "e", "e"): 3,  # mmee
    ("e", "mu", "mu"): 4,  # eemm
    ("mu", "mu", "e"): 5,  # meem
    ("e", "tau", "tau"): 6,  # eett
    ("mu", "tau", "tau"): 7,  # mmtt
    ("tau", "tau", "tau"): 8,  # tttt
    ("e", "e", "tau"): 9,  # ette
    ("mu", "mu", "tau"): 10,  # mttm
    ("tau", "tau", "e"): 11,  # teet
    ("tau", "tau", "mu"): 12,  # tmmt
    ("tau", "e", "e"): 13,  # ttee
    ("tau", "mu", "mu"): 14,  # ttmm
}

# ─── BSM mode flags (mirrors C++ in cross_sections.h) ────────────────────────
SM_ONLY = 0  # pure SM W+Z
INTERFERENCE = 1  # SM × BSM interference
BSM_ONLY = 2  # pure BSM Z'²
SM_AND_BSM = 3  # full SM + BSM (linear + quadratic propagator)
