"""
neptune.const
=============
Physical constants and SM parameters for neutrino trident calculations.

Imports everything from DarkNews.const and adds/overrides trident-specific values.
Values match those in cross_sections.h of the original BSMscan C++ code.
"""

# Import all from DarkNews.const first
from DarkNews.const import *  # noqa: F401, F403

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
# After a Fierz rearrangement, W and Z exchange combine into
#
#   L_eff = (G_F/√2) [ν̄ γ^a (1 - γ5) ν] [ℓ̄1 γ_a (Vijk - Aijk γ5) ℓ2],
#
# with ℓ1 the negative and ℓ2 the positive charged lepton. The CC diagram
# exists when the charged lepton of the projectile's own flavor appears:
# ℓ1⁻ for ν_α (ν_α → ℓ_α⁻ W⁺), ℓ2⁺ for ν̄_α (ν̄_α → ℓ_α⁺ W⁻). The NC diagram
# exists when ℓ1 = ℓ2.
#
#   CC + NC  (ν_μ → μ⁻μ⁺, ν̄_μ → μ⁻μ⁺):   Vijk = 1/2 + 2 s_W²,  Aijk = 1/2
#   CC only  (ν_μ → μ⁻e⁺, ν̄_μ → e⁻μ⁺):   Vijk = 1,             Aijk = 1
#   NC only  (ν_μ → e⁻e⁺):               Vijk = -1/2 + 2 s_W², Aijk = -1/2
#
# These are vertex couplings, identical for ν and ν̄. The antineutrino
# neutrino-line trace flips the sign of the V×A interference, which
# TridentSMModel applies to the VA term after any Z' shift. Equivalently,
# σ(ν̄_α → ℓ1⁻ ℓ2⁺) = σ(ν_α → ℓ2⁻ ℓ1⁺) by CP at one-photon exchange.


def _trident_channel(nu_flavor: str, l1_flavor: str, l2_flavor: str, is_nubar: bool):
    """Return (has_CC, has_NC) for the trident channel, or raise if forbidden."""
    cc_lepton = l2_flavor if is_nubar else l1_flavor
    has_cc = cc_lepton == nu_flavor
    has_nc = l1_flavor == l2_flavor
    if not (has_cc or has_nc):
        kind = "nubar" if is_nubar else "nu"
        raise ValueError(
            f"Channel {kind}_{nu_flavor} -> {l1_flavor}- {l2_flavor}+ violates "
            f"lepton-flavor conservation. For {kind}_{nu_flavor}, the "
            f"{'positive' if is_nubar else 'negative'} lepton must be "
            f"'{nu_flavor}' when l1 != l2."
        )
    return has_cc, has_nc


def get_Vijk(
    nu_flavor: str, l1_flavor: str, l2_flavor: str, is_nubar: bool = False
) -> float:
    """
    SM vector vertex coupling Vijk for ν_α (ν̄_α) → ℓ1⁻ ℓ2⁺.

    ``is_nubar`` selects which charged lepton closes the CC diagram; it does
    not change the sign. Raises ValueError for flavor-violating channels.
    """
    has_cc, has_nc = _trident_channel(nu_flavor, l1_flavor, l2_flavor, is_nubar)
    return (1.0 if has_cc else 0.0) + (gV_lep if has_nc else 0.0)


def get_Aijk(
    nu_flavor: str, l1_flavor: str, l2_flavor: str, is_nubar: bool = False
) -> float:
    """
    SM axial vertex coupling Aijk for ν_α (ν̄_α) → ℓ1⁻ ℓ2⁺.

    The antineutrino sign flip of the V×A term is applied by the model,
    not here. Raises ValueError for flavor-violating channels.
    """
    has_cc, has_nc = _trident_channel(nu_flavor, l1_flavor, l2_flavor, is_nubar)
    return (1.0 if has_cc else 0.0) + (gA_lep if has_nc else 0.0)


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
