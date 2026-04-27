"""
neptune.model
=============
Model classes for neutrino trident cross section calculations.

Mirrors the DarkNews model.py (HNLModel) but for trident physics.
A model encapsulates:
  - The lepton channel (nu_alpha → nu l1 l2)
  - The SM and BSM couplings
  - Helper methods to compute effective coupling combinations (Vijk, Aijk)
    and BSM propagator factors

Following the C++ BSMscan conventions:
  V2, A2, VA in the matrix element are the *effective squared couplings*:
    SM only:         V2 = Vijk²,  A2 = Aijk², VA = Vijk*Aijk
    SM+BSM (full):   V2 = (Vijk + VBSM/prop)², etc. (see final_xsec in integrator.cxx)
    Basis pieces:    V2=1,A2=0,AV=0 → pure V²-basis component only (for separate integration)

For neptune we use the "full" approach in the integrand directly.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from neptune import const


@dataclass
class TridentSMModel:
    """
    Standard Model W+Z trident model.

    The process is:  nu_alpha + N → nu' + l1⁻ + l2⁺ + N'

    The amplitude involves two diagrams (Diag11, Diag22) and their interference
    (Diag12).  The effective couplings Vijk and Aijk encode the W+Z structure
    (see integrator.cxx BSMscan constructor).

    Attributes
    ----------
    nu_flavor : str
        Incoming neutrino flavor: 'e', 'mu', or 'tau'.
    l1_flavor : str
        Flavor of the negatively charged lepton (l1⁻).
    l2_flavor : str
        Flavor of the positively charged lepton (l2⁺).
    is_nubar : bool
        If True, compute antineutrino cross section (flips Aijk sign).
    Diag11 : float
        Include diagram 1 × diagram 1 (default 1).
    Diag22 : float
        Include diagram 2 × diagram 2 (default 1).
    Diag12 : float
        Include diagram 1 × diagram 2 interference (default 1).
    """

    nu_flavor: str = "mu"
    l1_flavor: str = "mu"
    l2_flavor: str = "mu"
    is_nubar: bool = False

    # Diagram flags (can be set to 0 to suppress individual diagrams)
    Diag11: float = 1.0
    Diag22: float = 1.0
    Diag12: float = 1.0

    def __post_init__(self):
        self.ml1 = const.get_lepton_mass(self.l1_flavor)
        self.ml2 = const.get_lepton_mass(self.l2_flavor)
        self.Vijk = const.get_Vijk(
            self.nu_flavor, self.l1_flavor, self.l2_flavor, self.is_nubar
        )
        self.Aijk = const.get_Aijk(
            self.nu_flavor, self.l1_flavor, self.l2_flavor, self.is_nubar
        )
        self.channel_flag = const.CHANNEL_FLAGS.get(
            (self.nu_flavor, self.l1_flavor, self.l2_flavor), -1
        )

    @property
    def lepton_sum_mass(self) -> float:
        return self.ml1 + self.ml2

    def get_coupling_terms(self, m6: np.ndarray = None) -> dict:
        """
        Return the effective coupling combinations V2, A2, VA for the matrix element.

        For the SM-only model these are constant: Vijk², Aijk², Vijk·Aijk.
        The optional m6 argument is ignored (kept for interface compatibility with BSM).

        Returns
        -------
        dict with keys 'V2', 'A2', 'VA'
        """
        return {
            "V2": self.Vijk**2,
            "A2": self.Aijk**2,
            "VA": self.Vijk * self.Aijk,
        }

    def couplings(self, nu_alpha, l1, l2, l=None, t=None, q=None):
        """
        Return (Vijk, Aijk) effective couplings for use in the EPA integrand.

        For the SM model, couplings are constant (no propagator dependence).
        The EPA integrand calls this with l, t, q as arrays.

        Returns
        -------
        (Vijk, Aijk) : scalar or ndarray
        """
        return self.Vijk, self.Aijk

    def __repr__(self):
        return (
            f"TridentSMModel(nu={self.nu_flavor}, l1={self.l1_flavor}, "
            f"l2={self.l2_flavor}, nubar={self.is_nubar})"
        )


@dataclass
class TridentBSMModel(TridentSMModel):
    """
    SM + BSM Z' trident model.

    Extends TridentSMModel with a light Z' mediator.  The BSM contribution
    enters through the effective couplings as:

        V_eff(m6) = Vijk + VBSM * prop(m6)
        A_eff(m6) = Aijk + ABSM * prop(m6)

    where
        VBSM = -gV_prime² / (2√2 Gf) * CHARGE
        ABSM = -gA_prime² / (2√2 Gf) * CHARGE
        prop(m6) = 1 / (2*m6 + mzprime²)

    and m6 is the integration variable related to p'_nu · p_l1.

    This matches the SMandBSM mode in the C++ code (scan.cxx / integrator.cxx).

    Attributes
    ----------
    mzprime : float
        Z' mass in GeV.
    gV_prime : float
        Z' vector coupling to the charged lepton l1 (or l2, assumed universal).
        Corresponds to gprimeV in C++ code.
    gA_prime : float
        Z' axial coupling to the charged lepton (gprimeA in C++ code).
        Default 0 (pure vector Z').
    CHARGE : float
        Nuclear charge coupling for Z' (coherent: Z of nucleus; diffractive: 1).
        Default 1.0 (must be set appropriately for the scattering regime).
    bsm_mode : int
        Integration mode: SM_ONLY=0, INTERFERENCE=1, BSM_ONLY=2, SM_AND_BSM=3.
        Default SM_AND_BSM (full calculation).
    """

    mzprime: float = 0.1
    gV_prime: float = 1e-3
    gA_prime: float = 0.0
    CHARGE: float = 1.0
    bsm_mode: int = field(default=const.SM_AND_BSM)

    def __post_init__(self):
        super().__post_init__()
        # pre-compute BSM coupling factors (constant, not m6-dependent)
        self._VBSM = -self.gV_prime**2 / (2.0 * np.sqrt(2.0) * const.Gf) * self.CHARGE
        self._ABSM = -self.gA_prime**2 / (2.0 * np.sqrt(2.0) * const.Gf) * self.CHARGE

    def get_coupling_terms(self, m6: np.ndarray = None) -> dict:
        """
        Compute effective V2, A2, VA at each value of the integration variable m6.

        Parameters
        ----------
        m6 : array-like or None
            Value(s) of the p'_nu · p_l1 dot product variable.
            If None and bsm_mode is SM_ONLY, returns SM-only couplings.

        Returns
        -------
        dict with keys 'V2', 'A2', 'VA', and optionally 'prop' (the propagator).
        """
        if self.bsm_mode == const.SM_ONLY or m6 is None:
            return {
                "V2": self.Vijk**2,
                "A2": self.Aijk**2,
                "VA": self.Vijk * self.Aijk,
            }

        m6 = np.asarray(m6, dtype=float)
        prop = 1.0 / (2.0 * m6 + self.mzprime**2)

        if self.bsm_mode == const.SM_AND_BSM:
            V_eff = self.Vijk + self._VBSM * prop
            A_eff = self.Aijk + self._ABSM * prop
            return {
                "V2": V_eff**2,
                "A2": A_eff**2,
                "VA": V_eff * A_eff,
                "prop": prop,
            }
        elif self.bsm_mode == const.INTERFERENCE:
            # SM × BSM cross term only
            return {
                "V2": 2.0 * self.Vijk * self._VBSM * prop,
                "A2": 2.0 * self.Aijk * self._ABSM * prop,
                "VA": self.Vijk * self._ABSM * prop + self.Aijk * self._VBSM * prop,
                "prop": prop,
            }
        elif self.bsm_mode == const.BSM_ONLY:
            prop2 = prop**2
            return {
                "V2": self._VBSM**2 * prop2,
                "A2": self._ABSM**2 * prop2,
                "VA": self._VBSM * self._ABSM * prop2,
                "prop": prop,
            }
        else:
            raise ValueError(f"Unknown bsm_mode={self.bsm_mode}")

    def couplings(self, nu_alpha, l1, l2, l=None, t=None, q=None):
        """
        Return (Vijk_eff, Aijk_eff) including BSM Z' propagator.

        The Z' propagator uses k^2 = l - t (where l = l_true + t_true as
        passed by the EPA integrand).  This matches the C++ dsigma_dPS
        convention: 1/(l - t - mzprime^2).

        Returns
        -------
        (Vijk_eff, Aijk_eff) : scalar or ndarray
        """
        if self.bsm_mode == const.SM_ONLY or l is None or t is None:
            return self.Vijk, self.Aijk

        l = np.asarray(l, dtype=float)
        t = np.asarray(t, dtype=float)
        # k^2 = l_true = l - t (since l passed to function = l_true + t_true)
        prop = 1.0 / (l - t - self.mzprime**2)

        if self.bsm_mode == const.SM_AND_BSM:
            V_eff = self.Vijk + self._VBSM * prop
            A_eff = self.Aijk + self._ABSM * prop
        elif self.bsm_mode == const.INTERFERENCE:
            V_eff = self.Vijk + self._VBSM * prop
            A_eff = self.Aijk + self._ABSM * prop
        elif self.bsm_mode == const.BSM_ONLY:
            V_eff = self._VBSM * prop
            A_eff = self._ABSM * prop
        else:
            raise ValueError(f"Unknown bsm_mode={self.bsm_mode}")
        return V_eff, A_eff

    def __repr__(self):
        return (
            f"TridentBSMModel(nu={self.nu_flavor}, l1={self.l1_flavor}, "
            f"l2={self.l2_flavor}, mzprime={self.mzprime:.3g}, "
            f"gV={self.gV_prime:.3g}, gA={self.gA_prime:.3g})"
        )
