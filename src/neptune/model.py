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

The matrix element is linear in the coupling combinations V2, A2, and VA:
    SM:       V2 = Vijk²,  A2 = Aijk²,  VA = ±Vijk Aijk
    SM + Z':  V_eff = Vijk + VBSM·prop(m6), A_eff = Aijk + ABSM·prop(m6),
              V2 = V_eff², A2 = A_eff², VA = ±V_eff A_eff
with the minus sign for antineutrinos.
"""

from dataclasses import dataclass, field
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
        If True, the projectile is ν̄_α. The CC diagram then requires
        ``l2_flavor == nu_flavor`` (ν̄_α → ℓ_α⁺ W⁻), and the V×A term
        changes sign. ``Vijk`` and ``Aijk`` are vertex couplings, the same
        for ν and ν̄.
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
        # ν̄ flips the antisymmetric part of the neutrino tensor, i.e. VA → -VA.
        self._va_sign = -1.0 if self.is_nubar else 1.0
        # CP image of a ν̄ channel is the ν channel with ℓ1 ↔ ℓ2.
        key = (
            (self.nu_flavor, self.l2_flavor, self.l1_flavor)
            if self.is_nubar
            else (self.nu_flavor, self.l1_flavor, self.l2_flavor)
        )
        self.channel_flag = const.CHANNEL_FLAGS.get(key, -1)

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
            "VA": self._va_sign * self.Vijk * self.Aijk,
        }

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

    and m6 = p_ν · p'_ν, so that -2 m6 is the momentum transfer squared of
    the neutrino line. The Z' couples flavor-diagonally, so VBSM = ABSM = 0
    when l1 != l2. For ν̄ the sign flip of the VA term acts on the full
    V_eff × A_eff product.

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
        norm = -self.CHARGE / (2.0 * np.sqrt(2.0) * const.Gf)
        if self.l1_flavor != self.l2_flavor:
            norm = 0.0  # flavor-diagonal Z' cannot produce l1- l2+ with l1 != l2
        self._VBSM = norm * self.gV_prime**2
        self._ABSM = norm * self.gA_prime**2

    def get_coupling_terms(self, m6: np.ndarray = None) -> dict:
        """
        Compute effective V2, A2, VA at each value of the integration variable m6.

        Parameters
        ----------
        m6 : array-like or None
            Value(s) of m6 = p_ν · p'_ν.
            If None and bsm_mode is SM_ONLY, returns SM-only couplings.

        Returns
        -------
        dict with keys 'V2', 'A2', 'VA', and optionally 'prop' (the propagator).
        """
        if self.bsm_mode == const.SM_ONLY or m6 is None:
            return {
                "V2": self.Vijk**2,
                "A2": self.Aijk**2,
                "VA": self._va_sign * self.Vijk * self.Aijk,
            }

        m6 = np.asarray(m6, dtype=float)
        prop = 1.0 / (2.0 * m6 + self.mzprime**2)

        if self.bsm_mode == const.SM_AND_BSM:
            V_eff = self.Vijk + self._VBSM * prop
            A_eff = self.Aijk + self._ABSM * prop
            return {
                "V2": V_eff**2,
                "A2": A_eff**2,
                "VA": self._va_sign * V_eff * A_eff,
                "prop": prop,
            }
        elif self.bsm_mode == const.INTERFERENCE:
            # SM × BSM cross term only
            return {
                "V2": 2.0 * self.Vijk * self._VBSM * prop,
                "A2": 2.0 * self.Aijk * self._ABSM * prop,
                "VA": self._va_sign
                * (self.Vijk * self._ABSM + self.Aijk * self._VBSM)
                * prop,
                "prop": prop,
            }
        elif self.bsm_mode == const.BSM_ONLY:
            prop2 = prop**2
            return {
                "V2": self._VBSM**2 * prop2,
                "A2": self._ABSM**2 * prop2,
                "VA": self._va_sign * self._VBSM * self._ABSM * prop2,
                "prop": prop,
            }
        else:
            raise ValueError(f"Unknown bsm_mode={self.bsm_mode}")

    def __repr__(self):
        return (
            f"TridentBSMModel(nu={self.nu_flavor}, l1={self.l1_flavor}, "
            f"l2={self.l2_flavor}, mzprime={self.mzprime:.3g}, "
            f"gV={self.gV_prime:.3g}, gA={self.gA_prime:.3g})"
        )
