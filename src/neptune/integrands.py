"""
integrands.py
=============
Vegas BatchIntegrand classes for neutrino trident cross section integration.

Two integration schemes are provided:

DiffractiveTridentIntegrand
    6-dim (fixed Enu) or 7-dim (with flux convolution) integration using the
    equivalent-photon approximation (EPA) for nucleon targets.
    Variables: [s, phi, theta, t, l, q, (Enu)]
    Translated from Diffractive_EPA / Diffractive_EPA_E in integrands.cxx.

CoherentTridentIntegrand
    Same EPA integration scheme but with the nuclear (Woods-Saxon) form factor
    and Z^2 factor instead of the nucleon dipole form factor.
    Variables: [s, phi, theta, t, l, q, (Enu)]
    Translated from Coherent_EPA / Coherent_EPA_E in integrands.cxx.

References
----------
Czyz et al., Phys.Rev. 177 (1969) 2311
MH Hostert C++ implementation in bsm_trident/MH_Code/clean_code/
"""

import numpy as np
import vegas

from neptune import const
from neptune.const import m_proton, alphaQED, Gf, GeV2_to_cm2
from neptune.amplitudes import diffractive_dsigma, coherent_epa_dsigma, coherent_dsigma
from neptune.nuclear_tools import F_diffractive, FF_WS, F_pauli_block
from neptune.phase_space import map_unit_to_physical


# Default Q cuts in GeV (same convention as C++ Q_MAX, Q_MAX_DIFF in
# bsm_trident/.../cross_sections.h):
#   Q_max_coh  = Λ_QCD / A^(1/3)  — boundary between coherent (nuclear FF
#                                   has support) and diffractive (nucleon-elastic).
#                                   This is A-dependent; use Q_max_coh_for_A(A).
#   Q_max_diff = 1.0 GeV          — upper end of nucleon-elastic regime.
LAMBDA_QCD = 0.217  # GeV
Q_MAX_DIFF_DEFAULT = 1.0  # GeV


def Q_max_coh_for_A(A: float) -> float:
    """Return the standard A-dependent coherent/diffractive boundary Q [GeV]."""
    return LAMBDA_QCD / float(A) ** (1.0 / 3.0)


# Backwards-compatible scalar default (used only if A is unavailable).
# Roughly matches Argon-40 (0.064 GeV).
Q_MAX_COH_DEFAULT = Q_max_coh_for_A(40)


class DiffractiveTridentIntegrand(vegas.BatchIntegrand):
    """
    Vegas BatchIntegrand for diffractive trident production.

    Integration domain is the 6-dimensional (or 7-dim with flux) unit
    hypercube corresponding to the physical variables:

        xx[0] = s_unit   → s  in [(ml1+ml2)^2, 2*Enu*q - (ml1+ml2)^2]
        xx[1] = phi_unit → phi in [0, 2*pi]
        xx[2] = theta_unit → theta in [0, pi]
        xx[3] = t_unit   → t  in [(ml1+ml2)^2, s]
        xx[4] = l_unit   → l  in [(ml1+ml2)^2 - t, 0]  (see variable sub.)
        xx[5] = q_unit   → q  in [q_min, Q_max_diff]
        xx[6] = E_unit   → Enu in [Emin, Emax]  (optional, for flux integral)

    The Jacobian = pi * 2*pi * (2*Enu*q - ml12^2) * (t - ml12^2) * (s - ml12^2)
                   * (Q_max - q_min)  [* (Emax-Emin) for flux]

    Parameters
    ----------
    nu_alpha : str
        Neutrino flavour ('e', 'mu', 'tau').
    l1 : str
        Lepton l- flavour ('e', 'mu', 'tau').
    l2 : str
        Lepton l+ flavour ('e', 'mu', 'tau').
    model : TridentSMModel or TridentBSMModel
        Physics model providing Vijk, Aijk couplings.
    ml1, ml2 : float
        Lepton masses [GeV].
    Enu : float or None
        Fixed neutrino energy [GeV]. If None, flux integration is used.
    Emin, Emax : float
        Energy range for flux integration [GeV].
    flux : callable or None
        flux(Enu) [neutrinos/GeV/cm^2/...]. Required if Enu is None.
    Q_max_coh : float
        Upper Q limit for coherent scattering (lower limit for diffractive).
    Q_max_diff : float
        Upper Q limit for diffractive scattering.
    """

    def __init__(
        self,
        nu_alpha,
        l1,
        l2,
        model,
        ml1,
        ml2,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
        Q_max_coh=Q_MAX_COH_DEFAULT,
        Q_max_diff=Q_MAX_DIFF_DEFAULT,
    ):
        self.nu_alpha = nu_alpha
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.ml1 = ml1
        self.ml2 = ml2
        self.ml12 = ml1 + ml2
        self.ml12sq = self.ml12**2

        self.fixed_Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux

        self.Q_max_coh = Q_max_coh
        self.Q_max_diff = Q_max_diff

        self.ndim = 6 if Enu is not None else 7

    def __call__(self, xx):
        """Evaluate the integrand at batch of unit-hypercube points xx."""
        # xx shape: (nbatch, ndim)
        xx = np.asarray(xx)
        nbatch = xx.shape[0]

        s_u = xx[:, 0]
        phi_u = xx[:, 1]
        theta_u = xx[:, 2]
        t_u = xx[:, 3]
        l_u = xx[:, 4]
        q_u = xx[:, 5]

        if self.fixed_Enu is not None:
            Enu = np.full(nbatch, self.fixed_Enu)
        else:
            E_u = xx[:, 6]
            Enu = (self.Emax - self.Emin) * E_u + self.Emin

        ml12sq = self.ml12sq

        q_min = np.maximum(self.Q_max_coh, ml12sq / (2.0 * Enu))
        q_max = self.Q_max_diff

        # Variable substitutions (unit cube → physical)
        q_true = q_u * (q_max - q_min) + q_min
        s_true = s_u * (2 * Enu * q_true - ml12sq) + ml12sq
        phi_true = phi_u * 2 * np.pi
        theta_true = theta_u * np.pi
        t_true = t_u * (s_true - ml12sq) + ml12sq
        l_true = (
            l_u * (t_true - ml12sq) + ml12sq - t_true
        )  # l_true in [ml12sq-t_true, 0]

        # Jacobian
        jacob = (
            np.pi
            * 2
            * np.pi
            * (2 * Enu * q_true - ml12sq)
            * (t_true - ml12sq)
            * (s_true - ml12sq)
            * (q_max - q_min)
        )
        if self.fixed_Enu is None:
            jacob *= self.Emax - self.Emin

        # Get couplings: Vijk, Aijk (may depend on l, t, q for BSM)
        Vijk, Aijk = self.model.couplings(
            self.nu_alpha,
            self.l1,
            self.l2,
            l=l_true + t_true,
            t=t_true,
            q=q_true,
        )

        # Evaluate matrix element (diffractive nucleon)
        dsigma = diffractive_dsigma(
            s_true,
            phi_true,
            theta_true,
            t_true,
            l_true + t_true,
            q_true,
            Enu,
            self.ml1,
            self.ml2,
            Vijk,
            Aijk,
        )

        # Pauli blocking suppression
        pauli = F_pauli_block(q_true)

        result = dsigma * jacob * pauli

        if self.fixed_Enu is None and self.flux is not None:
            result *= self.flux(Enu)

        # Convert from multiple evaluations to shape (nbatch, 1)
        return result.reshape(nbatch, 1)


class CoherentTridentIntegrand(vegas.BatchIntegrand):
    """
    Vegas BatchIntegrand for coherent (nuclear) trident production.

    Uses the same EPA integration scheme as DiffractiveTridentIntegrand but
    with the nuclear Woods-Saxon form factor and Z^2 instead of the nucleon
    dipole form factor.

    Integration variables and Jacobian are identical to DiffractiveTridentIntegrand
    but q runs in [q_min_coh, Q_max_coh] (the coherent domain).

    Parameters
    ----------
    nu_alpha : str
        Neutrino flavour ('e', 'mu', 'tau').
    l1 : str
        Lepton l- flavour ('e', 'mu', 'tau').
    l2 : str
        Lepton l+ flavour ('e', 'mu', 'tau').
    model : TridentSMModel or TridentBSMModel
        Physics model providing Vijk, Aijk couplings.
    ml1, ml2 : float
        Lepton masses [GeV].
    Z, A : int
        Atomic and mass numbers of the nucleus.
    Enu : float or None
        Fixed neutrino energy [GeV]. If None, flux integration is used.
    Emin, Emax : float
        Energy range for flux integration [GeV].
    flux : callable or None
        flux(Enu) [neutrinos/GeV/...]. Required if Enu is None.
    Q_max_coh : float
        Upper Q limit for coherent integration.
    """

    def __init__(
        self,
        nu_alpha,
        l1,
        l2,
        model,
        ml1,
        ml2,
        Z,
        A,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
        Q_max_coh=Q_MAX_COH_DEFAULT,
    ):
        self.nu_alpha = nu_alpha
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.ml1 = ml1
        self.ml2 = ml2
        self.ml12 = ml1 + ml2
        self.ml12sq = self.ml12**2
        self.Z = Z
        self.A = A

        self.fixed_Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux

        self.Q_max_coh = Q_max_coh

        self.ndim = 6 if Enu is not None else 7

    def __call__(self, xx):
        """Evaluate the integrand at batch of unit-hypercube points xx."""
        xx = np.asarray(xx)
        nbatch = xx.shape[0]

        s_u = xx[:, 0]
        phi_u = xx[:, 1]
        theta_u = xx[:, 2]
        t_u = xx[:, 3]
        l_u = xx[:, 4]
        q_u = xx[:, 5]

        if self.fixed_Enu is not None:
            Enu = np.full(nbatch, self.fixed_Enu)
        else:
            E_u = xx[:, 6]
            Enu = (self.Emax - self.Emin) * E_u + self.Emin

        ml12sq = self.ml12sq

        q_min = ml12sq / (2.0 * Enu)
        q_max = self.Q_max_coh

        # Skip points where q_min >= q_max (below threshold)
        valid = q_min < q_max
        if not np.any(valid):
            return np.zeros((nbatch, 1))

        q_true = q_u * (q_max - q_min) + q_min
        s_true = s_u * (2 * Enu * q_true - ml12sq) + ml12sq
        phi_true = phi_u * 2 * np.pi
        theta_true = theta_u * np.pi
        t_true = t_u * (s_true - ml12sq) + ml12sq
        l_true = l_u * (t_true - ml12sq) + ml12sq - t_true

        jacob = (
            np.pi
            * 2
            * np.pi
            * (2 * Enu * q_true - ml12sq)
            * (t_true - ml12sq)
            * (s_true - ml12sq)
            * (q_max - q_min)
        )
        if self.fixed_Enu is None:
            jacob *= self.Emax - self.Emin

        Vijk, Aijk = self.model.couplings(
            self.nu_alpha,
            self.l1,
            self.l2,
            l=l_true + t_true,
            t=t_true,
            q=q_true,
        )

        dsigma = coherent_epa_dsigma(
            s_true,
            phi_true,
            theta_true,
            t_true,
            l_true + t_true,
            q_true,
            Enu,
            self.ml1,
            self.ml2,
            Vijk,
            Aijk,
            self.Z,
            self.A,
        )

        result = dsigma * jacob
        result = np.where(valid, result, 0.0)

        if self.fixed_Enu is None and self.flux is not None:
            result *= self.flux(Enu)

        return result.reshape(nbatch, 1)


class CoherentTridentIntegrandFull(vegas.BatchIntegrand):
    """
    Full (non-EPA) coherent trident integrand on the 8-D physical phase space.

    .. warning::

        **Experimental.** The matrix element ``coherent_dsigma`` and phase-space
        Jacobian ``map_unit_to_physical`` were auto-translated from the C++
        ``cross_sections.cxx`` polynomial. Their overall normalisation has not
        yet been validated against the C++ reference for this Python build —
        smoke comparisons against the EPA path show O(few) discrepancies that
        are still under investigation. **Use EPA (``use_epa=True``, default)
        for production results.** The shape of differential distributions in
        ``x1, x2, ..., x6`` should still be correct.

    Uses ``coherent_dsigma`` (the exact polynomial matrix element from
    ``cross_sections.cxx``) together with ``phase_space.map_unit_to_physical``
    to integrate over all 8 invariants (x1..x6, x7, x8) with a log/inverse
    mapping for x1, x3 and m6 that flattens the dominant peaks (Q^2, dilepton
    mass, BSM propagator).

    This is the ``Coherent`` variant of the C++ integrator (no EPA).  Once
    validated, it will be the recommended path when:

      * The Z' is light (M_Z' ~ tens of MeV) — EPA misses propagator-driven
        kinematics outside the EPA region.
      * Higher precision is needed for the coherent regime overall.

    Domain
    ------
    8-D unit hypercube (or 9-D with flux).  Variables match
    ``map_unit_to_physical``: [u1_s, u2_s, u3_s, PHI2_s, x5_s, u6_s,
    x7_s, x8_s, (E_s)].

    Parameters
    ----------
    nu_alpha, l1, l2 : str
    model : TridentSMModel or TridentBSMModel
    ml1, ml2 : float
    Z, A : int
    Mn : float
        Target nuclear mass [GeV].  Defaults to A * m_AVG.
    Enu : float or None
    Emin, Emax : float
    flux : callable or None
    """

    def __init__(
        self,
        nu_alpha,
        l1,
        l2,
        model,
        ml1,
        ml2,
        Z,
        A,
        Mn=None,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
    ):
        self.nu_alpha = nu_alpha
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.ml1 = ml1
        self.ml2 = ml2
        self.ml12 = ml1 + ml2
        self.ml12sq = self.ml12**2
        self.Z = Z
        self.A = A
        self.Mn = Mn if Mn is not None else float(A) * const.m_AVG

        self.fixed_Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux

        # bsm_mode and mzprime drive the m6 variable change
        self.bsm_mode = getattr(model, "bsm_mode", const.SM_ONLY)
        self.mzprime = getattr(model, "mzprime", 0.0)

        # 8-D phase space + optional Enu axis
        self.ndim = 8 if Enu is not None else 9

    def __call__(self, xx):
        xx = np.asarray(xx)
        nbatch = xx.shape[0]

        if self.fixed_Enu is not None:
            Enu = np.full(nbatch, self.fixed_Enu)
            x_phase = xx
        else:
            Enu = (self.Emax - self.Emin) * xx[:, 8] + self.Emin
            x_phase = xx[:, :8]

        ps = map_unit_to_physical(
            x_phase,
            Enu=Enu,
            ml1=self.ml1,
            ml2=self.ml2,
            Mn=self.Mn,
            mzprime=self.mzprime,
            bsm_mode=self.bsm_mode,
        )

        # Effective squared couplings on the m6 grid
        coup = self.model.get_coupling_terms(m6=ps["m6"])
        V2 = coup["V2"]
        A2 = coup["A2"]
        VA = coup["VA"]

        dsigma = coherent_dsigma(
            ps["x1"], ps["x2"], ps["x3"], ps["x4"], ps["x5"], ps["x6"],
            Enu, self.Mn, self.ml1, self.ml2,
            V2, A2, VA, self.Z, self.A,
            Diag11=getattr(self.model, "Diag11", 1.0),
            Diag22=getattr(self.model, "Diag22", 1.0),
            Diag12=getattr(self.model, "Diag12", 1.0),
        )

        # coherent_dsigma is in raw GeV units; Jacob has the remaining GeV
        # powers so result*Jacob is dimensionless / GeV^-2; convert to cm^2.
        result = dsigma * ps["Jacob"] * GeV2_to_cm2
        result = np.where(ps["valid"] & np.isfinite(result), result, 0.0)

        if self.fixed_Enu is None:
            # Account for the (Emax - Emin) Jacobian on the energy variable
            result = result * (self.Emax - self.Emin)
            if self.flux is not None:
                result = result * self.flux(Enu)

        return result.reshape(nbatch, 1)
