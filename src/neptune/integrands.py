"""
integrands.py
=============
Vegas BatchIntegrand classes for neutrino trident cross sections.

Both regimes share the same 8-D (or 9-D with flux convolution) phase-space
mapping ``phase_space.map_unit_to_physical`` over the Lorentz invariants
x1..x6 plus two rotation angles. The differential cross section is built
from the T/L decomposition

    dσ = ( h_T(regime; x1, x2, ...) · σT_lep(x1..x6, ...)
            + h_L(regime; x1, x2, ...) · σL_lep(x1..x6, ...) )
            / (64 π² x1 x2)

where the leptonic σT, σL are identical for both regimes and only the
hadronic flux h_T, h_L distinguishes coherent (Woods-Saxon nuclear FF)
from diffractive (nucleon dipole H1/H2).

Each integrand exposes ``mode``:

  "full"          : keep both T and L pieces with full x1 dependence.
  "improved-epa"  : drop h_L · σL                (transverse only, x1≠0).
  "epa"           : drop h_L · σL AND set q²=0 in σT.

References
----------
Czyz et al., Phys.Rev. 177 (1969) 2311
``mathematica/BSM_trident.nb`` and ``mathematica/export_TL_pieces.wl``.
"""

import numpy as np
import vegas

from neptune import const
from neptune.const import GeV2_to_cm2
from neptune._tl_amplitude import differential_cross_section_tl
from neptune.nuclear_tools import FF_WS, F_pauli_block
from neptune.phase_space import map_unit_to_physical, trident_threshold


class CoherentTridentIntegrand(vegas.BatchIntegrand):
    """
    Coherent trident integrand on the 8-D x1..x6+x7+x8 phase space.

    Uses the Woods-Saxon nuclear form factor (or a user-supplied callable)
    on the coherent hadronic flux ``h_T_coh`` / ``h_L_coh``. Combined with
    the leptonic ``sigma_T_lep`` / ``sigma_L_lep`` via
    ``differential_cross_section_tl``.
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
        form_factor=None,
        mode="full",
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
        self.form_factor = form_factor
        self.mode = mode

        self.bsm_mode = getattr(model, "bsm_mode", const.SM_ONLY)
        self.mzprime = getattr(model, "mzprime", 0.0)
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

        lab_threshold = trident_threshold(self.ml1, self.ml2, self.Mn)
        energy_valid = Enu > lab_threshold
        if not np.any(energy_valid):
            return np.zeros((nbatch, 1))

        Enu_phase = np.where(energy_valid, Enu, lab_threshold * (1.0 + 1e-9))
        ps = map_unit_to_physical(
            x_phase,
            Enu=Enu_phase,
            ml1=self.ml1,
            ml2=self.ml2,
            Mn=self.Mn,
            mzprime=self.mzprime,
            bsm_mode=self.bsm_mode,
        )

        coup = self.model.get_coupling_terms(m6=ps["m6"])

        if self.form_factor is None:
            FF = FF_WS(np.sqrt(np.maximum(ps["x1"], 0.0)), self.A)
        else:
            FF = self.form_factor(ps["x1"])

        dsigma = differential_cross_section_tl(
            ps["x1"], ps["x2"], ps["x3"], ps["x4"], ps["x5"], ps["x6"],
            Enu, self.Mn, self.ml1, self.ml2,
            coup["V2"], coup["A2"], coup["VA"],
            regime="coherent",
            mode=self.mode,
            Z=self.Z, FormFactor=FF,
        )

        result = dsigma * ps["Jacob"] * GeV2_to_cm2
        result = np.where(
            energy_valid & ps["valid"] & np.isfinite(result),
            result,
            0.0,
        )

        if self.fixed_Enu is None:
            result = result * (self.Emax - self.Emin)
            if self.flux is not None:
                result = result * self.flux(Enu)

        return result.reshape(nbatch, 1)


class DiffractiveTridentIntegrand(vegas.BatchIntegrand):
    """
    Diffractive (per-nucleon) trident integrand on the 8-D phase space.

    Uses the nucleon dipole form-factor combinations ``HH1``, ``HH2`` on
    the diffractive hadronic flux ``h_T_dif`` / ``h_L_dif``. Pauli blocking
    is applied unless disabled.
    """

    def __init__(
        self,
        nu_alpha,
        l1,
        l2,
        model,
        ml1,
        ml2,
        Mn=None,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
        nucleon="proton",
        apply_pauli_blocking=True,
        mode="full",
    ):
        self.nu_alpha = nu_alpha
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.ml1 = ml1
        self.ml2 = ml2
        self.ml12 = ml1 + ml2
        self.ml12sq = self.ml12**2
        self.nucleon = nucleon
        if Mn is None:
            self.Mn = const.m_neutron if nucleon in ("n", "neutron") else const.m_proton
        else:
            self.Mn = Mn

        self.fixed_Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        self.apply_pauli_blocking = apply_pauli_blocking
        self.mode = mode

        self.bsm_mode = getattr(model, "bsm_mode", const.SM_ONLY)
        self.mzprime = getattr(model, "mzprime", 0.0)
        self.ndim = 8 if Enu is not None else 9

    def __call__(self, xx):
        from neptune.nuclear_tools import H1_n, H1_p, H2_n, H2_p

        xx = np.asarray(xx)
        nbatch = xx.shape[0]

        if self.fixed_Enu is not None:
            Enu = np.full(nbatch, self.fixed_Enu)
            x_phase = xx
        else:
            Enu = (self.Emax - self.Emin) * xx[:, 8] + self.Emin
            x_phase = xx[:, :8]

        lab_threshold = trident_threshold(self.ml1, self.ml2, self.Mn)
        energy_valid = Enu > lab_threshold
        if not np.any(energy_valid):
            return np.zeros((nbatch, 1))

        Enu_phase = np.where(energy_valid, Enu, lab_threshold * (1.0 + 1e-9))
        ps = map_unit_to_physical(
            x_phase,
            Enu=Enu_phase,
            ml1=self.ml1,
            ml2=self.ml2,
            Mn=self.Mn,
            mzprime=self.mzprime,
            bsm_mode=self.bsm_mode,
        )

        coup = self.model.get_coupling_terms(m6=ps["m6"])

        q = np.sqrt(np.maximum(ps["x1"], 0.0))
        if self.nucleon in ("p", "proton"):
            HH1, HH2 = H1_p(q), H2_p(q)
        else:
            HH1, HH2 = H1_n(q), H2_n(q)

        dsigma = differential_cross_section_tl(
            ps["x1"], ps["x2"], ps["x3"], ps["x4"], ps["x5"], ps["x6"],
            Enu, self.Mn, self.ml1, self.ml2,
            coup["V2"], coup["A2"], coup["VA"],
            regime="diffractive",
            mode=self.mode,
            HH1=HH1, HH2=HH2,
        )

        result = dsigma * ps["Jacob"] * GeV2_to_cm2
        if self.apply_pauli_blocking:
            result = result * F_pauli_block(ps["x1"])
        result = np.where(
            energy_valid & ps["valid"] & np.isfinite(result),
            result,
            0.0,
        )

        if self.fixed_Enu is None:
            result = result * (self.Emax - self.Emin)
            if self.flux is not None:
                result = result * self.flux(Enu)

        return result.reshape(nbatch, 1)
