"""
integrands.py
=============
Vegas BatchIntegrand classes for neutrino trident cross sections.

The elastic regimes (coherent, diffractive, electron) share the 8-D (9-D
with flux convolution) phase-space mapping ``phase_space.map_unit_to_physical``
over the Lorentz invariants x1..x6 plus two rotation angles. The inelastic
regimes (DIS, resonance) add Bjorken xB through
``phase_space.map_dis_unit_to_trident_physical``. The differential cross
section is built from the T/L decomposition

    dσ = ( h_T(regime; x1, x2, ...) · σT_lep(x1..x6, ...)
            + h_L(regime; x1, x2, ...) · σL_lep(x1..x6, ...) )
            / (64 π² x1 x2)

where the leptonic σT, σL are common to all regimes and only the hadronic
flux h_T, h_L changes: nuclear form factor (coherent), nucleon H1/H2
(diffractive), structure functions F1/F2 (DIS, resonance), or a point-like
electron.

Each integrand exposes ``mode``:

  "full"          : keep both T and L pieces with full x1 dependence.
  "improved-epa"  : drop h_L · σL                (transverse only, x1≠0).
  "epa"           : drop h_L · σL AND set q²=0 in σT.

References
----------
Czyz et al., Phys.Rev. 177 (1969) 2311
``mathematica/BSM_trident.nb``.
"""

import numpy as np
import vegas

from neptune import const
from neptune.const import GeV2_to_cm2
from neptune._tl_amplitude import differential_cross_section_tl
from neptune.nuclear_models import FermiGas, FreeNucleon, _NuclearModelBase
from neptune.nuclear_tools import FF_WS_Q2, H1_n, H1_p, H2_n, H2_p
from neptune.phase_space import (
    map_dis_unit_to_trident_physical,
    map_unit_to_physical,
    trident_threshold,
)


def _check_nuclear_model(nuclear_model, default):
    if nuclear_model is None:
        return default
    if not isinstance(nuclear_model, _NuclearModelBase):
        raise TypeError(
            "nuclear_model must be an instance of "
            "neptune.nuclear_models.{FreeNucleon, FermiGas, LocalFermiGas}"
        )
    return nuclear_model


def _qmag(Q2, Mn, xB=1.0):
    """
    Three-momentum transfer in the target rest frame,
    ``|q|² = Q² + ν²`` with ``ν = Q²/(2 Mn xB)``. ``xB = 1`` is elastic.
    """
    nu = Q2 / (2.0 * Mn * xB)
    return np.sqrt(Q2 + nu * nu)


class _TridentIntegrandBase(vegas.BatchIntegrand):
    """
    Shared bookkeeping for all trident integrands.

    Subclasses set ``n_phase`` (number of phase-space dimensions) and
    ``Mn`` (target mass that sets the threshold), then implement
    ``_dsigma(x_phase, Enu, Enu_phase)`` returning ``(dsigma_cm2, mask)``.
    When ``Enu`` is ``None`` the last unit coordinate maps linearly onto
    ``[Emin, Emax]`` and the result is folded with ``flux`` (if given).
    """

    n_phase = 8

    def _init_common(self, nu_alpha, l1, l2, model, ml1, ml2,
                     Enu, Emin, Emax, flux, mode):
        self.nu_alpha = nu_alpha
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.ml1 = ml1
        self.ml2 = ml2
        self.fixed_Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        self.mode = mode
        self.bsm_mode = getattr(model, "bsm_mode", const.SM_ONLY)
        self.mzprime = getattr(model, "mzprime", 0.0)
        self.ndim = self.n_phase if Enu is not None else self.n_phase + 1

    def __call__(self, xx):
        xx = np.asarray(xx)
        nbatch = xx.shape[0]

        if self.fixed_Enu is not None:
            Enu = np.full(nbatch, self.fixed_Enu)
            x_phase = xx
        else:
            Enu = (self.Emax - self.Emin) * xx[:, self.n_phase] + self.Emin
            x_phase = xx[:, :self.n_phase]

        lab_threshold = trident_threshold(self.ml1, self.ml2, self.Mn)
        energy_valid = Enu > lab_threshold
        if not np.any(energy_valid):
            return np.zeros((nbatch, 1))
        Enu_phase = np.where(energy_valid, Enu, lab_threshold * (1.0 + 1e-9))

        out = self._dsigma(x_phase, Enu, Enu_phase)
        if out is None:
            return np.zeros((nbatch, 1))
        result, mask = out
        result = np.where(energy_valid & mask & np.isfinite(result), result, 0.0)

        if self.fixed_Enu is None:
            result = result * (self.Emax - self.Emin)
            if self.flux is not None:
                result = result * self.flux(Enu)

        return result.reshape(nbatch, 1)

    def _elastic_ps(self, x_phase, Enu_phase):
        return map_unit_to_physical(
            x_phase,
            Enu=Enu_phase,
            ml1=self.ml1,
            ml2=self.ml2,
            Mn=self.Mn,
            mzprime=self.mzprime,
            bsm_mode=self.bsm_mode,
        )

    def _tl(self, ps, Enu, regime, **hadronic):
        coup = self.model.get_coupling_terms(m6=ps["m6"])
        dsigma = differential_cross_section_tl(
            ps["x1"], ps["x2"], ps["x3"], ps["x4"], ps["x5"], ps["x6"],
            Enu, self.Mn, self.ml1, self.ml2,
            coup["V2"], coup["A2"], coup["VA"],
            regime=regime,
            mode=self.mode,
            **hadronic,
        )
        return dsigma * ps["Jacob"] * GeV2_to_cm2


class CoherentTridentIntegrand(_TridentIntegrandBase):
    """
    Coherent trident integrand on the 8-D x1..x6+x7+x8 phase space.

    Uses the Woods-Saxon nuclear form factor (or a user-supplied callable
    ``f(Q²)``) in the coherent hadronic flux ``h_T_coh`` / ``h_L_coh``.
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
        self._init_common(nu_alpha, l1, l2, model, ml1, ml2,
                          Enu, Emin, Emax, flux, mode)
        self.Z = Z
        self.A = A
        self.Mn = Mn if Mn is not None else float(A) * const.m_AVG
        self.form_factor = form_factor

    def _dsigma(self, x_phase, Enu, Enu_phase):
        ps = self._elastic_ps(x_phase, Enu_phase)
        if self.form_factor is None:
            FF = FF_WS_Q2(ps["x1"], self.A)
        else:
            FF = self.form_factor(ps["x1"])
        result = self._tl(ps, Enu, "coherent", Z=self.Z, FormFactor=FF)
        return result, ps["valid"]


class DiffractiveTridentIntegrand(_TridentIntegrandBase):
    """
    Diffractive (per-nucleon) trident integrand on the 8-D phase space.

    Uses the nucleon dipole form-factor combinations ``HH1``, ``HH2`` in
    the diffractive hadronic flux ``h_T_dif`` / ``h_L_dif``. Pauli blocking
    follows ``nuclear_model``; if omitted, ``FermiGas(p_F=0.221)`` when
    ``apply_pauli_blocking`` is true and ``FreeNucleon()`` otherwise.
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
        nuclear_model=None,
        mode="full",
    ):
        self._init_common(nu_alpha, l1, l2, model, ml1, ml2,
                          Enu, Emin, Emax, flux, mode)
        if nucleon not in ("p", "proton", "n", "neutron"):
            raise ValueError(f"nucleon must be 'proton' or 'neutron'; got {nucleon!r}")
        self.nucleon = nucleon
        self._is_proton = nucleon in ("p", "proton")
        if Mn is None:
            self.Mn = const.m_proton if self._is_proton else const.m_neutron
        else:
            self.Mn = Mn
        self.apply_pauli_blocking = apply_pauli_blocking
        default = FermiGas(p_F=0.221) if apply_pauli_blocking else FreeNucleon()
        self.nuclear_model = _check_nuclear_model(nuclear_model, default)

    def _dsigma(self, x_phase, Enu, Enu_phase):
        ps = self._elastic_ps(x_phase, Enu_phase)
        q = np.sqrt(np.maximum(ps["x1"], 0.0))
        if self._is_proton:
            HH1, HH2 = H1_p(q), H2_p(q)
        else:
            HH1, HH2 = H1_n(q), H2_n(q)
        result = self._tl(ps, Enu, "diffractive", HH1=HH1, HH2=HH2)
        result = result * self.nuclear_model.pauli_factor(_qmag(ps["x1"], self.Mn))
        return result, ps["valid"]


class _InelasticTridentIntegrand(_TridentIntegrandBase):
    """Shared DIS / resonance logic on the 9-D ``x1..x6 + xB`` phase space."""

    n_phase = 9

    def _sf_domain(self):
        sf = self.structure_functions
        q2_min = max(float(self.Q2_min), float(getattr(sf, "q2_min", 0.0)))
        q2_max = self.Q2_max
        sf_q2_max = float(getattr(sf, "q2_max", np.inf))
        if np.isfinite(sf_q2_max):
            q2_max = sf_q2_max if q2_max is None else min(float(q2_max), sf_q2_max)
        xB_min = max(float(self.xB_min), float(getattr(sf, "x_min", 0.0)))
        return q2_min, q2_max, xB_min

    def _inelastic(self, x_phase, Enu, Enu_phase):
        q2_min, q2_max, xB_min = self._sf_domain()
        if q2_max is not None and q2_min >= q2_max:
            return None
        ps = map_dis_unit_to_trident_physical(
            x_phase,
            Enu=Enu_phase,
            ml1=self.ml1,
            ml2=self.ml2,
            Mn=self.Mn,
            Q2_min=q2_min,
            Q2_max=q2_max,
            W2_min=self.W2_min,
            xB_min=xB_min,
            mzprime=self.mzprime,
            bsm_mode=self.bsm_mode,
        )
        F1 = self.structure_functions.F1(ps["xB"], ps["x1"])
        F2 = self.structure_functions.F2(ps["xB"], ps["x1"])
        result = self._tl(ps, Enu, "dis", xB=ps["xB"], F1=F1, F2=F2)
        qmag = _qmag(ps["x1"], self.Mn, ps["xB"])
        result = result * self.nuclear_model.pauli_factor(qmag)
        return ps, result


class DISTridentIntegrand(_InelasticTridentIntegrand):
    """
    Inclusive DIS trident integrand on the 9-D ``x1..x6 + xB`` phase space.

    The supplied ``structure_functions`` object must provide ``F1(x, Q2)`` and
    ``F2(x, Q2)`` methods. If those structure functions are summed over the
    target nucleons, the result is a per-nucleus cross section. No Pauli
    blocking is applied unless ``nuclear_model`` is given.
    """

    def __init__(
        self,
        nu_alpha,
        l1,
        l2,
        model,
        ml1,
        ml2,
        structure_functions,
        Mn=None,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
        Q2_min=1.0,
        Q2_max=None,
        W2_min=4.0,
        xB_min=1e-6,
        nuclear_model=None,
        mode="full",
    ):
        if structure_functions is None:
            raise ValueError("DIS integrand requires a structure_functions object")
        self._init_common(nu_alpha, l1, l2, model, ml1, ml2,
                          Enu, Emin, Emax, flux, mode)
        self.Mn = const.m_AVG if Mn is None else Mn
        self.structure_functions = structure_functions
        self.nuclear_model = _check_nuclear_model(nuclear_model, FreeNucleon())
        self.Q2_min = Q2_min
        self.Q2_max = Q2_max
        self.W2_min = W2_min
        self.xB_min = xB_min

    def _dsigma(self, x_phase, Enu, Enu_phase):
        out = self._inelastic(x_phase, Enu, Enu_phase)
        if out is None:
            return None
        ps, result = out
        return result, ps["valid"]


class ResonantTridentIntegrand(_InelasticTridentIntegrand):
    """
    Δ(1232)-resonance trident integrand on the 9-D ``x1..x6 + xB`` phase
    space.

    Shares the DIS mapping and the ``regime='dis'`` h_T/h_L flux functions,
    since the EM N→Δ hadronic tensor decomposes into the same two structure
    functions ``F1, F2``. The physics sits in the structure functions: a
    Lalakulich–Paschos vector vertex times a relativistic Δ Breit–Wigner
    (see ``neptune.resonance``). The W² window keeps Vegas in the resonance
    region, and ``xB_min`` is raised since the Δ at GeV-scale Q² lives at
    ``xB`` of order 0.3–0.6.

    Parameters
    ----------
    structure_functions : object, optional
        Provider of ``F1(xB, Q2)`` and ``F2(xB, Q2)``. Defaults to
        ``DeltaResonanceStructureFunctions(A=1)`` (per nucleon).
    M_R, Gamma0 : float
        Δ pole mass and on-shell width [GeV]; used only to set the
        default ``W2_max`` window.
    W2_min : float, optional
        Lower W² cutoff. Defaults to the Nπ threshold ``(Mn + m_π)²``.
    W2_max : float, optional
        Upper W² cutoff. Defaults to ``(M_R + 5 Gamma0)²``, which keeps
        the Δ tail without leaking into the second-resonance region.
    """

    def __init__(
        self,
        nu_alpha,
        l1,
        l2,
        model,
        ml1,
        ml2,
        structure_functions=None,
        Mn=None,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
        M_R=None,
        Gamma0=None,
        Q2_min=0.0,
        Q2_max=None,
        W2_min=None,
        W2_max=None,
        xB_min=1e-3,
        nuclear_model=None,
        mode="full",
    ):
        from neptune.resonance import (
            DeltaResonanceStructureFunctions,
            GAMMA0_DEFAULT,
            M_PI,
            M_R_DEFAULT,
        )

        self._init_common(nu_alpha, l1, l2, model, ml1, ml2,
                          Enu, Emin, Emax, flux, mode)
        self.M_R = float(M_R_DEFAULT if M_R is None else M_R)
        self.Gamma0 = float(GAMMA0_DEFAULT if Gamma0 is None else Gamma0)
        if structure_functions is None:
            structure_functions = DeltaResonanceStructureFunctions(
                A=1.0, M_R=self.M_R, Gamma0=self.Gamma0,
            )
        self.structure_functions = structure_functions
        self.Mn = const.m_AVG if Mn is None else float(Mn)
        self.nuclear_model = _check_nuclear_model(nuclear_model, FreeNucleon())
        self.Q2_min = float(Q2_min)
        self.Q2_max = Q2_max
        self.W2_min = float((self.Mn + M_PI) ** 2 if W2_min is None else W2_min)
        self.W2_max = float((self.M_R + 5.0 * self.Gamma0) ** 2 if W2_max is None else W2_max)
        self.xB_min = float(xB_min)

    def _dsigma(self, x_phase, Enu, Enu_phase):
        out = self._inelastic(x_phase, Enu, Enu_phase)
        if out is None:
            return None
        ps, result = out
        # Resonance window: drop points above W2_max (suppressed BW tail).
        return result, ps["valid"] & (ps["W2"] <= self.W2_max)


class ElectronTridentIntegrand(_TridentIntegrandBase):
    """
    Trident production on a free electron target on the same 8-D
    ``x1..x6 + x7, x8`` phase space used by the nucleon-target integrands.

    The hadronic flux uses ``regime='electron'`` (point-like electron).
    Returns a **per-electron** cross section; ``TridentProcess.sigma_electron``
    multiplies by ``Z``. No Pauli blocking is applied. For ``l1 = e`` the
    exchange diagrams with the target electron are not included.
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
        me=None,
        mode="full",
    ):
        self._init_common(nu_alpha, l1, l2, model, ml1, ml2,
                          Enu, Emin, Emax, flux, mode)
        self.me = float(const.m_e if me is None else me)
        # The target electron plays the role of "Mn" in the kinematics.
        self.Mn = self.me

    def _dsigma(self, x_phase, Enu, Enu_phase):
        ps = self._elastic_ps(x_phase, Enu_phase)
        result = self._tl(ps, Enu, "electron", me=self.me)
        return result, ps["valid"]
