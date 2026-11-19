"""
processes.py
============
High-level cross section calculator for neutrino trident production.

Wraps the T/L-decomposition integrands (`CoherentTridentIntegrand`,
`DiffractiveTridentIntegrand`) in Vegas integrators to compute total and
per-regime cross sections. Once a target ``Z, A`` is specified, public
cross-section methods return per-target cross sections. The diffractive
piece is the incoherent sum over protons and neutrons in the target.

The single ``mode`` parameter selects:

* ``'full'``         — keep both T and L contributions with full x1 dep.
* ``'improved-epa'`` — drop ``h_L · σL``; keep ``σT(x1, ...)``.
* ``'epa'``          — drop ``h_L · σL``; evaluate ``σT`` at q² = 0.

Example
-------
>>> from neptune.processes import TridentProcess
>>> from neptune.model import TridentSMModel
>>> model = TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')
>>> proc = TridentProcess(model, Z=18, A=40, Enu=10.0)
>>> result = proc.sigma_total()
>>> print(result)
"""

import numpy as np
import vegas

from neptune import const
from neptune.integrands import (
    CoherentTridentIntegrand,
    DiffractiveTridentIntegrand,
    DISTridentIntegrand,
    ElectronTridentIntegrand,
    ResonantTridentIntegrand,
)


_VALID_MODES = ("full", "improved-epa", "epa")


def _normalise_mode(mode):
    m = mode.lower().replace("_", "-")
    aliases = {"iepa": "improved-epa", "improved_epa": "improved-epa"}
    m = aliases.get(m, m)
    if m not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
    return m


class TridentProcess:
    """
    Neutrino trident total cross section calculator.

    Computes the cross section for::

        ν_α + N      → ν' + ℓ⁻ + ℓ⁺ + N'         (diffractive, summed over nucleons)
        ν_α + nucleus → ν' + ℓ⁻ + ℓ⁺ + nucleus    (coherent)

    Parameters
    ----------
    model : TridentSMModel or TridentBSMModel
        Physics model with couplings and lepton flavors.
    Z, A : int
        Atomic and mass number of the target nucleus.
    Enu : float or None
        Fixed neutrino energy [GeV]. If None, supply ``flux``.
    Emin, Emax : float
        Energy range for flux-averaged calculations [GeV].
    flux : callable or None
        Neutrino flux dN/dE. Called as ``flux(Enu)``.
    mode : {'full', 'improved-epa', 'epa'}
        Which T/L combination to use; default ``'full'``. See module
        docstring.
    Mn : float, optional
        Target mass for the coherent integrand [GeV]. Defaults to
        ``A * m_AVG``.
    form_factor : str or callable
        Coherent nuclear form factor specification. ``'woods-saxon'``
        (default), ``'darknews'``, or any callable ``f(Q²)``.
    atomic_screening : str or None
        Atomic screening of the coherent form factor (``'tsai'``,
        ``'moliere'``). Default ``None`` (bare nucleus).
    nuclear_target : str or None
        Optional nucleus name passed to ``get_form_factor``.
    nuclear_model : neptune.nuclear_models instance or None
        Pauli-blocking model for the diffractive, DIS, and resonance
        integrands. ``None`` keeps each integrand's default.
    nitn : int
        Number of Vegas training iterations.
    neval : int
        Number of Vegas function evaluations per iteration.
    dis_structure_functions : object, optional
        Object with ``F1(x, Q2)`` and ``F2(x, Q2)`` methods for DIS.
    dis_pdf_set : str, optional
        LHAPDF set name used lazily to build DIS structure functions.
    dis_Q2_min, dis_W2_min : float
        DIS cuts [GeV²]; defaults Q² > 1 and W² > 4.
    resonance_* : optional
        Overrides for the Δ(1232) structure functions and kinematic window.
    """

    def __init__(
        self,
        model,
        Z,
        A,
        Enu=None,
        Emin=0.0,
        Emax=100.0,
        flux=None,
        mode="full",
        Mn=None,
        form_factor="woods-saxon",
        atomic_screening=None,
        nuclear_target=None,
        dis_structure_functions=None,
        dis_pdf_set=None,
        dis_pdf_member=0,
        dis_pdf_mode="isospin",
        dis_Q2_min=1.0,
        dis_Q2_max=None,
        dis_W2_min=4.0,
        dis_xB_min=1e-6,
        resonance_structure_functions=None,
        resonance_M_R=None,
        resonance_Gamma0=None,
        resonance_W2_max=None,
        resonance_Q2_min=0.0,
        resonance_Q2_max=None,
        resonance_xB_min=1e-3,
        nuclear_model=None,
        nitn=10,
        neval=10_000,
    ):
        from neptune.nuclear_tools import apply_atomic_screening, get_form_factor

        self.model = model
        self.Z = Z
        self.A = A
        self.Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        self.mode = _normalise_mode(mode)
        self.Mn = Mn
        self.form_factor_spec = form_factor
        self.atomic_screening = atomic_screening
        self.nuclear_target = nuclear_target
        self._form_factor = get_form_factor(form_factor, Z, A,
                                            nuclear_target=nuclear_target)
        # Optionally wrap the bare nuclear form factor with the atomic
        # screening modulator G_at(Q²); leaves it untouched if None.
        if atomic_screening is not None:
            self._form_factor = apply_atomic_screening(
                self._form_factor, Z, kind=atomic_screening,
            )
        self.dis_structure_functions = dis_structure_functions
        self.dis_pdf_set = dis_pdf_set
        self.dis_pdf_member = dis_pdf_member
        self.dis_pdf_mode = dis_pdf_mode
        self.dis_Q2_min = dis_Q2_min
        self.dis_Q2_max = dis_Q2_max
        self.dis_W2_min = dis_W2_min
        self.dis_xB_min = dis_xB_min
        self.resonance_structure_functions = resonance_structure_functions
        self.resonance_M_R = resonance_M_R
        self.resonance_Gamma0 = resonance_Gamma0
        self.resonance_W2_max = resonance_W2_max
        self.resonance_Q2_min = resonance_Q2_min
        self.resonance_Q2_max = resonance_Q2_max
        self.resonance_xB_min = resonance_xB_min
        self.nuclear_model = nuclear_model
        self.nitn = nitn
        self.neval = neval

        self.ml1 = model.ml1
        self.ml2 = model.ml2

        self._diffractive_result = None
        self._coherent_result = None
        self._dis_result = None
        self._resonance_result = None
        self._electron_result = None

    def _resolve_mode(self, mode):
        return self.mode if mode is None else _normalise_mode(mode)

    def _resolve_nuclear_model(self, nuclear_model):
        """Per-call override falls back to the constructor default."""
        return self.nuclear_model if nuclear_model is None else nuclear_model

    def _make_diffractive_integrand(self, mode=None, nucleon="proton",
                                    nuclear_model=None):
        return DiffractiveTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.ml1,
            self.ml2,
            Mn=self.Mn,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            nucleon=nucleon,
            nuclear_model=self._resolve_nuclear_model(nuclear_model),
            mode=self._resolve_mode(mode),
        )

    def _make_coherent_integrand(self, mode=None):
        return CoherentTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.ml1,
            self.ml2,
            self.Z,
            self.A,
            Mn=self.Mn,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            form_factor=self._form_factor,
            mode=self._resolve_mode(mode),
        )

    def _resolve_dis_structure_functions(self, structure_functions=None):
        if structure_functions is not None:
            return structure_functions
        if self.dis_structure_functions is not None:
            return self.dis_structure_functions
        if self.dis_pdf_set is None:
            raise ValueError(
                "DIS requires either dis_structure_functions or dis_pdf_set. "
                "Use neptune.dis.DISStructureFunctions with an LHAPDF set."
            )
        from neptune.dis import make_lhapdf_structure_functions

        self.dis_structure_functions = make_lhapdf_structure_functions(
            self.dis_pdf_set,
            member=self.dis_pdf_member,
            Z=self.Z,
            A=self.A,
            mode=self.dis_pdf_mode,
        )
        return self.dis_structure_functions

    def _make_dis_integrand(self, mode=None, structure_functions=None,
                            nuclear_model=None):
        return DISTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.ml1,
            self.ml2,
            self._resolve_dis_structure_functions(structure_functions),
            Mn=const.m_AVG,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            Q2_min=self.dis_Q2_min,
            Q2_max=self.dis_Q2_max,
            W2_min=self.dis_W2_min,
            xB_min=self.dis_xB_min,
            nuclear_model=self._resolve_nuclear_model(nuclear_model),
            mode=self._resolve_mode(mode),
        )

    def _resolve_resonance_structure_functions(self, structure_functions=None):
        if structure_functions is not None:
            return structure_functions
        if self.resonance_structure_functions is not None:
            return self.resonance_structure_functions
        from neptune.resonance import (
            DeltaResonanceStructureFunctions,
            GAMMA0_DEFAULT,
            M_R_DEFAULT,
        )

        self.resonance_structure_functions = DeltaResonanceStructureFunctions(
            A=float(self.A),
            M_R=M_R_DEFAULT if self.resonance_M_R is None else self.resonance_M_R,
            Gamma0=GAMMA0_DEFAULT if self.resonance_Gamma0 is None else self.resonance_Gamma0,
        )
        return self.resonance_structure_functions

    def _make_electron_integrand(self, mode=None):
        return ElectronTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.ml1,
            self.ml2,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            mode=self._resolve_mode(mode),
        )

    def _make_resonance_integrand(self, mode=None, structure_functions=None,
                                  nuclear_model=None):
        return ResonantTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.ml1,
            self.ml2,
            structure_functions=self._resolve_resonance_structure_functions(structure_functions),
            Mn=const.m_AVG,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            M_R=self.resonance_M_R,
            Gamma0=self.resonance_Gamma0,
            Q2_min=self.resonance_Q2_min,
            Q2_max=self.resonance_Q2_max,
            W2_max=self.resonance_W2_max,
            xB_min=self.resonance_xB_min,
            nuclear_model=self._resolve_nuclear_model(nuclear_model),
            mode=self._resolve_mode(mode),
        )

    def sigma_diffractive_nucleon(
        self, nucleon="proton", nitn=None, neval=None, verbose=False, mode=None,
        nuclear_model=None,
    ):
        """Compute the single-nucleon diffractive cross section [cm²]."""
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_diffractive_integrand(mode=mode, nucleon=nucleon,
                                              nuclear_model=nuclear_model)
        integ = vegas.Integrator(f.ndim * [[0, 1]])
        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print(f"Diffractive ({nucleon}):", result.summary())

        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_diffractive(self, nitn=None, neval=None, verbose=False, mode=None,
                          nuclear_model=None):
        """
        Compute the target diffractive cross section [cm² per target].

        This is the incoherent nucleon-elastic sum

            Z * sigma_p + (A - Z) * sigma_n

        using proton and neutron electromagnetic form factors.
        """
        Z = int(self.Z)
        N = max(int(self.A) - Z, 0)

        mean = 0.0
        sdev2 = 0.0
        results = {}

        if Z:
            p_mean, p_sdev = self.sigma_diffractive_nucleon(
                "proton", nitn=nitn, neval=neval, verbose=verbose, mode=mode,
                nuclear_model=nuclear_model,
            )
            mean += Z * p_mean
            sdev2 += (Z * p_sdev) ** 2
            results["proton"] = (p_mean, p_sdev)

        if N:
            n_mean, n_sdev = self.sigma_diffractive_nucleon(
                "neutron", nitn=nitn, neval=neval, verbose=verbose, mode=mode,
                nuclear_model=nuclear_model,
            )
            mean += N * n_mean
            sdev2 += (N * n_sdev) ** 2
            results["neutron"] = (n_mean, n_sdev)

        self._diffractive_result = results
        return float(mean), float(np.sqrt(sdev2))

    def sigma_coherent(self, nitn=None, neval=None, verbose=False, mode=None):
        """Compute the coherent cross section [cm² per nucleus]."""
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_coherent_integrand(mode=mode)
        integ = vegas.Integrator(f.ndim * [[0, 1]])
        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("Coherent:", result.summary())

        self._coherent_result = result
        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_dis(
        self,
        nitn=None,
        neval=None,
        verbose=False,
        mode=None,
        structure_functions=None,
        nuclear_model=None,
    ):
        """
        Compute the inclusive DIS cross section [cm²].

        The normalization follows the supplied structure functions. The
        default LHAPDF construction uses ``Z * F_i^p + (A-Z) * F_i^n`` and
        therefore returns a per-target cross section.
        """
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_dis_integrand(
            mode=mode, structure_functions=structure_functions,
            nuclear_model=nuclear_model,
        )
        integ = vegas.Integrator(f.ndim * [[0, 1]])
        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("DIS:", result.summary())

        self._dis_result = result
        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_electron_per_e(self, nitn=None, neval=None, verbose=False, mode=None):
        """Per-electron trident cross section [cm²] (free-electron approximation)."""
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_electron_integrand(mode=mode)
        integ = vegas.Integrator(f.ndim * [[0, 1]])
        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("Electron (per e):", result.summary())

        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_electron(self, nitn=None, neval=None, verbose=False, mode=None):
        """
        Trident cross section on the **Z atomic electrons** of the target
        nucleus [cm² per atom].

        Free-electron (impulse-approximation) treatment: the bound electrons
        are treated as a free Fermi gas of identical scatterers, with the
        per-atom rate equal to ``Z`` times the per-electron rate. Atomic
        binding (~ keV–MeV) is neglected — small for trident at GeV-scale
        ``E_ν``. No Pauli blocking is applied.
        """
        s_per_e, sd_per_e = self.sigma_electron_per_e(
            nitn=nitn, neval=neval, verbose=verbose, mode=mode,
        )
        Z = float(self.Z)
        self._electron_result = (s_per_e, sd_per_e)
        return Z * s_per_e, Z * sd_per_e

    def sigma_resonance(
        self,
        nitn=None,
        neval=None,
        verbose=False,
        mode=None,
        structure_functions=None,
        nuclear_model=None,
    ):
        """
        Compute the Δ(1232)-resonance cross section [cm² per target].

        Uses ``ResonantTridentIntegrand`` with default
        ``DeltaResonanceStructureFunctions(A=A)`` (per-nucleus, with the
        EM isospin factor 1/3 baked in). Supply ``structure_functions``
        to override (e.g. for higher resonances).
        """
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_resonance_integrand(
            mode=mode, structure_functions=structure_functions,
            nuclear_model=nuclear_model,
        )
        integ = vegas.Integrator(f.ndim * [[0, 1]])
        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("Resonance:", result.summary())

        self._resonance_result = result
        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_total(
        self,
        nitn=None,
        neval=None,
        verbose=False,
        mode=None,
        include_dis=False,
        include_resonance=False,
        include_electron=False,
        dis_structure_functions=None,
        resonance_structure_functions=None,
        nuclear_model=None,
    ):
        """
        Compute the total cross section (per nucleus).

        Returns
        -------
        dict with keys
            'diffractive'       : (mean, sdev) [cm² per target]
            'coherent'          : (mean, sdev) [cm² per target]
            'dis', 'resonance', 'electron' : (mean, sdev), only if requested
            'total_per_nucleus' : (mean, sdev) [cm² per target]
        """
        dif_mean, dif_sdev = self.sigma_diffractive(
            nitn=nitn, neval=neval, verbose=verbose, mode=mode,
            nuclear_model=nuclear_model,
        )
        coh_mean, coh_sdev = self.sigma_coherent(
            nitn=nitn, neval=neval, verbose=verbose, mode=mode,
        )

        total_mean = dif_mean + coh_mean
        total_sdev2 = dif_sdev**2 + coh_sdev**2

        out = {
            "diffractive": (dif_mean, dif_sdev),
            "coherent": (coh_mean, coh_sdev),
        }

        if include_dis:
            dis_mean, dis_sdev = self.sigma_dis(
                nitn=nitn,
                neval=neval,
                verbose=verbose,
                mode=mode,
                structure_functions=dis_structure_functions,
                nuclear_model=nuclear_model,
            )
            total_mean += dis_mean
            total_sdev2 += dis_sdev**2
            out["dis"] = (dis_mean, dis_sdev)

        if include_resonance:
            res_mean, res_sdev = self.sigma_resonance(
                nitn=nitn,
                neval=neval,
                verbose=verbose,
                mode=mode,
                structure_functions=resonance_structure_functions,
                nuclear_model=nuclear_model,
            )
            total_mean += res_mean
            total_sdev2 += res_sdev**2
            out["resonance"] = (res_mean, res_sdev)

        if include_electron:
            ele_mean, ele_sdev = self.sigma_electron(
                nitn=nitn, neval=neval, verbose=verbose, mode=mode,
            )
            total_mean += ele_mean
            total_sdev2 += ele_sdev**2
            out["electron"] = (ele_mean, ele_sdev)

        out["total_per_nucleus"] = (total_mean, np.sqrt(total_sdev2))
        return out

    def sigma_scan(self, Enu_arr, nitn=None, neval=None, verbose=False, mode=None):
        """Compute cross sections at multiple neutrino energies."""
        Enu_arr = np.asarray(Enu_arr)
        dif = np.zeros(len(Enu_arr))
        dif_err = np.zeros(len(Enu_arr))
        coh = np.zeros(len(Enu_arr))
        coh_err = np.zeros(len(Enu_arr))

        old_Enu = self.Enu
        try:
            for i, E in enumerate(Enu_arr):
                self.Enu = E
                dif[i], dif_err[i] = self.sigma_diffractive(
                    nitn=nitn, neval=neval, verbose=verbose, mode=mode,
                )
                coh[i], coh_err[i] = self.sigma_coherent(
                    nitn=nitn, neval=neval, verbose=verbose, mode=mode,
                )
        finally:
            self.Enu = old_Enu

        return {
            "Enu": Enu_arr,
            "diffractive": dif,
            "diffractive_err": dif_err,
            "coherent": coh,
            "coherent_err": coh_err,
        }
