"""
processes.py
============
High-level cross section calculator for neutrino trident production.

Wraps the T/L-decomposition integrands (`CoherentTridentIntegrand`,
`DiffractiveTridentIntegrand`) in Vegas integrators to compute total and
per-regime cross sections. Both regimes share the same 8-D x1..x6+x7+x8
phase space and only differ in the choice of hadronic flux ``h_T``,
``h_L``; the leptonic ``σT``, ``σL`` are identical.

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

from neptune.integrands import (
    CoherentTridentIntegrand,
    DiffractiveTridentIntegrand,
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

        ν_α + N      → ν' + ℓ⁻ + ℓ⁺ + N'         (diffractive, per nucleon)
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
        (default) or any callable ``f(Q²)``.
    nuclear_target : str or None
        Optional nucleus name passed to ``get_form_factor``.
    nitn : int
        Number of Vegas training iterations.
    neval : int
        Number of Vegas function evaluations per iteration.
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
        nuclear_target=None,
        nitn=10,
        neval=10_000,
    ):
        from neptune.nuclear_tools import get_form_factor

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
        self.nuclear_target = nuclear_target
        self._form_factor = get_form_factor(form_factor, Z, A,
                                            nuclear_target=nuclear_target)
        self.nitn = nitn
        self.neval = neval

        self.ml1 = model.ml1
        self.ml2 = model.ml2

        self._diffractive_result = None
        self._coherent_result = None

    def _resolve_mode(self, mode):
        return self.mode if mode is None else _normalise_mode(mode)

    def _make_diffractive_integrand(self, mode=None):
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
            nucleon="proton",
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

    def sigma_diffractive(self, nitn=None, neval=None, verbose=False, mode=None):
        """Compute the diffractive cross section [cm² per nucleon]."""
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_diffractive_integrand(mode=mode)
        integ = vegas.Integrator(f.ndim * [[0, 1]])
        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("Diffractive:", result.summary())

        self._diffractive_result = result
        r = result[0]
        return float(r.mean), float(r.sdev)

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

    def sigma_total(self, nitn=None, neval=None, verbose=False, mode=None):
        """
        Compute the total cross section (per nucleus) = Z·σ_diff + σ_coh.

        Returns
        -------
        dict with keys
            'diffractive'       : (mean, sdev) [cm² per proton]
            'coherent'          : (mean, sdev) [cm² per nucleus]
            'total_per_nucleus' : (mean, sdev) [cm² per nucleus]

        ``sigma_diffractive`` uses the proton dipole form factor and
        therefore returns the per-proton (nucleon-elastic) cross section.
        The per-nucleus diffractive rate is approximated by ``Z · σ_diff``,
        ignoring the sub-dominant neutron magnetic contribution.
        """
        dif_mean, dif_sdev = self.sigma_diffractive(
            nitn=nitn, neval=neval, verbose=verbose, mode=mode,
        )
        coh_mean, coh_sdev = self.sigma_coherent(
            nitn=nitn, neval=neval, verbose=verbose, mode=mode,
        )

        dif_total = dif_mean * self.Z
        dif_total_sdev = dif_sdev * self.Z

        total_mean = dif_total + coh_mean
        total_sdev = np.sqrt(dif_total_sdev**2 + coh_sdev**2)

        return {
            "diffractive": (dif_mean, dif_sdev),
            "coherent": (coh_mean, coh_sdev),
            "total_per_nucleus": (total_mean, total_sdev),
        }

    def sigma_scan(self, Enu_arr, nitn=None, neval=None, verbose=False, mode=None):
        """Compute cross sections at multiple neutrino energies."""
        Enu_arr = np.asarray(Enu_arr)
        dif = np.zeros(len(Enu_arr))
        dif_err = np.zeros(len(Enu_arr))
        coh = np.zeros(len(Enu_arr))
        coh_err = np.zeros(len(Enu_arr))

        for i, E in enumerate(Enu_arr):
            old_Enu = self.Enu
            self.Enu = E
            dif[i], dif_err[i] = self.sigma_diffractive(
                nitn=nitn, neval=neval, verbose=verbose, mode=mode,
            )
            coh[i], coh_err[i] = self.sigma_coherent(
                nitn=nitn, neval=neval, verbose=verbose, mode=mode,
            )
            self.Enu = old_Enu

        return {
            "Enu": Enu_arr,
            "diffractive": dif,
            "diffractive_err": dif_err,
            "coherent": coh,
            "coherent_err": coh_err,
        }
