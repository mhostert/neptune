"""
processes.py
============
High-level cross section calculator for neutrino trident production.

Provides the TridentProcess class which wraps the integrands in Vegas
integrators to compute total and differential cross sections.

Example
-------
>>> from neptune.processes import TridentProcess
>>> from neptune.model import TridentSMModel
>>> model = TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')
>>> proc = TridentProcess(model, Z=13, A=27, Enu=10.0)
>>> result = proc.sigma_total()
>>> print(result)  # cm^2 per nucleon (diffractive + coherent)
"""

import numpy as np
import vegas

from neptune.const import m_mu, m_e, m_tau
from neptune.model import TridentSMModel, TridentBSMModel
from neptune.integrands import (
    DiffractiveTridentIntegrand,
    CoherentTridentIntegrand,
    CoherentTridentIntegrandFull,
    Q_MAX_COH_DEFAULT,
    Q_MAX_DIFF_DEFAULT,
    Q_max_coh_for_A,
)


class TridentProcess:
    """
    Neutrino trident total cross section calculator.

    Computes the cross section for:
        nu_alpha + N → nu' + l1⁻ + l2⁺ + N' (diffractive, per nucleon)
        nu_alpha + nucleus → nu' + l1⁻ + l2⁺ + nucleus (coherent)

    Parameters
    ----------
    model : TridentSMModel or TridentBSMModel
        Physics model with couplings and lepton flavors.
    Z : int
        Atomic number of the target nucleus.
    A : int
        Mass number of the target nucleus.
    Enu : float or None
        Fixed neutrino energy [GeV]. If None, must supply flux for
        flux-averaged cross sections.
    Emin, Emax : float
        Energy range for flux-averaged calculations [GeV].
    flux : callable or None
        Neutrino flux dN/dE [arbitrary units]. Called as flux(Enu).
    Q_max_coh : float
        Upper Q limit for coherent / EPA-coherent integration.
    Q_max_diff : float
        Upper Q limit for diffractive integration.
    use_epa : bool
        If True (default), use the equivalent-photon approximation for the
        coherent regime — fast, 6-D Vegas integration that factorises the
        nuclear photon flux from the lepton-pair production amplitude.
        If False, use the full 8-D coherent matrix element with no EPA
        factorisation (slower; in principle exact but currently
        **experimental** — overall normalisation of the auto-translated
        polynomial is still being validated against the C++ reference).
    Mn : float, optional
        Target mass for the full (non-EPA) coherent integration [GeV].
        Defaults to A * m_AVG.  Ignored when use_epa=True.
    nitn : int
        Number of Vegas training iterations.
    neval : int
        Number of function evaluations per Vegas iteration.
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
        Q_max_coh=None,
        Q_max_diff=Q_MAX_DIFF_DEFAULT,
        use_epa=True,
        Mn=None,
        nitn=10,
        neval=10_000,
    ):
        self.model = model
        self.Z = Z
        self.A = A
        self.Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        # A-dependent coherent/diffractive boundary (Λ_QCD / A^(1/3))
        self.Q_max_coh = Q_max_coh_for_A(A) if Q_max_coh is None else Q_max_coh
        self.Q_max_diff = Q_max_diff
        self.use_epa = use_epa
        self.Mn = Mn
        self.nitn = nitn
        self.neval = neval

        self.ml1 = model.ml1
        self.ml2 = model.ml2

        self._diffractive_result = None
        self._coherent_result = None

    def _make_diffractive_integrand(self):
        return DiffractiveTridentIntegrand(
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
            Q_max_coh=self.Q_max_coh,
            Q_max_diff=self.Q_max_diff,
        )

    def _make_coherent_integrand(self):
        if self.use_epa:
            return CoherentTridentIntegrand(
                self.model.nu_flavor,
                self.model.l1_flavor,
                self.model.l2_flavor,
                self.model,
                self.ml1,
                self.ml2,
                self.Z,
                self.A,
                Enu=self.Enu,
                Emin=self.Emin,
                Emax=self.Emax,
                flux=self.flux,
                Q_max_coh=self.Q_max_coh,
            )
        return CoherentTridentIntegrandFull(
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
        )

    def sigma_diffractive(self, nitn=None, neval=None, verbose=False):
        """
        Compute the diffractive trident cross section [cm^2 per nucleon].

        Uses Vegas to integrate over the 6-dim (or 7-dim with flux) phase space.

        Parameters
        ----------
        nitn : int, optional
            Vegas training iterations (overrides self.nitn).
        neval : int, optional
            Vegas evaluations per iteration (overrides self.neval).
        verbose : bool
            If True, print Vegas summary.

        Returns
        -------
        (mean, sdev) : tuple of float
            Cross section mean and standard deviation [cm^2].
        """
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_diffractive_integrand()
        ndim = f.ndim
        integ = vegas.Integrator(ndim * [[0, 1]])

        # Training
        integ(f, nitn=nitn // 2, neval=neval)
        # Final result
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("Diffractive:", result.summary())

        self._diffractive_result = result
        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_coherent(self, nitn=None, neval=None, verbose=False):
        """
        Compute the coherent nuclear trident cross section [cm^2 per nucleus].

        Parameters
        ----------
        nitn : int, optional
        neval : int, optional
        verbose : bool

        Returns
        -------
        (mean, sdev) : tuple of float
            Cross section mean and standard deviation [cm^2].
        """
        nitn = nitn or self.nitn
        neval = neval or self.neval

        f = self._make_coherent_integrand()
        ndim = f.ndim
        integ = vegas.Integrator(ndim * [[0, 1]])

        integ(f, nitn=nitn // 2, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        if verbose:
            print("Coherent:", result.summary())

        self._coherent_result = result
        r = result[0]
        return float(r.mean), float(r.sdev)

    def sigma_total(self, nitn=None, neval=None, verbose=False):
        """
        Compute total trident cross section (diffractive + coherent) [cm^2].

        ``sigma_diffractive`` uses the **proton** dipole form factor and
        therefore returns the per-proton (nucleon-elastic) cross section.
        The per-nucleus diffractive rate is approximated by ``Z * σ_diff``,
        ignoring the sub-dominant neutron magnetic contribution. Coherent
        is per-nucleus already.

        Returns
        -------
        dict with keys:
            'diffractive'       : (mean, sdev) [cm^2 per proton]
            'coherent'          : (mean, sdev) [cm^2 per nucleus]
            'total_per_nucleus' : (mean, sdev) [cm^2 per nucleus]
        """
        dif_mean, dif_sdev = self.sigma_diffractive(
            nitn=nitn, neval=neval, verbose=verbose
        )
        coh_mean, coh_sdev = self.sigma_coherent(
            nitn=nitn, neval=neval, verbose=verbose
        )

        # diffractive is per proton; scale to per nucleus by Z
        dif_total = dif_mean * self.Z
        dif_total_sdev = dif_sdev * self.Z

        total_mean = dif_total + coh_mean
        total_sdev = np.sqrt(dif_total_sdev**2 + coh_sdev**2)

        return {
            "diffractive": (dif_mean, dif_sdev),
            "coherent": (coh_mean, coh_sdev),
            "total_per_nucleus": (total_mean, total_sdev),
        }

    def sigma_scan(self, Enu_arr, nitn=None, neval=None, verbose=False):
        """
        Compute cross sections at multiple neutrino energies.

        Parameters
        ----------
        Enu_arr : array-like
            Neutrino energies [GeV].
        nitn, neval : int, optional
        verbose : bool

        Returns
        -------
        dict with keys 'Enu', 'diffractive', 'diffractive_err',
                       'coherent', 'coherent_err' [cm^2]
        """
        Enu_arr = np.asarray(Enu_arr)
        dif = np.zeros(len(Enu_arr))
        dif_err = np.zeros(len(Enu_arr))
        coh = np.zeros(len(Enu_arr))
        coh_err = np.zeros(len(Enu_arr))

        for i, E in enumerate(Enu_arr):
            old_Enu = self.Enu
            self.Enu = E
            dif[i], dif_err[i] = self.sigma_diffractive(
                nitn=nitn, neval=neval, verbose=verbose
            )
            coh[i], coh_err[i] = self.sigma_coherent(
                nitn=nitn, neval=neval, verbose=verbose
            )
            self.Enu = old_Enu

        return {
            "Enu": Enu_arr,
            "diffractive": dif,
            "diffractive_err": dif_err,
            "coherent": coh,
            "coherent_err": coh_err,
        }
