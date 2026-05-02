"""
MC.py
=====
Monte Carlo event generator for neutrino trident events.

Generates weighted phase-space samples distributed according to the
T/L-decomposition differential cross section using Vegas importance
sampling.

Example
-------
>>> from neptune.MC import TridentGenerator
>>> from neptune.model import TridentSMModel
>>> model = TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')
>>> gen = TridentGenerator(model, Z=18, A=40, Enu=10.0, n_events=1000)
>>> events = gen.generate()
>>> events.keys()
dict_keys(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'm6', 'Enu', 'weight', 'mode'])
"""

import numpy as np
import vegas

from neptune.integrands import (
    CoherentTridentIntegrand,
    DiffractiveTridentIntegrand,
)
from neptune.phase_space import map_unit_to_physical
from neptune.processes import _normalise_mode


class TridentGenerator:
    """
    Monte Carlo event generator for neutrino trident production.

    Trains Vegas importance-sampling maps for both diffractive and coherent
    integrands, then samples weighted events from those maps. Output is in
    the BSM_trident.nb x1..x6 invariant phase space.

    Parameters
    ----------
    model : TridentSMModel or TridentBSMModel
    Z, A : int
        Target nucleus.
    Enu : float or None
        Fixed neutrino energy [GeV]. If None, use ``flux``.
    Emin, Emax : float
        Energy range for flux-averaged generation [GeV].
    flux : callable or None
        Flux function dN/dE.
    mode : {'full', 'improved-epa', 'epa'}
        T/L combination passed through to the integrands. Default 'full'.
    Mn : float, optional
        Target nucleus mass [GeV]; defaults to A * m_AVG.
    form_factor : str or callable
        Coherent nuclear form-factor specification.
    nuclear_target : str or None
        Optional nucleus name for the form-factor lookup.
    n_events : int
        Target number of weighted events to generate.
    nitn : int
        Vegas training iterations.
    neval : int
        Vegas evaluations per iteration (controls map resolution).
    seed : int or None
        Random seed.
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
        n_events=10_000,
        nitn=10,
        neval=50_000,
        seed=None,
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
        self.n_events = n_events
        self.nitn = nitn
        self.neval = neval
        self.seed = seed

        self._dif_integrand = None
        self._coh_integrand = None
        self._dif_map = None
        self._coh_map = None

    def _make_diffractive_integrand(self):
        return DiffractiveTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.model.ml1,
            self.model.ml2,
            Mn=self.Mn,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            nucleon="proton",
            mode=self.mode,
        )

    def _make_coherent_integrand(self):
        return CoherentTridentIntegrand(
            self.model.nu_flavor,
            self.model.l1_flavor,
            self.model.l2_flavor,
            self.model,
            self.model.ml1,
            self.model.ml2,
            self.Z,
            self.A,
            Mn=self.Mn,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
            form_factor=self._form_factor,
            mode=self.mode,
        )

    def train(self, verbose=False):
        """Train Vegas importance-sampling maps. Call before ``generate()``."""
        dif_f = self._make_diffractive_integrand()
        coh_f = self._make_coherent_integrand()

        self._dif_integrand = dif_f
        self._dif_map = vegas.Integrator(dif_f.ndim * [[0, 1]])
        self._dif_map(dif_f, nitn=self.nitn, neval=self.neval, adapt=True)
        if verbose:
            print("Diffractive map trained.")

        self._coh_integrand = coh_f
        self._coh_map = vegas.Integrator(coh_f.ndim * [[0, 1]])
        self._coh_map(coh_f, nitn=self.nitn, neval=self.neval, adapt=True)
        if verbose:
            print("Coherent map trained.")

    def _generate_from_map(self, integrand, integrator, n_events, regime):
        """
        Sample weighted events from a trained Vegas map and convert to the
        physical x1..x6+m6 representation.
        """
        target_raw = max(n_events * 20, 200_000)

        all_x = []
        all_w = []
        total = 0
        for x_batch, w_batch in integrator.random_batch():
            all_x.append(x_batch)
            all_w.append(w_batch)
            total += len(x_batch)
            if total >= target_raw:
                break

        xx = np.concatenate(all_x, axis=0)[:target_raw]
        ww = np.concatenate(all_w, axis=0)[:target_raw]

        fval = integrand(xx).flatten()
        weights = fval * ww
        pos = weights > 0
        xx = xx[pos]
        weights = weights[pos]

        if integrand.fixed_Enu is not None:
            Enu_arr = np.full(len(xx), integrand.fixed_Enu)
            x_phase = xx
        else:
            Enu_arr = (integrand.Emax - integrand.Emin) * xx[:, 8] + integrand.Emin
            x_phase = xx[:, :8]

        ps = map_unit_to_physical(
            x_phase,
            Enu=Enu_arr,
            ml1=integrand.ml1,
            ml2=integrand.ml2,
            Mn=integrand.Mn,
            mzprime=integrand.mzprime,
            bsm_mode=integrand.bsm_mode,
        )
        return {
            "x1": ps["x1"], "x2": ps["x2"], "x3": ps["x3"],
            "x4": ps["x4"], "x5": ps["x5"], "x6": ps["x6"],
            "m6": ps["m6"],
            "Enu": Enu_arr,
            "weight": weights,
            "mode": np.full(len(xx), regime),
        }

    def generate(self, verbose=False):
        """
        Generate weighted trident events from both regimes.

        If the Vegas maps have not been trained, ``train()`` is called first.

        Returns
        -------
        dict
            Keys: ``x1, x2, x3, x4, x5, x6, m6, Enu, weight, mode``.
            All values are 1-D numpy arrays.
        """
        if self._dif_map is None or self._coh_map is None:
            self.train(verbose=verbose)

        n_half = self.n_events // 2

        dif_events = self._generate_from_map(
            self._dif_integrand, self._dif_map, n_half, "diffractive"
        )
        coh_events = self._generate_from_map(
            self._coh_integrand, self._coh_map, n_half, "coherent"
        )

        keys = ["x1", "x2", "x3", "x4", "x5", "x6", "m6", "Enu", "weight", "mode"]
        return {k: np.concatenate([dif_events[k], coh_events[k]]) for k in keys}
