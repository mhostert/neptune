"""
MC.py
=====
Monte Carlo event generator for neutrino trident events.

Generates weighted phase-space samples distributed according to the
T/L-decomposition differential cross section using Vegas importance
sampling. Weights are in cm² per target nucleus: the coherent weights sum to
``TridentProcess.sigma_coherent`` and the diffractive weights (protons and
neutrons) to ``TridentProcess.sigma_diffractive``.

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

    Trains Vegas importance-sampling maps for the coherent integrand and for
    the diffractive integrand on protons and on neutrons, then samples
    weighted events from those maps. Diffractive proton (neutron) weights
    are multiplied by Z (A - Z). Output is in the BSM_trident.nb x1..x6
    invariant phase space.

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
    atomic_screening : str or None
        Atomic screening of the coherent form factor ('tsai', 'moliere').
    nuclear_model : neptune.nuclear_models instance or None
        Pauli-blocking model for the diffractive integrand.
    nuclear_target : str or None
        Optional nucleus name for the form-factor lookup.
    n_events : int
        Minimum total number of raw samples, split evenly across regimes and
        rounded up to whole Vegas passes of ``neval`` points. Zero-weight
        samples are dropped, so fewer events may be returned.
    nitn : int
        Vegas training iterations.
    neval : int
        Vegas evaluations per iteration (controls map resolution).
    seed : int or None
        Random seed for the Vegas maps and sampling (reproducible output).
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
        nuclear_model=None,
        nuclear_target=None,
        n_events=10_000,
        nitn=10,
        neval=50_000,
        seed=None,
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
        self.nuclear_model = nuclear_model
        self.nuclear_target = nuclear_target
        self._form_factor = get_form_factor(form_factor, Z, A,
                                            nuclear_target=nuclear_target)
        if atomic_screening is not None:
            self._form_factor = apply_atomic_screening(
                self._form_factor, Z, kind=atomic_screening,
            )
        self.n_events = n_events
        self.nitn = nitn
        self.neval = neval
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # regime -> (integrand, trained vegas map, weight multiplier)
        self._maps = None

    def _regimes(self):
        """(label, integrand factory, multiplier) for each sampled regime."""
        Z = int(self.Z)
        N = max(int(self.A) - Z, 0)
        out = [("coherent", self._make_coherent_integrand, 1.0)]
        if Z:
            out.append(("diffractive",
                        lambda: self._make_diffractive_integrand("proton"), float(Z)))
        if N:
            out.append(("diffractive",
                        lambda: self._make_diffractive_integrand("neutron"), float(N)))
        return out

    def _make_diffractive_integrand(self, nucleon="proton"):
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
            nucleon=nucleon,
            nuclear_model=self.nuclear_model,
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

    def _integrator(self, ndim):
        return vegas.Integrator(ndim * [[0, 1]],
                                ran_array_generator=self._rng.random)

    def train(self, verbose=False):
        """Train Vegas importance-sampling maps. Call before ``generate()``."""
        self._maps = []
        for label, make, mult in self._regimes():
            f = make()
            integ = self._integrator(f.ndim)
            integ(f, nitn=self.nitn, neval=self.neval, adapt=True)
            self._maps.append((label, f, integ, mult))
            if verbose:
                print(f"{label} map trained ({getattr(f, 'nucleon', 'nucleus')}).")

    def _generate_from_map(self, integrand, integrator, n_events, regime):
        """
        Sample weighted events from a trained Vegas map and convert to the
        physical x1..x6+m6 representation.

        ``random_batch()`` covers one Vegas pass, whose weights ``wgt · f``
        sum to the integral. We draw whole passes until ``n_events``
        samples are collected and divide the weights by the number of passes.
        """
        target_raw = max(int(n_events), 1)

        all_x = []
        all_w = []
        total = 0
        n_pass = 0
        while total < target_raw:
            for x_batch, w_batch in integrator.random_batch():
                all_x.append(np.array(x_batch))
                all_w.append(np.array(w_batch))
                total += len(x_batch)
            n_pass += 1

        xx = np.concatenate(all_x, axis=0)
        ww = np.concatenate(all_w, axis=0) / n_pass

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
        Generate weighted trident events from the coherent and diffractive
        (proton + neutron) regimes.

        If the Vegas maps have not been trained, ``train()`` is called first.

        Returns
        -------
        dict
            Keys: ``x1, x2, x3, x4, x5, x6, m6, Enu, weight, mode``.
            All values are 1-D numpy arrays. ``weight`` is in cm² per target.
        """
        if self._maps is None:
            self.train(verbose=verbose)

        n_each = max(self.n_events // len(self._maps), 1)
        keys = ["x1", "x2", "x3", "x4", "x5", "x6", "m6", "Enu", "weight", "mode"]
        parts = []
        for label, f, integ, mult in self._maps:
            ev = self._generate_from_map(f, integ, n_each, label)
            ev["weight"] = ev["weight"] * mult
            parts.append(ev)
        return {k: np.concatenate([p[k] for p in parts]) for k in keys}
