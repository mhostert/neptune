"""
MC.py
=====
Monte Carlo event generator for neutrino trident events.

Generates unweighted phase space samples distributed according to the
differential cross section using Vegas importance sampling followed by
hit-or-miss rejection.

Example
-------
>>> from neptune.MC import TridentGenerator
>>> from neptune.model import TridentSMModel
>>> model = TridentSMModel(nu_flavor='mu', l1_flavor='mu', l2_flavor='mu')
>>> gen = TridentGenerator(model, Z=13, A=27, Enu=10.0, n_events=1000)
>>> events = gen.generate()
>>> events.keys()
dict_keys(['s', 'phi', 'theta', 't', 'l', 'q', 'Enu', 'weight', 'mode'])
"""

import numpy as np
import vegas

from neptune.model import TridentSMModel, TridentBSMModel
from neptune.integrands import (
    DiffractiveTridentIntegrand,
    CoherentTridentIntegrand,
    CoherentTridentIntegrandFull,
    Q_MAX_COH_DEFAULT,
    Q_MAX_DIFF_DEFAULT,
    Q_max_coh_for_A,
)
from neptune.phase_space import map_unit_to_physical


class TridentGenerator:
    """
    Monte Carlo event generator for neutrino trident production.

    Uses Vegas to build an importance sampling map, then generates weighted
    events by sampling from the trained map.  Optionally applies unweighting
    (hit-or-miss) to produce unweighted event samples.

    Parameters
    ----------
    model : TridentSMModel or TridentBSMModel
        Physics model.
    Z, A : int
        Target nucleus.
    Enu : float or None
        Fixed neutrino energy [GeV]. If None, use flux.
    Emin, Emax : float
        Energy range if using flux [GeV].
    flux : callable or None
        Flux function dN/dE.
    Q_max_coh : float
        Upper Q^2 limit for coherent regime.
    Q_max_diff : float
        Upper Q^2 limit for diffractive regime.
    use_epa : bool
        If True (default) use the 6-D EPA-factorised coherent integrand;
        if False use the full 8-D coherent matrix element. Diffractive
        always uses EPA.
    Mn : float, optional
        Target mass for the full coherent integrand [GeV]; defaults to
        A * m_AVG. Ignored when use_epa=True.
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
        Q_max_coh=None,
        Q_max_diff=Q_MAX_DIFF_DEFAULT,
        use_epa=True,
        Mn=None,
        n_events=10_000,
        nitn=10,
        neval=50_000,
        seed=None,
    ):
        self.model = model
        self.Z = Z
        self.A = A
        self.Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        self.Q_max_coh = Q_max_coh_for_A(A) if Q_max_coh is None else Q_max_coh
        self.Q_max_diff = Q_max_diff
        self.use_epa = use_epa
        self.Mn = Mn
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
                self.model.ml1,
                self.model.ml2,
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
            self.model.ml1,
            self.model.ml2,
            self.Z,
            self.A,
            Mn=self.Mn,
            Enu=self.Enu,
            Emin=self.Emin,
            Emax=self.Emax,
            flux=self.flux,
        )

    def train(self, verbose=False):
        """
        Train Vegas importance sampling maps for both diffractive and coherent.

        Must be called before generate().
        """
        dif_f = self._make_diffractive_integrand()
        coh_f = self._make_coherent_integrand()

        # Train diffractive map
        self._dif_integrand = dif_f
        self._dif_map = vegas.Integrator(dif_f.ndim * [[0, 1]])
        self._dif_map(dif_f, nitn=self.nitn, neval=self.neval, adapt=True)
        if verbose:
            print("Diffractive map trained.")

        # Train coherent map
        self._coh_integrand = coh_f
        self._coh_map = vegas.Integrator(coh_f.ndim * [[0, 1]])
        self._coh_map(coh_f, nitn=self.nitn, neval=self.neval, adapt=True)
        if verbose:
            print("Coherent map trained.")

    def _generate_from_map(self, integrand, integrator, n_events, mode):
        """
        Generate weighted events by sampling from a trained Vegas map.

        Parameters
        ----------
        integrand : vegas.BatchIntegrand
        integrator : vegas.Integrator
            Trained Vegas integrator.
        n_events : int
        mode : str
            'diffractive' or 'coherent'.

        Returns
        -------
        dict of arrays.  Keys depend on the integrand variant:
          * EPA (6/7-D): s, phi, theta, t, l, q, Enu, weight, mode
          * Full coherent (8/9-D): x1..x6, m6, Enu, weight, mode
        """
        # Collect raw samples from the trained Vegas map.
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

        if isinstance(integrand, CoherentTridentIntegrandFull):
            # Full 8-D coherent kinematics.
            if integrand.fixed_Enu is not None:
                Enu_arr = np.full(len(xx), integrand.fixed_Enu)
                x_phase = xx
            else:
                Enu_arr = (
                    (integrand.Emax - integrand.Emin) * xx[:, 8] + integrand.Emin
                )
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
                "mode": np.full(len(xx), mode),
            }

        # EPA path: map xx columns to (s, phi, theta, t, l, q, Enu)
        ml12sq = integrand.ml12sq
        if integrand.fixed_Enu is not None:
            Enu_arr = np.full(len(xx), integrand.fixed_Enu)
        else:
            Enu_arr = (
                (integrand.Emax - integrand.Emin) * xx[:, 6] + integrand.Emin
            )

        if mode == "diffractive":
            q_min = np.maximum(integrand.Q_max_coh, ml12sq / (2.0 * Enu_arr))
            q_max = integrand.Q_max_diff
        else:
            q_min = ml12sq / (2.0 * Enu_arr)
            q_max = integrand.Q_max_coh

        q_true = xx[:, 5] * (q_max - q_min) + q_min
        s_true = xx[:, 0] * (2 * Enu_arr * q_true - ml12sq) + ml12sq
        phi_true = xx[:, 1] * 2 * np.pi
        theta_true = xx[:, 2] * np.pi
        t_true = xx[:, 3] * (s_true - ml12sq) + ml12sq
        l_true = xx[:, 4] * (t_true - ml12sq) + ml12sq - t_true

        return {
            "s": s_true,
            "phi": phi_true,
            "theta": theta_true,
            "t": t_true,
            "l": l_true,
            "q": q_true,
            "Enu": Enu_arr,
            "weight": weights,
            "mode": np.full(len(xx), mode),
        }

    def generate(self, verbose=False):
        """
        Generate weighted trident events from both diffractive and coherent.

        If the Vegas maps have not been trained, train() is called first.

        Parameters
        ----------
        verbose : bool

        Returns
        -------
        dict
            Keys: 's', 'phi', 'theta', 't', 'l', 'q', 'Enu', 'weight', 'mode'.
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

        # Diffractive uses EPA (s, phi, theta, t, l, q); coherent may use either
        # EPA or the full 8-D scheme (x1..x6, m6).  Take the union of keys and
        # pad missing values with NaN, matching DarkNews-style table output.
        all_keys = list(dict.fromkeys(list(dif_events) + list(coh_events)))
        n_dif = len(dif_events["weight"])
        n_coh = len(coh_events["weight"])
        combined = {}
        for k in all_keys:
            if k == "mode":
                combined[k] = np.concatenate([
                    dif_events.get(k, np.full(n_dif, "diffractive")),
                    coh_events.get(k, np.full(n_coh, "coherent")),
                ])
            else:
                a = dif_events[k] if k in dif_events else np.full(n_dif, np.nan)
                b = coh_events[k] if k in coh_events else np.full(n_coh, np.nan)
                combined[k] = np.concatenate([a, b])

        return combined
