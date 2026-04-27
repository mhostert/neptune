"""
neptune.nu_electron
===================
Neutrino-electron elastic scattering: cross sections and event generator.

Process:
    nu_alpha + e- → nu_alpha + e-

Both Standard Model (W + Z exchange) and BSM (extra vector Z') contributions
are supported.  The BSM Z' adds a contact-like correction to the SM vector
coupling Cv:

    Cv → Cv + QV * QL * g'² / (M_Z'² + 2 m_e T_e) / (2√2 G_F)

where QV is the Z' coupling to the neutrino, QL its coupling to the
electron, M_Z' the Z' mass, and T_e the recoil kinetic energy.

Differential cross section (in GeV^-2):

    dσ/dT_e = 2 m_e G_F² / π *
              [ C_L² + C_R² (1 - T_e/E_ν)² - C_L C_R m_e T_e / E_ν² ]

with
    C_L = (Cv + Ca)/2,  C_R = (Cv - Ca)/2          (neutrino mode)
    C_L = (Cv - Ca)/2,  C_R = (Cv + Ca)/2          (antineutrino mode)

SM couplings:
    nu_e + e:    Cv = +0.5 + 2 sw²,  Ca = +0.5    (NC + CC)
    nu_mu + e:   Cv = -0.5 + 2 sw²,  Ca = -0.5    (NC only)
    nu_tau + e:  Cv = -0.5 + 2 sw²,  Ca = -0.5

References
----------
Vogel & Engel, Phys.Rev. D39 (1989) 3378
de Gouvea & Jenkins, Phys.Rev. D74 (2006) 033004
Lindner et al., 1803.00060
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from neptune import const


# ─── Cross-section primitives ────────────────────────────────────────────────


def _sm_couplings(nu_flavor: str, is_nubar: bool):
    """Return (Cv, Ca) for nu_alpha + e elastic scattering in the SM."""
    if nu_flavor == "e":
        Cv = +0.5 + 2.0 * const.sw2
        Ca = +0.5
    else:
        Cv = -0.5 + 2.0 * const.sw2
        Ca = -0.5
    return Cv, Ca


def _CL_CR(Cv, Ca, is_nubar: bool):
    """Helicity projections (CL, CR). Antineutrino mode swaps CL ↔ CR."""
    if is_nubar:
        CL = 0.5 * (Cv - Ca)
        CR = 0.5 * (Cv + Ca)
    else:
        CL = 0.5 * (Cv + Ca)
        CR = 0.5 * (Cv - Ca)
    return CL, CR


def Te_max(Enu: np.ndarray) -> np.ndarray:
    """Kinematic upper limit on the electron recoil kinetic energy."""
    Enu = np.asarray(Enu, dtype=float)
    return 2.0 * Enu**2 / (const.m_e + 2.0 * Enu)


def dsigma_dTe(
    Enu: np.ndarray,
    Te: np.ndarray,
    nu_flavor: str = "mu",
    is_nubar: bool = False,
    mzprime: Optional[float] = None,
    gprime: float = 0.0,
    QV: float = 1.0,
    QL: float = 1.0,
) -> np.ndarray:
    """
    Differential cross section dσ/dT_e for nu_alpha + e → nu_alpha + e [cm²/GeV].

    Parameters
    ----------
    Enu : float or array
        Incoming neutrino energy [GeV].
    Te : float or array
        Outgoing electron recoil kinetic energy [GeV].
    nu_flavor : str
        'e', 'mu', or 'tau'.
    is_nubar : bool
        Antineutrino mode.
    mzprime : float, optional
        Z' mass [GeV]. If None or gprime=0, pure SM is returned.
    gprime : float
        Z' coupling.
    QV, QL : float
        Z' charges of the neutrino (QV) and the electron (QL).

    Returns
    -------
    dsigma/dT_e [cm²/GeV] (array, same broadcast shape as Enu and Te)
    """
    Enu = np.asarray(Enu, dtype=float)
    Te = np.asarray(Te, dtype=float)

    Cv_SM, Ca_SM = _sm_couplings(nu_flavor, is_nubar)
    if mzprime is not None and gprime != 0.0:
        bsm = (
            QV * QL * gprime**2
            / (2.0 * np.sqrt(2.0) * const.Gf)
            / (mzprime**2 + 2.0 * const.m_e * Te)
        )
        Cv = Cv_SM + bsm
    else:
        Cv = Cv_SM
    Ca = Ca_SM

    CL, CR = _CL_CR(Cv, Ca, is_nubar)

    pref = 2.0 * const.m_e * const.Gf**2 / np.pi
    ds = pref * (
        CL**2
        + CR**2 * (1.0 - Te / Enu) ** 2
        - CL * CR * const.m_e * Te / Enu**2
    )
    # Cut off above kinematic limit (Te > Te_max → 0)
    ds = np.where(Te < Enu - const.m_e, ds, 0.0)
    return ds * const.GeV2_to_cm2


def total_xsec(
    Enu: np.ndarray,
    nu_flavor: str = "mu",
    is_nubar: bool = False,
    T_min: float = 0.0,
    T_max: Optional[float] = None,
    mzprime: Optional[float] = None,
    gprime: float = 0.0,
    QV: float = 1.0,
    QL: float = 1.0,
    n_grid: int = 2000,
) -> np.ndarray:
    """
    Total nu-e elastic cross section above electron recoil threshold T_min [cm²].

    Pure SM result is computed analytically from the closed-form integral.
    BSM cases use a fast trapezoidal quadrature over the analytic
    differential cross section.

    Parameters
    ----------
    Enu : float or array
        Neutrino energy [GeV].
    nu_flavor : str
        'e', 'mu', or 'tau'.
    is_nubar : bool
    T_min : float
        Minimum electron recoil threshold [GeV].
    T_max : float, optional
        Maximum recoil cut [GeV]. If None, kinematic max is used.
    mzprime : float, optional
        Z' mass.  If None or gprime=0, pure SM.
    gprime : float
    QV, QL : float
    n_grid : int
        Number of quadrature points used in the BSM case.

    Returns
    -------
    sigma [cm²]
    """
    Enu = np.atleast_1d(np.asarray(Enu, dtype=float))
    out = np.zeros_like(Enu)

    Cv_SM, Ca_SM = _sm_couplings(nu_flavor, is_nubar)
    CL_SM, CR_SM = _CL_CR(Cv_SM, Ca_SM, is_nubar)

    Tmax_kin = Te_max(Enu)
    T1 = Tmax_kin if T_max is None else np.minimum(Tmax_kin, T_max)
    T0 = np.full_like(Enu, T_min)

    valid = T1 > T0
    if not np.any(valid):
        return out if out.size > 1 else float(out[0])

    if mzprime is None or gprime == 0.0:
        E = Enu[valid]
        a, b = T0[valid], T1[valid]
        pref = 2.0 * const.m_e * const.Gf**2 / np.pi
        # ∫ CL² + CR²(1-T/E)² − CL CR m_e T/E²  dT
        I = (
            (CL_SM**2 + CR_SM**2) * (b - a)
            - CR_SM**2 / E * (b**2 - a**2)
            + CR_SM**2 / (3.0 * E**2) * (b**3 - a**3)
            - CL_SM * CR_SM * const.m_e / (2.0 * E**2) * (b**2 - a**2)
        )
        out[valid] = pref * I * const.GeV2_to_cm2
    else:
        # Quadrature for BSM: vectorised in Enu
        E = Enu[valid]
        a, b = T0[valid], T1[valid]
        u = np.linspace(0.0, 1.0, n_grid)[None, :]
        T = a[:, None] + (b - a)[:, None] * u
        ds = dsigma_dTe(
            E[:, None],
            T,
            nu_flavor=nu_flavor,
            is_nubar=is_nubar,
            mzprime=mzprime,
            gprime=gprime,
            QV=QV,
            QL=QL,
        )
        out[valid] = np.trapezoid(ds, T, axis=1)

    if out.size == 1:
        return float(out[0])
    return out


# ─── Model classes mirroring DarkNews / TridentSMModel structure ─────────────


@dataclass
class NuElectronSMModel:
    """
    Standard Model neutrino-electron elastic scattering model.

    Parameters
    ----------
    nu_flavor : str
        'e', 'mu', or 'tau'.
    is_nubar : bool
        Antineutrino flag.
    """

    nu_flavor: str = "mu"
    is_nubar: bool = False

    def __post_init__(self):
        self.Cv, self.Ca = _sm_couplings(self.nu_flavor, self.is_nubar)
        self.CL, self.CR = _CL_CR(self.Cv, self.Ca, self.is_nubar)

    def dsigma_dTe(self, Enu, Te):
        """SM differential cross section [cm²/GeV]."""
        return dsigma_dTe(
            Enu, Te,
            nu_flavor=self.nu_flavor,
            is_nubar=self.is_nubar,
        )

    def total_xsec(self, Enu, T_min=0.0, T_max=None):
        """SM total cross section above threshold [cm²]."""
        return total_xsec(
            Enu,
            nu_flavor=self.nu_flavor,
            is_nubar=self.is_nubar,
            T_min=T_min, T_max=T_max,
        )

    def __repr__(self):
        kind = "nubar" if self.is_nubar else "nu"
        return f"NuElectronSMModel({kind}_{self.nu_flavor} + e)"


@dataclass
class NuElectronBSMModel(NuElectronSMModel):
    """
    SM + BSM Z' neutrino-electron scattering model.

    The BSM piece adds an L-mu - L-tau-like (or kinetic-mixed) Z' correction:

        Cv_eff = Cv_SM + QV * QL * gprime² / (M² + 2 m_e T) / (2√2 G_F)

    Parameters
    ----------
    mzprime : float
        Z' mass [GeV].
    gprime : float
        Z' coupling.
    QV : float
        Z' charge of the incoming neutrino flavor (vector).
    QL : float
        Z' charge of the electron (vector).
    """

    mzprime: float = 0.1
    gprime: float = 1e-3
    QV: float = 1.0
    QL: float = 1.0

    def dsigma_dTe(self, Enu, Te):
        """SM+BSM differential cross section [cm²/GeV]."""
        return dsigma_dTe(
            Enu, Te,
            nu_flavor=self.nu_flavor,
            is_nubar=self.is_nubar,
            mzprime=self.mzprime, gprime=self.gprime,
            QV=self.QV, QL=self.QL,
        )

    def total_xsec(self, Enu, T_min=0.0, T_max=None, n_grid=2000):
        return total_xsec(
            Enu,
            nu_flavor=self.nu_flavor,
            is_nubar=self.is_nubar,
            T_min=T_min, T_max=T_max,
            mzprime=self.mzprime, gprime=self.gprime,
            QV=self.QV, QL=self.QL,
            n_grid=n_grid,
        )

    def __repr__(self):
        kind = "nubar" if self.is_nubar else "nu"
        return (
            f"NuElectronBSMModel({kind}_{self.nu_flavor} + e, "
            f"mzprime={self.mzprime:.3g}, gprime={self.gprime:.3g}, "
            f"QV={self.QV}, QL={self.QL})"
        )


# ─── Process: cross-section calculator with optional flux convolution ─────────


class NuElectronProcess:
    """
    Total / differential cross section calculator for nu + e elastic scattering.

    Parameters
    ----------
    model : NuElectronSMModel or NuElectronBSMModel
    Enu : float, optional
        Fixed neutrino energy [GeV].
    Emin, Emax : float
        Energy window for flux convolution [GeV].
    flux : callable, optional
        Flux dN/dE [arbitrary units]. If supplied alongside Emin/Emax,
        flux_avg_xsec() returns the flux-averaged cross section.
    T_min, T_max : float
        Electron recoil cuts [GeV].
    """

    def __init__(
        self,
        model,
        Enu: Optional[float] = None,
        Emin: float = 0.01,
        Emax: float = 100.0,
        flux: Optional[Callable] = None,
        T_min: float = 0.0,
        T_max: Optional[float] = None,
    ):
        self.model = model
        self.Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        self.T_min = T_min
        self.T_max = T_max

    def total_xsec(self, Enu=None):
        """Total nu-e cross section at fixed energy [cm²]."""
        E = self.Enu if Enu is None else Enu
        return self.model.total_xsec(E, T_min=self.T_min, T_max=self.T_max)

    def dsigma_dTe(self, Te, Enu=None):
        """Differential cross section dσ/dT_e [cm²/GeV]."""
        E = self.Enu if Enu is None else Enu
        return self.model.dsigma_dTe(E, Te)

    def sigma_scan(self, Enu_arr):
        """Total cross section at multiple energies. Returns (Enu, sigma)."""
        Enu_arr = np.asarray(Enu_arr, dtype=float)
        sig = self.model.total_xsec(Enu_arr, T_min=self.T_min, T_max=self.T_max)
        sig = np.atleast_1d(sig)
        return Enu_arr, sig

    def flux_avg_xsec(self, n_grid: int = 200):
        """
        Flux-averaged cross section ∫ dE σ(E) Φ(E) / ∫ dE Φ(E) [cm²].
        """
        if self.flux is None:
            raise ValueError("Flux must be provided to compute flux-averaged xsec.")
        E_arr = np.linspace(self.Emin, self.Emax, n_grid)
        f = np.asarray(self.flux(E_arr), dtype=float)
        sig = self.model.total_xsec(E_arr, T_min=self.T_min, T_max=self.T_max)
        sig = np.atleast_1d(sig)
        denom = np.trapezoid(f, E_arr)
        if denom <= 0:
            raise ValueError("Flux integral is non-positive on [Emin, Emax].")
        return np.trapezoid(sig * f, E_arr) / denom


# ─── Event generator with full kinematics (4-vectors) ─────────────────────────


class NuElectronGenerator:
    """
    Monte Carlo generator for nu_alpha + e- → nu_alpha + e- events.

    Produces weighted events with full 4-momenta in the lab frame
    (electron at rest as the target).  The event format mirrors DarkNews:
    a pandas DataFrame with MultiIndex columns (P_projectile, P_target,
    P_decay_ell_minus, P_decay_N_daughter, w_event_rate, ...).

    Parameters
    ----------
    model : NuElectronSMModel or NuElectronBSMModel
    Enu : float, optional
        Fixed neutrino energy [GeV]. If None, sample from flux.
    Emin, Emax : float
        Energy bounds for flux mode [GeV].
    flux : callable, optional
        dN/dE flux function (used if Enu is None).
    T_min, T_max : float
        Cuts on the electron recoil kinetic energy.
    n_events : int
        Number of weighted events to produce.
    seed : int, optional
        RNG seed.
    """

    def __init__(
        self,
        model,
        Enu: Optional[float] = None,
        Emin: float = 0.01,
        Emax: float = 100.0,
        flux: Optional[Callable] = None,
        T_min: float = 0.0,
        T_max: Optional[float] = None,
        n_events: int = 10_000,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.Enu = Enu
        self.Emin = Emin
        self.Emax = Emax
        self.flux = flux
        self.T_min = T_min
        self.T_max = T_max
        self.n_events = n_events
        self.rng = np.random.default_rng(seed)

    def _sample_Enu(self, n: int) -> np.ndarray:
        if self.Enu is not None:
            return np.full(n, self.Enu)
        if self.flux is None:
            return self.rng.uniform(self.Emin, self.Emax, n)
        # Flux sampling via simple grid + inverse-CDF on a fine grid
        grid = np.linspace(self.Emin, self.Emax, 4000)
        f = np.asarray(self.flux(grid), dtype=float)
        f = np.clip(f, 0.0, None)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))])
        cdf /= cdf[-1]
        u = self.rng.uniform(0.0, 1.0, n)
        return np.interp(u, cdf, grid)

    def generate(self):
        """
        Generate n weighted events.

        Returns
        -------
        pandas.DataFrame
            DarkNews-style MultiIndex columns:
              - P_projectile (0..3)         : incoming nu 4-vector
              - P_target (0..3)             : electron at rest
              - P_decay_ell_minus (0..3)    : outgoing electron
              - P_decay_N_daughter (0..3)   : outgoing nu
              - Te                          : electron kinetic energy
              - theta_e                     : electron scattering angle (rad)
              - Enu                         : neutrino energy
              - w_event_rate                : event weight (∝ cross section)
              - flavor, is_nubar, mode
        """
        import pandas as pd

        n = self.n_events
        Enu = self._sample_Enu(n)
        Tmax_kin = Te_max(Enu)
        T_low = np.full(n, self.T_min)
        T_high = Tmax_kin if self.T_max is None else np.minimum(Tmax_kin, self.T_max)

        # Reject any events with negative window (shouldn't happen above threshold)
        good = T_high > T_low
        if not np.all(good):
            Enu = Enu[good]
            T_low = T_low[good]
            T_high = T_high[good]
            n = good.sum()

        # Sample T uniformly in [T_low, T_high] and weight by dσ/dT * (T_high - T_low)
        u = self.rng.uniform(0.0, 1.0, n)
        Te = T_low + (T_high - T_low) * u
        ds = self.model.dsigma_dTe(Enu, Te)
        weight = ds * (T_high - T_low)

        # If sampling from flux: divide by (Emax-Emin)/<flux normalisation>  → handled
        # implicitly by sampling Enu from flux, so weight already represents dN/dEnu*dσ.

        # Reconstruct 4-vectors in the lab frame (electron at rest)
        me = const.m_e
        Ee = Te + me
        pe = np.sqrt(np.maximum(Ee**2 - me**2, 0.0))
        # Energy-momentum conservation: cos θ_e from
        #   Ee = Enu - Enu' ;  pe cos θ_e = Enu - Enu' cos θ_nu  (combined)
        # Closed-form: cos θ_e = (Enu + me) / Enu * pe / Ee_term
        # Use the standard relation:
        #   cos θ_e = (1 + me/Enu) * sqrt(Te / (Te + 2 me))
        cos_te = np.minimum(
            (1.0 + me / Enu) * np.sqrt(Te / (Te + 2.0 * me)),
            1.0,
        )
        sin_te = np.sqrt(np.maximum(1.0 - cos_te**2, 0.0))
        theta_e = np.arccos(cos_te)

        # Random azimuth
        phi = self.rng.uniform(0.0, 2.0 * np.pi, n)

        # Incoming nu along +z
        P_nu_in = np.column_stack([Enu, np.zeros(n), np.zeros(n), Enu])
        # Target electron at rest
        P_e_in = np.column_stack([np.full(n, me), np.zeros(n), np.zeros(n), np.zeros(n)])
        # Outgoing electron
        px_e = pe * sin_te * np.cos(phi)
        py_e = pe * sin_te * np.sin(phi)
        pz_e = pe * cos_te
        P_e_out = np.column_stack([Ee, px_e, py_e, pz_e])
        # Outgoing nu = P_nu_in + P_e_in - P_e_out
        P_nu_out = P_nu_in + P_e_in - P_e_out

        # Build DataFrame matching DarkNews-style MultiIndex columns
        idx = ["0", "1", "2", "3"]
        data = {
            ("P_projectile", "0"): P_nu_in[:, 0],
            ("P_projectile", "1"): P_nu_in[:, 1],
            ("P_projectile", "2"): P_nu_in[:, 2],
            ("P_projectile", "3"): P_nu_in[:, 3],
            ("P_target", "0"): P_e_in[:, 0],
            ("P_target", "1"): P_e_in[:, 1],
            ("P_target", "2"): P_e_in[:, 2],
            ("P_target", "3"): P_e_in[:, 3],
            ("P_decay_ell_minus", "0"): P_e_out[:, 0],
            ("P_decay_ell_minus", "1"): P_e_out[:, 1],
            ("P_decay_ell_minus", "2"): P_e_out[:, 2],
            ("P_decay_ell_minus", "3"): P_e_out[:, 3],
            ("P_decay_N_daughter", "0"): P_nu_out[:, 0],
            ("P_decay_N_daughter", "1"): P_nu_out[:, 1],
            ("P_decay_N_daughter", "2"): P_nu_out[:, 2],
            ("P_decay_N_daughter", "3"): P_nu_out[:, 3],
            ("Enu", ""): Enu,
            ("Te", ""): Te,
            ("theta_e", ""): theta_e,
            ("w_event_rate", ""): weight,
        }
        df = pd.DataFrame(data)
        df.attrs["model"] = repr(self.model)
        df.attrs["nu_flavor"] = self.model.nu_flavor
        df.attrs["is_nubar"] = self.model.is_nubar
        df.attrs["target"] = "electron"
        df.attrs["scattering_regime"] = "elastic"
        return df
