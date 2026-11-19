"""
Nuclear models for the trident pipeline.

Three options for the per-nucleon Pauli-blocking factor that multiplies
the diffractive / DIS / resonance integrands:

* ``FreeNucleon``    — no nuclear effects; Pauli factor ≡ 1.
* ``FermiGas``       — uniform (relativistic) Fermi gas with constant
                       Fermi momentum ``p_F``. Smith–Moniz analytic
                       formula.
* ``LocalFermiGas``  — density-folded Pauli factor via a Woods–Saxon
                       profile and a local Fermi momentum
                       ``p_F(r) = ℏc · (3π²/2 · ρ(r))^{1/3}``
                       (isoscalar approximation).

All three expose the same interface ``nuclear_model.pauli_factor(qmag)``
returning the multiplicative Pauli factor for a 3-momentum transfer
``|q|`` (in GeV).

Standard Woods–Saxon parameters from de Vries et al. (At. Data Nucl.
Data Tables 36, 495 (1987)) are tabulated in ``WOODS_SAXON_PARAMS`` and
exposed via ``woods_saxon_density_for(symbol)``.

Effective constant Fermi momenta for the ``FermiGas`` choice are
tabulated in ``FERMI_MOMENTUM_GEV``. The diffractive integrand defaults to
``FermiGas(p_F=0.221)``, inherited from the BSMscan C++ code for all
targets. This equals the Moniz value for C12, not the tabulated Ar40 value.
"""

from functools import cached_property

import numpy as np


# ℏc = 0.197327 GeV·fm — converts (3π² ρ_fm)^{1/3} [fm⁻¹] to [GeV].
HBARC = 0.197327


# ---------------------------------------------------------------------------
# Smith–Moniz Pauli blocking factor
# ---------------------------------------------------------------------------

def pauli_blocking_factor(qmag, p_F):
    """
    Smith–Moniz Pauli-blocking factor for a uniform Fermi gas of
    momentum ``p_F``::

        f_PB(|q|) = 3 |q| / (4 p_F)  -  |q|³ / (16 p_F³)    if |q| ≤ 2 p_F
                  = 1                                            otherwise

    Both ``qmag`` and ``p_F`` may be arrays; broadcasting follows numpy.
    Returns 0 when ``p_F = 0`` (avoids 0/0).
    """
    qmag = np.asarray(qmag, dtype=float)
    p_F = np.asarray(p_F, dtype=float)
    safe_pF = np.where(p_F > 0.0, p_F, 1.0)  # avoid divide-by-0; we mask below
    xi = qmag / (2.0 * safe_pF)
    fpb = np.where(xi >= 1.0, 1.0, 1.5 * xi - 0.5 * xi**3)
    return np.where(p_F > 0.0, fpb, 0.0)


# ---------------------------------------------------------------------------
# Woods–Saxon nuclear density profile
# ---------------------------------------------------------------------------

# 2-parameter Fermi (= Woods–Saxon) parameters [fm] from de Vries et al.
# Table II ("DV") of At. Data Nucl. Data Tables 36 (1987) 495 where available;
# values for which the source uses harmonic-oscillator or Fourier-Bessel
# parameterisations are converted to an effective ``c, a`` of equal rms radius.
WOODS_SAXON_PARAMS = {
    "C12":  dict(A=12,  Z=6,  c=2.355, a=0.5224),
    "O16":  dict(A=16,  Z=8,  c=2.608, a=0.513),
    "Ar40": dict(A=40,  Z=18, c=3.53,  a=0.542),
    "Fe56": dict(A=56,  Z=26, c=4.106, a=0.519),
    "Pb208":dict(A=208, Z=82, c=6.62,  a=0.546),
}

# Conventional Smith–Moniz constant Fermi momenta [GeV].
FERMI_MOMENTUM_GEV = {
    "C12":   0.221,
    "O16":   0.225,
    "Ar40":  0.251,
    "Fe56":  0.255,
    "Pb208": 0.265,
}


class WoodsSaxonDensity:
    """
    Two-parameter Fermi (Woods–Saxon) nuclear density profile

    .. math::
        \\rho(r) = \\frac{\\rho_0}{1 + \\exp\\big((r-c)/a\\big)},
        \\qquad 4\\pi \\int_0^\\infty r^2\\,\\rho(r)\\,dr = A.

    All distances in fm, density in fm⁻³.

    Parameters
    ----------
    A : int
        Mass number (used to fix the normalisation ``ρ_0``).
    Z : int
        Atomic number (only used by ``rho_p``/``rho_n`` species splits).
    c : float
        Half-density radius [fm].
    a : float
        Surface diffuseness [fm].
    """

    def __init__(self, A, Z, c, a):
        self.A = float(A)
        self.Z = float(Z)
        self.c = float(c)
        self.a = float(a)

    @property
    def r_max(self):
        """Practical outer cutoff (15 diffuseness lengths past the half radius)."""
        return self.c + 15.0 * self.a

    @cached_property
    def rho0(self):
        """Central density [fm⁻³] fixed by ``∫ 4πr² ρ dr = A``."""
        # Two-parameter Fermi normalisation has a closed form involving
        # polylogarithms; numerical quadrature is plenty fast here.
        from scipy.integrate import quad
        norm, _ = quad(
            lambda r: r * r / (1.0 + np.exp((r - self.c) / self.a)),
            0.0, self.r_max,
        )
        return self.A / (4.0 * np.pi * norm)

    def rho(self, r):
        """Total nucleon density ρ(r) [fm⁻³]."""
        r = np.asarray(r, dtype=float)
        return self.rho0 / (1.0 + np.exp((r - self.c) / self.a))

    def rho_p(self, r):
        """Proton-only density (uniform Z/A fraction)."""
        return (self.Z / self.A) * self.rho(r)

    def rho_n(self, r):
        """Neutron-only density."""
        return ((self.A - self.Z) / self.A) * self.rho(r)

    def p_F_isoscalar(self, r):
        """
        Local isoscalar Fermi momentum [GeV] from the per-species
        density ``ρ/2``::

            p_F(r) = ℏc · (3π² ρ(r) / 2)^{1/3}.

        Accurate to a few percent for asymmetric nuclei; the per-species
        formulas are ``p_F^p = ℏc·(3π² ρ_p)^{1/3}`` and similarly for n.
        """
        rho = self.rho(r)
        return HBARC * (3.0 * np.pi**2 * rho / 2.0) ** (1.0 / 3.0)

    def __repr__(self):
        return (f"WoodsSaxonDensity(A={self.A:g}, Z={self.Z:g}, "
                f"c={self.c} fm, a={self.a} fm)")


def woods_saxon_density_for(symbol):
    """
    Look up a tabulated Woods–Saxon density by symbol (``'Ar40'``, ``'Pb208'``, …).

    See ``WOODS_SAXON_PARAMS`` for the available entries.
    """
    if symbol not in WOODS_SAXON_PARAMS:
        raise KeyError(
            f"Unknown nucleus {symbol!r}. Known: {sorted(WOODS_SAXON_PARAMS)}"
        )
    return WoodsSaxonDensity(**WOODS_SAXON_PARAMS[symbol])


# ---------------------------------------------------------------------------
# Nuclear model classes (FNA, FG, LFG)
# ---------------------------------------------------------------------------

class _NuclearModelBase:
    """Common interface: ``pauli_factor(qmag)`` → np.ndarray."""

    name = "abstract"

    def pauli_factor(self, qmag):                                  # pragma: no cover
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__}>"


class FreeNucleon(_NuclearModelBase):
    """Free-nucleon approximation: Pauli factor ≡ 1 everywhere."""

    name = "free-nucleon"

    def pauli_factor(self, qmag):
        return np.ones_like(np.asarray(qmag, dtype=float))


class FermiGas(_NuclearModelBase):
    """
    Uniform (constant ``p_F``) relativistic Fermi gas — Smith–Moniz.

    Parameters
    ----------
    p_F : float
        Fermi momentum in GeV. Defaults to 0.221 GeV, the BSMscan C++
        value used for every target (Moniz C12 value).
    """

    name = "fermi-gas"

    def __init__(self, p_F=0.221):
        self.p_F = float(p_F)

    @classmethod
    def for_nucleus(cls, symbol):
        """Fermi gas with the tabulated effective ``p_F`` for a nucleus."""
        if symbol not in FERMI_MOMENTUM_GEV:
            raise KeyError(f"No tabulated p_F for {symbol!r}. "
                           f"Known: {sorted(FERMI_MOMENTUM_GEV)}")
        return cls(p_F=FERMI_MOMENTUM_GEV[symbol])

    def pauli_factor(self, qmag):
        return pauli_blocking_factor(qmag, self.p_F)

    def __repr__(self):
        return f"FermiGas(p_F={self.p_F:.3f} GeV)"


class LocalFermiGas(_NuclearModelBase):
    """
    Density-folded local Fermi gas. Pauli factor at three-momentum ``|q|``
    is the radial average

    .. math::
        f_{PB}^{LFG}(|q|) = \\frac{1}{A}\\int_0^\\infty 4\\pi r^2\\,\\rho(r)\\,
                            f_{PB}\\big(|q|, p_F(r)\\big)\\,dr,

    pre-tabulated on a ``|q|`` grid and interpolated at evaluation time so
    the cost per Vegas point is negligible.

    Parameters
    ----------
    density : WoodsSaxonDensity
        Nuclear density profile and bookkeeping.
    qmag_max : float
        Upper edge of the pre-tabulation grid [GeV]. For ``|q| > qmag_max``
        the factor is unity (well above ``2 p_F``).
    n_qmag : int
        Number of grid points in ``|q|``.
    n_radial : int
        Number of trapezoid samples in the radial integration.
    """

    name = "local-fermi-gas"

    def __init__(self, density, qmag_max=2.0, n_qmag=400, n_radial=256):
        if not isinstance(density, WoodsSaxonDensity):
            raise TypeError("density must be a WoodsSaxonDensity instance")
        self.density = density
        self.qmag_max = float(qmag_max)
        self.n_qmag = int(n_qmag)
        self.n_radial = int(n_radial)
        self._build_table()

    def _build_table(self):
        from scipy.interpolate import CubicSpline
        d = self.density
        r_grid = np.linspace(1e-3, d.r_max, self.n_radial)
        weight = 4.0 * np.pi * r_grid**2 * d.rho(r_grid)            # fm⁻¹
        pF_r = d.p_F_isoscalar(r_grid)                              # GeV

        qmag_grid = np.linspace(0.0, self.qmag_max, self.n_qmag)
        # broadcast: (nq, nr)
        f_PB = pauli_blocking_factor(qmag_grid[:, None], pF_r[None, :])
        avg = np.trapezoid(weight[None, :] * f_PB, r_grid, axis=-1) / d.A

        self._qmag_grid = qmag_grid
        self._f_avg = avg
        self._spline = CubicSpline(qmag_grid, avg, extrapolate=False)

    def pauli_factor(self, qmag):
        qmag = np.asarray(qmag, dtype=float)
        with np.errstate(invalid="ignore"):
            interp = self._spline(qmag)
        # Above the table top |q| ≫ 2 p_F, the Pauli factor saturates to 1.
        # Clip the cubic-spline overshoot at the 2 p_F kink (O(1e-7)).
        interp = np.clip(np.nan_to_num(interp, nan=1.0), 0.0, 1.0)
        return np.where(qmag < self.qmag_max, interp, 1.0)

    @property
    def effective_p_F(self):
        """Effective constant Fermi momentum for benchmarking against ``FermiGas``.

        Single-point match: the ``p_F`` for which the Fermi-gas factor equals
        the LFG factor at ``|q| = p_F(r=0)``.
        """
        from scipy.optimize import brentq
        pF_central = self.density.p_F_isoscalar(0.0)
        target = self.pauli_factor(pF_central)
        try:
            return brentq(
                lambda pF: pauli_blocking_factor(pF_central, pF) - target,
                0.05, 0.5,
            )
        except ValueError:
            return pF_central

    def __repr__(self):
        return f"LocalFermiGas({self.density!r})"
