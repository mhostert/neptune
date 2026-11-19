"""
Δ(1232) resonance contribution to neutrino trident production.

The electromagnetic N → Δ vertex is parity-conserving, so all axial form
factors vanish (C_3^A = C_4^A = C_5^A = C_6^A = 0). With Conserved Vector
Current and the small isoscalar amplitude dropped (C_5^V = 0), the EM
hadronic tensor reduces to the standard two structure functions
``F_1, F_2`` — exactly the same tensor decomposition as the inelastic
DIS regime. Consequently the leptonic projector and the
``hT/hL`` flux functions exported by ``mathematica/BSM_trident.nb``
for ``regime='dis'`` apply verbatim; only ``F_1, F_2`` change.

We use the Lalakulich–Paschos parameterisation (Phys. Rev. D 71, 074003
[hep-ph/0501109], hep-ph/0408185, 0704.1991):

* Modified-dipole vector form factors

  .. math::

      C_3^V(Q^2) &= \\frac{C_3^V(0)}{(1 + Q^2/M_V^2)^2}\\,
                                \\frac{1}{1 + Q^2/(4 M_V^2)}, \\\\
      C_4^V(Q^2, W) &= -\\frac{m_N}{W}\\, C_3^V(Q^2),

  with :math:`C_3^V(0) \\simeq 1.95`, :math:`M_V = 0.84` GeV.

* Vector-only :math:`V_1, V_2` from LP appendix A.

* Relativistic Breit–Wigner with running P-wave width

  .. math::

      \\rho_{BW}(W^2) = \\frac{1}{\\pi}\\,
          \\frac{M_R\\,\\Gamma_R(W)}{(W^2 - M_R^2)^2 + M_R^2\\Gamma_R^2(W)},
      \\qquad
      \\Gamma_R(W) = \\Gamma_0\\,[\\,p_\\pi(W)/p_\\pi(M_R)\\,]^3.

* Per-nucleus structure functions

  .. math::

      F_1^{RES,A}(x_B, Q^2) &= A \\cdot \\tfrac{1}{3} \\cdot V_1^V \\cdot \\rho_{BW}(W^2), \\\\
      F_2^{RES,A}(x_B, Q^2) &= A \\cdot \\tfrac{1}{3} \\cdot (qp/m_N^2)
                              \\cdot V_2^V \\cdot \\rho_{BW}(W^2),

  where the EM-channel isospin factor is ``1/3`` (Paschos–Schienbein–Yu
  eq. 2.1, *not* ``2/3``). The per-target multiplier is ``A`` because the
  EM matrix elements squared for :math:`\\gamma p \\to \\Delta^+` and
  :math:`\\gamma n \\to \\Delta^0` are equal.

Sign convention
---------------
The neptune pipeline carries the spacelike invariant ``x1 = Q^2 > 0`` and
``x2 = p_\\nu \\cdot (P_N - P'_N) > 0``. Lalakulich's :math:`q \\cdot p`
denotes :math:`q \\cdot P_N > 0` for physical kinematics. The two are
related by

    qp_LP = (W^2 + Q^2 - m_N^2) / 2 = Q^2 / (2 x_B).

This module works in Lalakulich's positive-``qp`` convention internally,
but exposes ``F_1(xB, Q^2)``, ``F_2(xB, Q^2)`` so it is duck-compatible
with the existing DIS structure-function interface used by
``DISTridentIntegrand``.
"""

import numpy as np

from neptune import const


# Default Lalakulich–Paschos parameters (PDG-consistent)
M_R_DEFAULT     = 1.232    # Δ(1232) pole mass [GeV]
GAMMA0_DEFAULT  = 0.118    # on-shell width [GeV]
MV_DEFAULT      = 0.840    # vector dipole scale [GeV]
C3V0_DEFAULT    = 1.95     # C_3^V(0)
M_PI            = 0.1396   # charged-pion mass [GeV]


def _safe_sqrt(x):
    """Square root that returns 0 for negative arguments (vectorised)."""
    return np.sqrt(np.maximum(x, 0.0))


def pion_cm_momentum(W, m_N=const.m_AVG, m_pi=M_PI):
    """
    Pion CM momentum :math:`p_\\pi(W)` in the :math:`N\\pi` rest frame.

    Returns 0 below threshold (W ≤ m_N + m_π).
    """
    W = np.asarray(W, dtype=float)
    above = W > (m_N + m_pi)
    W_safe = np.where(above, W, m_N + m_pi + 1e-30)
    arg = (W_safe**2 - (m_N + m_pi)**2) * (W_safe**2 - (m_N - m_pi)**2)
    p = _safe_sqrt(arg) / (2.0 * W_safe)
    return np.where(above, p, 0.0)


def delta_running_width(W, M_R=M_R_DEFAULT, Gamma0=GAMMA0_DEFAULT,
                        m_N=const.m_AVG, m_pi=M_PI):
    """
    P-wave running width :math:`\\Gamma_R(W) = \\Gamma_0\\,[p_\\pi(W)/p_\\pi(M_R)]^3`.

    Returns 0 below the :math:`N\\pi` threshold so that the BW vanishes
    cleanly in the kinematically forbidden region.
    """
    p_pi_W = pion_cm_momentum(W, m_N=m_N, m_pi=m_pi)
    p_pi_R = pion_cm_momentum(np.asarray(M_R), m_N=m_N, m_pi=m_pi)
    ratio = np.where(p_pi_R > 0.0, p_pi_W / p_pi_R, 0.0)
    return Gamma0 * ratio**3


def breit_wigner(W2, M_R=M_R_DEFAULT, Gamma0=GAMMA0_DEFAULT,
                 m_N=const.m_AVG, m_pi=M_PI):
    """
    Relativistic Breit–Wigner spectral density :math:`\\rho_{BW}(W^2)`.

    Normalised so that :math:`\\int \\rho_{BW}(W^2) dW^2 \\to 1` in the
    narrow-width limit. Returns 0 below :math:`N\\pi` threshold.
    """
    W2 = np.asarray(W2, dtype=float)
    above = W2 > (m_N + m_pi)**2
    W2_safe = np.where(above, W2, (m_N + m_pi)**2 + 1e-30)
    W = np.sqrt(W2_safe)
    Gam = delta_running_width(W, M_R=M_R, Gamma0=Gamma0, m_N=m_N, m_pi=m_pi)
    denom = (W2_safe - M_R**2)**2 + (M_R * Gam)**2
    val = (M_R * Gam) / (np.pi * denom)
    return np.where(above & (Gam > 0.0), val, 0.0)


# ---------------------------------------------------------------------------
# Lalakulich–Paschos vector form factors
# ---------------------------------------------------------------------------

def C3V(Q2, C3V0=C3V0_DEFAULT, MV=MV_DEFAULT):
    """Modified-dipole :math:`C_3^V(Q^2)`."""
    Q2 = np.asarray(Q2, dtype=float)
    return C3V0 / (1.0 + Q2/MV**2)**2 / (1.0 + Q2/(4.0 * MV**2))


def C4V(Q2, W, C3V0=C3V0_DEFAULT, MV=MV_DEFAULT, m_N=const.m_AVG):
    """LP relation :math:`C_4^V = -(m_N/W) C_3^V`."""
    return -(m_N / np.asarray(W, dtype=float)) * C3V(Q2, C3V0=C3V0, MV=MV)


def _V1V_V2V(Q2, qp, W, M_R=M_R_DEFAULT,
             C3V0=C3V0_DEFAULT, MV=MV_DEFAULT, m_N=const.m_AVG):
    """
    Vector-only :math:`V_1^V`, :math:`V_2^V` (LP appendix A, axial → 0).

    Parameters
    ----------
    Q2 : array
        :math:`Q^2 > 0`.
    qp : array
        :math:`(q \\cdot p)_{LP} = (W^2 + Q^2 - m_N^2)/2 > 0`.
    W : array
        Hadronic invariant mass.
    """
    Q2 = np.asarray(Q2, dtype=float)
    qp = np.asarray(qp, dtype=float)
    W = np.asarray(W, dtype=float)
    C3 = C3V(Q2, C3V0=C3V0, MV=MV)
    C4 = C4V(Q2, W, C3V0=C3V0, MV=MV, m_N=m_N)
    MR = M_R
    Mn = m_N

    V1 = (
        (C3**2 / Mn**2) * (2.0 / (3.0 * MR**2)) * (
            (qp - Q2)**2 * (qp + Mn**2)
            + MR**2 * (qp**2 + Q2*Mn**2 + Q2*Mn*MR)
        )
        + (C4**2 / Mn**4) * (2.0 / 3.0) * (qp - Q2)**2 * (qp + Mn**2 - Mn*MR)
        + (C3 * C4 / Mn**3) * (2.0 / (3.0 * MR)) * (qp - Q2) * (
            (qp - Q2) * (qp + Mn**2 - 2.0*Mn*MR) + MR**2 * qp
        )
    )

    V2 = (
        (C3**2) * (2.0 / (3.0 * MR**2)) * Q2 * (qp + Mn**2 + MR**2)
        + (C4**2 / Mn**2) * (2.0 / 3.0) * Q2 * (qp + Mn**2 - Mn*MR)
        + (C3 * C4 / Mn) * (2.0 / (3.0 * MR)) * Q2 * (qp + (MR - Mn)**2)
    )
    return V1, V2


# ---------------------------------------------------------------------------
# Structure-function callable for the Δ resonance
# ---------------------------------------------------------------------------

class DeltaResonanceStructureFunctions:
    """
    Per-nucleus EM ``F_1, F_2`` from the Lalakulich–Paschos Δ(1232)
    vertex with a relativistic Breit–Wigner spectral function.

    Duck-typed for the ``structure_functions`` slot of
    ``DISTridentIntegrand`` / ``ResonantTridentIntegrand``: exposes
    ``F1(xB, Q2)`` and ``F2(xB, Q2)`` as well as the bookkeeping
    attributes ``q2_min``, ``q2_max``, ``x_min``.

    Parameters
    ----------
    A : float
        Mass number of the target nucleus. Per-nucleus F1/F2 carry an
        overall factor of ``A`` (impulse approximation, EM matrix
        elements equal on p and n).
    M_R, Gamma0 : float
        Δ pole mass and on-shell width [GeV].
    MV, C3V0 : float
        Modified-dipole scale and ``C_3^V(0)`` value.
    m_N, m_pi : float
        Nucleon and pion masses used in the running-width and
        threshold logic.
    iso : float
        EM/CC isospin conversion factor. Default ``1/3`` (PSY07
        eq. 2.1, channel-summed).
    """

    def __init__(
        self,
        A,
        M_R=M_R_DEFAULT,
        Gamma0=GAMMA0_DEFAULT,
        MV=MV_DEFAULT,
        C3V0=C3V0_DEFAULT,
        m_N=const.m_AVG,
        m_pi=M_PI,
        iso=1.0 / 3.0,
    ):
        self.A = float(A)
        self.M_R = float(M_R)
        self.Gamma0 = float(Gamma0)
        self.MV = float(MV)
        self.C3V0 = float(C3V0)
        self.m_N = float(m_N)
        self.m_pi = float(m_pi)
        self.iso = float(iso)

        # Bookkeeping for the integrand: keep the resonance window tight
        # so Vegas focuses on the BW peak. xB ≈ Q²/(M_R² + Q² − m_N²) for
        # the on-shell Δ; the lower bound just guards against divide-by-0
        # in log mappings, the upper bound is the elastic limit xB = 1.
        self.x_min = 1e-3
        self.q2_min = 0.0
        self.q2_max = np.inf

        # Threshold W² above which BW is non-zero
        self.W2_threshold = (self.m_N + self.m_pi) ** 2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def W2(self, xB, Q2):
        """W² = m_N² + Q²(1/x_B − 1)."""
        xB = np.asarray(xB, dtype=float)
        Q2 = np.asarray(Q2, dtype=float)
        # qp_LP = Q²/(2 xB) (positive); W² = m_N² + 2 qp − Q²
        qp = Q2 / (2.0 * np.where(xB > 0.0, xB, 1e-30))
        return self.m_N**2 + 2.0 * qp - Q2

    def _F1_F2(self, xB, Q2):
        """Compute (F1, F2) jointly to share the V1/V2/BW evaluation."""
        xB = np.asarray(xB, dtype=float)
        Q2 = np.asarray(Q2, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            qp = Q2 / (2.0 * np.where(xB > 0.0, xB, 1e-30))
        W2 = self.m_N**2 + 2.0 * qp - Q2
        above = (W2 > self.W2_threshold) & (xB > 0.0) & (Q2 > 0.0)
        W2_safe = np.where(above, W2, self.W2_threshold + 1e-30)
        W = np.sqrt(W2_safe)
        V1, V2 = _V1V_V2V(Q2, qp, W,
                          M_R=self.M_R, C3V0=self.C3V0,
                          MV=self.MV, m_N=self.m_N)
        rho = breit_wigner(W2_safe, M_R=self.M_R, Gamma0=self.Gamma0,
                           m_N=self.m_N, m_pi=self.m_pi)
        F1 = self.A * self.iso * V1 * rho
        F2 = self.A * self.iso * (qp / self.m_N**2) * V2 * rho
        F1 = np.where(above, F1, 0.0)
        F2 = np.where(above, F2, 0.0)
        return F1, F2

    def F1(self, xB, Q2):
        return self._F1_F2(xB, Q2)[0]

    def F2(self, xB, Q2):
        return self._F1_F2(xB, Q2)[1]


class SumStructureFunctions:
    """
    Aggregator that sums ``F1, F2`` from multiple structure-function objects.

    Useful for combining the Δ resonance with additional resonances, an
    external Bosted–Christy implementation, or a non-resonant background
    fit — all without touching the integrand code.

    Each component must expose ``F1(xB, Q2)`` and ``F2(xB, Q2)`` and return
    zero outside its own domain (all neptune structure functions do). The
    optional bookkeeping attributes ``q2_min``, ``q2_max``, ``x_min`` are
    aggregated to the union of the component domains, so that, e.g., a Δ
    component at Q² < 1 GeV² survives when summed with an LHAPDF set.

    Example
    -------
    >>> delta = DeltaResonanceStructureFunctions(A=40)
    >>> # roper = SomeP11Implementation(A=40)            # supplied by user
    >>> # bc    = bosted_christy_wrap(target='isoscalar') # external lib
    >>> sf = SumStructureFunctions(delta)                # plus more if avail.
    >>> sf.F1(0.5, 0.5)                                   # = delta.F1(...)
    """

    def __init__(self, *components):
        if not components:
            raise ValueError("SumStructureFunctions needs at least one component")
        self.components = list(components)

        def _agg(name, default, op):
            vals = [getattr(c, name) for c in components if hasattr(c, name)]
            return op(vals) if vals else default

        self.q2_min = _agg("q2_min", 0.0, min)
        self.q2_max = _agg("q2_max", np.inf, max)
        self.x_min  = _agg("x_min",  0.0, min)

    def F1(self, xB, Q2):
        return sum(c.F1(xB, Q2) for c in self.components)

    def F2(self, xB, Q2):
        return sum(c.F2(xB, Q2) for c in self.components)

    def __repr__(self):
        names = ", ".join(type(c).__name__ for c in self.components)
        return f"SumStructureFunctions({names})"
