"""
neptune.phase_space
===================
Kinematic limits, integration variable mappings, and 4-vector reconstruction
for neutrino trident processes.

Coherent / diffractive trident is a 2->4 nuclear process. All functions
are NumPy-vectorized over the batch dimension (first axis).

Physics conventions (matching C++ cross_sections.cxx):
  ν(p_ν) + N(P_N) → ν'(p'_ν) + l1⁻(p3) + l2⁺(p4) + N'(P'_N)

Dot-product invariants:
  x1 = Q² = -(P_N - P'_N)²     [spacelike, positive]
  x2 = p_ν · (P_N - P'_N)
  x3 = q · p_l2                 [q = p_ν - p'_ν]
  x4 = q · p_l1
  x5 = p_ν · p_l2
  x6 = p'_ν · p_l1

Integration variables (unit hypercube → physical):
  index 0 : u1_s → x1  (log-mapped for Q² peak)
  index 1 : u2_s → x2
  index 2 : u3_s → x3  (log-mapped near threshold)
  index 3 : PHI2_s → PHI2  (azimuthal, [0, 2π])
  index 4 : x5_s  → x5
  index 5 : u6_s  → m6 → x6 (propagator-optimised)
  index 6 : x7_s  → x7 = 2π·x7_s  (rotation angle)
  index 7 : x8_s  → x8 = 2π·x8_s  (rotation angle)
"""

import numpy as np

from neptune import const


# ─── Threshold energy ─────────────────────────────────────────────────────────


def trident_threshold(ml1: float, ml2: float, Mn: float) -> float:
    """
    Neutrino energy threshold for the trident process.

    E_threshold = [(ml1+ml2)² + 2(ml1+ml2)Mn] / (2Mn)
    """
    msum = ml1 + ml2
    return (msum**2 + 2.0 * msum * Mn) / (2.0 * Mn)


# ─── Q² (x1) limits ───────────────────────────────────────────────────────────


def trident_Q2_upper(Enu: float, ml1: float, ml2: float, Mn: float) -> float:
    """
    Upper kinematic limit on Q² (= x1_u in C++ code).
    """
    msum = ml1 + ml2
    r = msum**2 / (2.0 * Enu**2) * (1.0 + Enu / Mn)
    return (
        2.0
        * Enu**2
        / (1.0 + 2.0 * Enu / Mn)
        * (
            1.0
            - r
            + np.sqrt(
                (1.0 - r) ** 2 - msum**4 / (4.0 * Enu**4) * (1.0 + 2.0 * Enu / Mn)
            )
        )
    )


def trident_Q2_lower(Enu: float, ml1: float, ml2: float, Mn: float) -> float:
    """
    Lower kinematic limit on Q² (= x1_l in C++ code).
    """
    x1_u = trident_Q2_upper(Enu, ml1, ml2, Mn)
    return (ml1 + ml2) ** 4 / x1_u / (1.0 + 2.0 * Enu / Mn)


# ─── x2 limits ────────────────────────────────────────────────────────────────


def trident_x2_limits(
    x1: np.ndarray,
    Enu: float,
    ml1: float,
    ml2: float,
    Mn: float,
    x1_u: float = None,
    x1_l: float = None,
) -> tuple:
    """
    Compute x2 kinematic limits and the transformed variable u2 upper bound.

    Returns
    -------
    (x2_l, x2_u, u2_u)  where u2_l = 0 always.
    """
    x1 = np.asarray(x1, dtype=float)

    if x1_u is None:
        x1_u = trident_Q2_upper(Enu, ml1, ml2, Mn)
    if x1_l is None:
        x1_l = trident_Q2_lower(Enu, ml1, ml2, Mn)

    x2_l = 0.5 * (x1 + (ml1 + ml2) ** 2)
    x2_u = Enu * (np.sqrt(x1 + x1**2 / (4.0 * Mn**2)) - x1 / (2.0 * Mn))

    # upper bound on u2 (the actual integration variable, not x2 directly)
    # u2_u comes from the change-of-variables that flattens the x2 peak
    u2_u = (
        (1.0 + 2.0 * Enu / Mn)
        * (x1_u - x1)
        * (x1 - x1_l)
        / (
            (ml1 + ml2) ** 2
            + x1 * (1.0 + Enu / Mn)
            + 2.0 * Enu * np.sqrt(x1 + x1**2 / (4.0 * Mn**2))
        )
    )

    return x2_l, x2_u, u2_u


def dis_trident_x2_upper(x1: np.ndarray, Enu, Mn: float, xB: np.ndarray):
    """
    Upper kinematic limit on ``x2 = p_nu.q`` for DIS kinematics.

    This is the elastic expression with ``P.q = -Q²/(2 xB)`` instead of
    ``P.q = -Q²/2``.
    """
    x1 = np.asarray(x1, dtype=float)
    xB = np.asarray(xB, dtype=float)
    return Enu * (
        np.sqrt(np.maximum(x1 + x1**2 / (4.0 * Mn**2 * xB**2), 0.0))
        - x1 / (2.0 * Mn * xB)
    )


def dis_trident_xB_limits(
    x1: np.ndarray,
    Enu,
    ml1: float,
    ml2: float,
    Mn: float,
    W2_min: float = 4.0,
    xB_min: float = 1e-6,
):
    """
    Bjorken-x limits for fixed ``Q²`` in the DIS trident map.

    The lower limit enforces a real transverse momentum for the exchanged
    photon at the minimum leptonic invariant mass, and the upper limit
    enforces ``W² >= W2_min``.
    """
    x1 = np.asarray(x1, dtype=float)
    Enu = np.asarray(Enu, dtype=float)
    x2_l = 0.5 * (x1 + (ml1 + ml2) ** 2)

    denom = Mn * Enu * (1.0 - x2_l**2 / (np.maximum(x1, 1e-300) * Enu**2))
    kin_min = np.where(denom > 0.0, x2_l / denom, np.inf)

    w_denom = x1 + W2_min - Mn**2
    w_max = np.where(w_denom > 0.0, x1 / w_denom, 1.0)
    xB_l = np.maximum(float(xB_min), kin_min)
    xB_u = np.minimum(1.0, w_max)
    return xB_l, xB_u


# ─── x5 limits ────────────────────────────────────────────────────────────────


def trident_x5_limits(x1: np.ndarray, x2: np.ndarray, ml1: float, ml2: float) -> tuple:
    """
    Compute x5 = p_nu · p_l2 kinematic limits.

    Returns
    -------
    (x5_l, x5_u)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    Delta = 2.0 * x2 - x1
    disc = np.sqrt(
        np.maximum((Delta + ml2**2 - ml1**2) ** 2 - 4.0 * ml2**2 * Delta, 0.0)
    )

    x5_l = x2 / (2.0 * Delta) * (Delta + ml2**2 - ml1**2 - disc)
    x5_u = x2 / (2.0 * Delta) * (Delta + ml2**2 - ml1**2 + disc)

    return x5_l, x5_u


# ─── x3 limits ────────────────────────────────────────────────────────────────


def trident_x3_limits(
    x1: np.ndarray, x2: np.ndarray, x5: np.ndarray, ml1: float, ml2: float
) -> tuple:
    """
    Compute x3 = q · p_l2 kinematic limits and the log variable u3 limits.

    Returns
    -------
    (x3_l, x3_u, u3_l, u3_u)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x5 = np.asarray(x5, dtype=float)

    x3_l = ml2**2 / 2.0 * x2 / x5 - x1 / 2.0 * x5 / x2
    x3_u = 0.5 * (2.0 * x2 - x1 + ml2**2 - ml1**2) - x5

    u3_l = np.log(2.0 * x3_l + x1)
    u3_u = np.log(2.0 * x3_u + x1)

    return x3_l, x3_u, u3_l, u3_u


# ─── Wc² frame intermediate variables ─────────────────────────────────────────


def trident_frame_variables(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x5: np.ndarray,
    ml1: float,
    ml2: float,
) -> dict:
    """
    Compute intermediate kinematic quantities in the 'Wc frame'
    (the centre-of-mass frame of the p'_nu + l1 system).

    Matches the C++ code's definitions immediately before the PHI2 computation.

    Returns
    -------
    dict with keys: Wc2, E1, E2, E4, q0, qvec, p4vec, Cq, Sq, m6_max
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x3 = np.asarray(x3, dtype=float)
    x5 = np.asarray(x5, dtype=float)

    Wc2 = 2.0 * (x2 - x3 - x5) - x1 + ml2**2
    sWc2 = np.sqrt(np.maximum(Wc2, 0.0))

    E1 = (x2 - x5) / sWc2
    E4 = (Wc2 + ml1**2) / (2.0 * sWc2)
    q0 = (x2 - x1 - x3) / sWc2
    qvec = np.sqrt(np.maximum(q0**2 + x1, 0.0))
    p4vec = (Wc2 - ml1**2) / (2.0 * sWc2)
    E2 = p4vec  # same formula as p4vec = (Wc2 - ml1²)/2/sqrt(Wc2)

    # Cq = cos(theta_q), angle of q w.r.t. z-axis in the Wc frame
    Cq = (q0 * E1 - x2) / (qvec * E1)
    Sq = np.sqrt(np.maximum(1.0 - Cq**2, 0.0))

    m6_max = 2.0 * E1 * E2

    return dict(
        Wc2=Wc2,
        E1=E1,
        E2=E2,
        E4=E4,
        q0=q0,
        qvec=qvec,
        p4vec=p4vec,
        Cq=Cq,
        Sq=Sq,
        m6_max=m6_max,
    )


# ─── Derive x4, x6 from m6 and PHI2 ──────────────────────────────────────────


def trident_x4_x6(m6: np.ndarray, PHI2: np.ndarray, frame: dict) -> tuple:
    """
    Compute x4 = q · p_l1 and x6 = p'_nu · p_l1 given the integration variable
    m6 and the azimuthal angle PHI2.

    Parameters
    ----------
    m6 : array-like
        Integration variable (p'_nu · p_l2 related, range [0, m6_max]).
    PHI2 : array-like
        Azimuthal angle [0, 2π].
    frame : dict
        Output of trident_frame_variables().

    Returns
    -------
    (x4, x6)
    """
    m6 = np.asarray(m6, dtype=float)
    PHI2 = np.asarray(PHI2, dtype=float)

    E1, E2, E4 = frame["E1"], frame["E2"], frame["E4"]
    q0, qvec = frame["q0"], frame["qvec"]
    Cq, Sq = frame["Cq"], frame["Sq"]

    C2 = 1.0 - m6 / (E1 * E2)
    S2 = np.sqrt(np.maximum(1.0 - C2**2, 0.0))

    x4 = q0 * E4 + (Sq * S2 * np.cos(PHI2) + Cq * C2) * E2 * qvec
    x6 = E1 * E4 + E1 * E2 * C2

    return x4, x6


# ─── Phase-space Jacobian ──────────────────────────────────────────────────────


def trident_jacobian_full(
    Wc2: np.ndarray,
    ml1: float,
    E1: np.ndarray,
    E2: np.ndarray,
    x2: np.ndarray,
    u1_range: np.ndarray,
    exp_u1: np.ndarray,
    u2_range: np.ndarray,
    u3_range: np.ndarray,
    exp_u3: np.ndarray,
    x5_range: np.ndarray,
    u6_range: np.ndarray,
    jacob_u6: np.ndarray,
) -> np.ndarray:
    """
    Compute the full 8D integration Jacobian.

    Parameters match the C++ code's `Jacob` variable exactly.

    Parameters
    ----------
    Wc2 : array
    ml1 : float    charged lepton 1 mass
    E1, E2 : array  energies in Wc frame
    x2 : array
    u1_range : array  = u1_u - u1_l
    exp_u1 : array    = exp(u1)
    u2_range : array  = u2_u - u2_l   (= u2_u since u2_l=0)
    u3_range : array  = u3_u - u3_l
    exp_u3 : array    = exp(u3)
    x5_range : array  = x5_u - x5_l
    u6_range : array  = u6_u - u6_l
    jacob_u6 : array  Jacobian from u6 variable change

    Returns
    -------
    Jacob : np.ndarray
    """
    ps = (Wc2 - ml1**2) / (8.0 * Wc2 * E1 * E2 * x2 * (2.0 * np.pi) ** 6)
    jac = (
        ps
        * u1_range
        * exp_u1
        * u2_range
        * 0.5
        * u3_range
        * 0.5
        * exp_u3
        * (2.0 * np.pi)  # PHI2
        * x5_range
        * u6_range
        * jacob_u6
        * (2.0 * np.pi)  # x7
        * (2.0 * np.pi)  # x8
    )
    return jac


# ─── m6 mapping (BSM-mode dependent) ─────────────────────────────────────────


def _map_m6(u6_s, m6_max, mzprime, bsm_mode):
    """
    Map the unit variable ``u6_s`` to ``m6 ∈ [0, m6_max]``.

    The change of variables flattens the Z' propagator ``1/(2 m6 + M_Z'²)``
    for the interference (log map) and BSM-only (inverse map) modes, and is
    linear otherwise.

    Returns
    -------
    (m6, u6_range, jacob_u6, prop)
    """
    m6_l = 0.0
    m6_u = m6_max

    if bsm_mode == const.INTERFERENCE:
        u6_l = np.log(1.0 / (2.0 * m6_u + mzprime**2))
        u6_u = np.log(1.0 / (2.0 * m6_l + mzprime**2))
        u6 = u6_s * (u6_u - u6_l) + u6_l
        m6 = -(mzprime**2) / 2.0 + np.exp(-u6) / 2.0
        return m6, u6_u - u6_l, np.exp(-u6) / 2.0, -np.exp(u6)

    if bsm_mode == const.BSM_ONLY:
        u6_l = 1.0 / (2.0 * m6_u + mzprime**2)
        u6_u = 1.0 / (2.0 * m6_l + mzprime**2)
        u6 = u6_s * (u6_u - u6_l) + u6_l
        m6 = -(mzprime**2) / 2.0 + 1.0 / (2.0 * u6)
        return m6, u6_u - u6_l, 1.0 / (u6**2) / 2.0, -u6

    m6 = u6_s * (m6_u - m6_l)
    ones = np.ones_like(m6)
    if bsm_mode == const.SM_ONLY:
        prop = np.zeros_like(m6)
    else:  # SM_AND_BSM: linear map, propagator evaluated at amplitude level
        prop = 1.0 / (2.0 * m6 + mzprime**2)
    return m6, m6_u - m6_l, ones, prop


# ─── Full unit-cube → physical variable mapping ───────────────────────────────


def map_unit_to_physical(
    xvec: np.ndarray,
    Enu: float,
    ml1: float,
    ml2: float,
    Mn: float,
    mzprime: float = 0.0,
    bsm_mode: int = 0,
) -> dict:
    """
    Map unit-hypercube sample xvec (shape [N, 8]) to physical kinematic variables.

    This is the Python translation of CohFromIntToPhysical_Efixed() from
    cross_sections.cxx, covering all BSM modes for the m6 variable.

    Parameters
    ----------
    xvec : np.ndarray, shape (N, 8)
        Each row is [u1_s, u2_s, u3_s, PHI2_s, x5_s, u6_s, x7_s, x8_s]
        where all components are in [0, 1].
    Enu : float
        Neutrino energy in GeV.
    ml1, ml2 : float
        Lepton masses in GeV.
    Mn : float
        Target nucleon/nucleus mass in GeV.
    mzprime : float
        Z' mass in GeV (used only for non-SM_ONLY bsm_mode).
    bsm_mode : int
        SM_ONLY=0, INTERFERENCE=1, BSM_ONLY=2, SM_AND_BSM=3.

    Returns
    -------
    dict with keys:
        x1, x2, x3, x4, x5, x6, x7, x8,
        PHI2, m6,
        Jacob,
        prop,           (propagator factor; zeros for SM_ONLY)
        jacob_u6,
        valid,          (boolean mask: True where kinematics are physical)
        Wc2, E1, E2, E4, q0, qvec, Cq, Sq
    """
    xvec = np.atleast_2d(np.asarray(xvec, dtype=float))

    u1_s = xvec[:, 0]
    u2_s = xvec[:, 1]
    u3_s = xvec[:, 2]
    PHI2_s = xvec[:, 3]
    x5_s = xvec[:, 4]
    u6_s = xvec[:, 5]
    x7_s = xvec[:, 6]
    x8_s = xvec[:, 7]

    # ── x1 (Q²): log mapping ──────────────────────────────────────────────────
    x1_u = trident_Q2_upper(Enu, ml1, ml2, Mn)
    x1_l = trident_Q2_lower(Enu, ml1, ml2, Mn)
    u1_l = np.log(x1_l)
    u1_u = np.log(x1_u)
    u1 = u1_s * (u1_u - u1_l) + u1_l
    x1 = np.exp(u1)

    # ── x2: special variable change ───────────────────────────────────────────
    _, _, u2_u = trident_x2_limits(x1, Enu, ml1, ml2, Mn, x1_u, x1_l)
    u2 = u2_s * u2_u  # u2_l = 0
    x2 = 0.5 * (u2 + (ml1 + ml2) ** 2 + x1)

    # ── x5 ────────────────────────────────────────────────────────────────────
    x5_l, x5_u = trident_x5_limits(x1, x2, ml1, ml2)
    x5 = x5_s * (x5_u - x5_l) + x5_l

    # ── x3: log mapping ───────────────────────────────────────────────────────
    _, _, u3_l, u3_u = trident_x3_limits(x1, x2, x5, ml1, ml2)
    u3 = u3_s * (u3_u - u3_l) + u3_l
    x3 = 0.5 * (np.exp(u3) - x1)

    # ── frame variables ───────────────────────────────────────────────────────
    frame = trident_frame_variables(x1, x2, x3, x5, ml1, ml2)
    Wc2 = frame["Wc2"]
    E1 = frame["E1"]
    E2 = frame["E2"]
    E4 = frame["E4"]
    q0 = frame["q0"]
    qvec = frame["qvec"]
    Cq = frame["Cq"]
    Sq = frame["Sq"]
    m6_max = frame["m6_max"]

    # ── m6 and x6 variable: BSM mode dependent ────────────────────────────────
    m6, u6_range, jacob_u6, prop = _map_m6(u6_s, m6_max, mzprime, bsm_mode)

    # ── x4 and x6 ─────────────────────────────────────────────────────────────
    PHI2 = PHI2_s * 2.0 * np.pi
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x4, x6 = trident_x4_x6(m6, PHI2, frame)

    # ── rotation angles (do not affect |M|²) ──────────────────────────────────
    x7 = x7_s * 2.0 * np.pi
    x8 = x8_s * 2.0 * np.pi

    # ── Jacobian ──────────────────────────────────────────────────────────────
    Jacob = trident_jacobian_full(
        Wc2,
        ml1,
        E1,
        E2,
        x2,
        u1_u - u1_l,
        np.exp(u1),
        u2_u,
        u3_u - u3_l,
        np.exp(u3),
        x5_u - x5_l,
        u6_range,
        jacob_u6,
    )

    # ── validity mask (all kinematic variables must be physical) ──────────────
    valid = (
        (x1 > 0)
        & (x2 > 0)
        & (2.0 * x3 + x1 > 0)
        & (x5 > 0)
        & (x6 > 0)
        & (Wc2 > 0)
        & np.isfinite(Jacob)
        & (Jacob > 0)
        & (x5_u > x5_l)
        & (u3_u > u3_l)
        & (m6 >= 0)
        & (m6 <= m6_max)
    )

    return dict(
        x1=x1,
        x2=x2,
        x3=x3,
        x4=x4,
        x5=x5,
        x6=x6,
        x7=x7,
        x8=x8,
        PHI2=PHI2,
        m6=m6,
        Jacob=Jacob,
        prop=prop,
        jacob_u6=jacob_u6,
        valid=valid,
        Wc2=Wc2,
        E1=E1,
        E2=E2,
        E4=E4,
        q0=q0,
        qvec=qvec,
        Cq=Cq,
        Sq=Sq,
    )


def map_dis_unit_to_trident_physical(
    xvec: np.ndarray,
    Enu,
    ml1: float,
    ml2: float,
    Mn: float = const.m_AVG,
    Q2_min: float = 1.0,
    Q2_max: float = None,
    W2_min: float = 4.0,
    xB_min: float = 1e-6,
    mzprime: float = 0.0,
    bsm_mode: int = 0,
) -> dict:
    """
    Map a unit hypercube to the inclusive DIS trident variables.

    The first eight coordinates follow ``map_unit_to_physical`` with the
    elastic ``x2`` bound replaced by the DIS bound at fixed Bjorken ``xB``.
    Coordinate 1 is Bjorken ``xB``; the remaining leptonic variables are
    shifted by one index:

      0: log Q², 1: log xB, 2: x2, 3: x3, 4: PHI2,
      5: x5, 6: x6 mapping, 7: x7, 8: x8.

    ``Jacob`` includes the extra DIS factor ``d(P.q) = Q² dxB/(2 xB²)``.
    """
    xvec = np.atleast_2d(np.asarray(xvec, dtype=float))
    N = xvec.shape[0]
    if xvec.shape[1] < 9:
        raise ValueError("DIS trident phase-space map expects at least 9 variables")

    Enu = np.asarray(Enu, dtype=float)
    if Enu.ndim == 0:
        Enu = np.full(N, float(Enu))

    u1_s = xvec[:, 0]
    xB_s = xvec[:, 1]
    u2_s = xvec[:, 2]
    u3_s = xvec[:, 3]
    PHI2_s = xvec[:, 4]
    x5_s = xvec[:, 5]
    u6_s = xvec[:, 6]
    x7_s = xvec[:, 7]
    x8_s = xvec[:, 8]

    q2_floor = max(float(Q2_min), 1e-12)
    q2_upper = trident_Q2_upper(Enu, ml1, ml2, Mn)
    if Q2_max is not None:
        q2_upper = np.minimum(q2_upper, float(Q2_max))
    q2_range_valid = q2_upper > q2_floor
    q2_upper_safe = np.where(q2_range_valid, q2_upper, q2_floor * (1.0 + 1e-12))

    u1_l = np.log(q2_floor)
    u1_u = np.log(q2_upper_safe)
    u1 = u1_s * (u1_u - u1_l) + u1_l
    x1 = np.exp(u1)

    xB_l, xB_u = dis_trident_xB_limits(
        x1, Enu, ml1, ml2, Mn, W2_min=W2_min, xB_min=xB_min,
    )
    xB_range_valid = (xB_u > xB_l) & np.isfinite(xB_l) & np.isfinite(xB_u)
    xB_l_safe = np.where(xB_range_valid, xB_l, max(float(xB_min), 1e-12))
    xB_u_safe = np.where(
        xB_range_valid,
        xB_u,
        np.maximum(xB_l_safe * (1.0 + 1e-12), xB_l_safe + 1e-12),
    )
    log_xB_l = np.log(xB_l_safe)
    log_xB_u = np.log(xB_u_safe)
    log_xB = xB_s * (log_xB_u - log_xB_l) + log_xB_l
    xB = np.exp(log_xB)
    jac_xB = (log_xB_u - log_xB_l) * xB

    x2_u = dis_trident_x2_upper(x1, Enu, Mn, xB)
    u2_u = 2.0 * x2_u - (ml1 + ml2) ** 2 - x1
    u2_valid = u2_u > 0.0
    u2_u_safe = np.where(u2_valid, u2_u, 1e-300)
    u2 = u2_s * u2_u_safe
    x2 = 0.5 * (u2 + (ml1 + ml2) ** 2 + x1)

    x5_l, x5_u = trident_x5_limits(x1, x2, ml1, ml2)
    x5 = x5_s * (x5_u - x5_l) + x5_l

    _, _, u3_l, u3_u = trident_x3_limits(x1, x2, x5, ml1, ml2)
    u3_range = u3_u - u3_l
    u3_valid = u3_range > 0.0
    u3_range_safe = np.where(u3_valid, u3_range, 1e-300)
    u3 = u3_s * u3_range_safe + u3_l
    x3 = 0.5 * (np.exp(u3) - x1)

    frame = trident_frame_variables(x1, x2, x3, x5, ml1, ml2)
    Wc2 = frame["Wc2"]
    E1 = frame["E1"]
    E2 = frame["E2"]
    E4 = frame["E4"]
    q0 = frame["q0"]
    qvec = frame["qvec"]
    Cq = frame["Cq"]
    Sq = frame["Sq"]
    m6_max = frame["m6_max"]

    m6, u6_range, jacob_u6, prop = _map_m6(u6_s, m6_max, mzprime, bsm_mode)

    PHI2 = PHI2_s * 2.0 * np.pi
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x4, x6 = trident_x4_x6(m6, PHI2, frame)

    x7 = x7_s * 2.0 * np.pi
    x8 = x8_s * 2.0 * np.pi

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        Jacob = trident_jacobian_full(
            Wc2,
            ml1,
            E1,
            E2,
            x2,
            u1_u - u1_l,
            np.exp(u1),
            u2_u_safe,
            u3_range_safe,
            np.exp(u3),
            x5_u - x5_l,
            u6_range,
            jacob_u6,
        )

    dis_jacobian = x1 / (2.0 * xB**2)
    Jacob = Jacob * jac_xB * dis_jacobian
    Jacob = np.where(np.isfinite(Jacob), Jacob, 0.0)
    W2 = Mn**2 + x1 / xB - x1

    valid = (
        q2_range_valid
        & xB_range_valid
        & u2_valid
        & (x1 > 0.0)
        & (xB > 0.0)
        & (xB <= 1.0)
        & (W2 >= W2_min)
        & (x2 > 0.0)
        & (2.0 * x3 + x1 > 0.0)
        & (x5 > 0.0)
        & (x6 > 0.0)
        & (Wc2 > 0.0)
        & np.isfinite(Jacob)
        & (Jacob > 0.0)
        & (x5_u > x5_l)
        & u3_valid
        & (m6 >= 0.0)
        & (m6 <= m6_max)
    )

    return dict(
        x1=x1,
        x2=x2,
        x3=x3,
        x4=x4,
        x5=x5,
        x6=x6,
        x7=x7,
        x8=x8,
        xB=xB,
        W2=W2,
        PHI2=PHI2,
        m6=m6,
        Jacob=Jacob,
        prop=prop,
        jacob_u6=jacob_u6,
        valid=valid,
        Wc2=Wc2,
        E1=E1,
        E2=E2,
        E4=E4,
        q0=q0,
        qvec=qvec,
        Cq=Cq,
        Sq=Sq,
    )
