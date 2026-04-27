"""
amplitudes.py
=============
Matrix elements for neutrino trident production.

Two scattering regimes:

Coherent (nuclear):  nu + N -> nu + l- + l+ + N
  Phase space variables: x1=Q2, x2..x6 Lorentz invariants of lepton momenta.
  Nuclear target scatters coherently; form factor is Woods-Saxon FF_WS(q,A).
  Exact polynomial translated from cross_sections.cxx (CohFromIntToPhysical).

Diffractive (nucleon):  nu + gamma* -> nu + l- + l+
  Phase space variables: s, t, l (Lorentz invariants), theta, phi (dilepton angles).
  Form factor is Dirac dipole F_diffractive(q^2).
  Exact expression translated from dsigma_dPS_diff in cross_sections.cxx.

References:
  Czyz et al., Phys.Rev. 177 (1969) 2311
  MH Hostert C++ implementation in bsm_trident/MH_Code/clean_code/
"""

import numpy as np


def coherent_dsigma(
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    Enu,
    Mn,
    ml1,
    ml2,
    V2,
    A2,
    VA,
    Z,
    A,
    Diag11=1.0,
    Diag22=1.0,
    Diag12=1.0,
):
    """
    Differential cross section integrand for coherent trident production.

    Translated exactly from CohFromIntToPhysical() in cross_sections.cxx.
    Includes nuclear form factor FF_WS, alphaQED^2, Gf^2, and Z^2 factors.
    Returns dsigma BEFORE multiplying by the phase space Jacobian.

    Parameters
    ----------
    x1..x6 : float or ndarray
        Lorentz-invariant kinematic variables.
        x1 = Q^2; x2..x6 defined in phase_space.py.
    Enu, Mn : float
        Neutrino energy and nucleon mass [GeV].
    ml1, ml2 : float
        Lepton masses: ml1 = m(l-), ml2 = m(l+) [GeV].
    V2, A2, VA : float or ndarray
        Effective coupling coefficients V^2, A^2, V*A.
    Z, A : int
        Atomic and mass numbers of the nucleus.
    Diag11, Diag22, Diag12 : float
        Diagram selector flags (1.0=include, 0.0=exclude).
        Diag12=0 when l1 != l2 (no t-channel/u-channel interference).

    Returns
    -------
    float or ndarray
        dsigma [GeV^-2] before Jacobian multiplication.
    """
    from neptune.nuclear_tools import FF_WS
    from neptune.const import alphaQED, Gf

    FF2 = FF_WS(np.sqrt(np.maximum(x1, 0.0)), A) ** 2
    Z2 = float(Z) ** 2
    Gf2 = Gf**2
    Mn2 = Mn**2
    Enu2 = Enu**2
    # D1 = -(Enu^2*Mn*x1) + Enu*x1*x2 + Mn*x2^2
    D1 = -(Enu2 * Mn * x1) + Enu * x1 * x2 + Mn * x2**2
    # Xsq = (-2*Enu*Mn + x2)^2
    Xsq = (-2 * Enu * Mn + x2) ** 2

    dsigma = (
        8
        * (alphaQED**2)
        * (FF2)
        * Gf2
        * (
            Diag22
            * (x1 + 2 * x4) ** 2
            * (
                A2
                * (
                    -2
                    * ml2**2
                    * Mn
                    * x2**3
                    * D1
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    - 2
                    * Mn
                    * x2**3
                    * D1
                    * x3
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    + 2
                    * ml2**2
                    * Mn
                    * x2**2
                    * D1
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    * x5
                    - x1**2
                    * x2**2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 3
                    * Mn
                    * x1
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 2
                    * x1**2
                    * x2
                    * (Xsq)
                    * x5**2
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 4
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5**2
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - x1**2
                    * (Xsq)
                    * x5**3
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5**3
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 4 * ml1 * ml2**3 * Mn * x2**2 * D1 * (x2 - x5 - x6)
                    + 4 * ml1 * ml2 * Mn * x1 * x2**2 * D1 * (x2 - x5 - x6)
                    + 4 * ml1 * ml2 * Mn * x2**2 * D1 * x3 * (x2 - x5 - x6)
                    + 2 * ml1 * ml2 * x1**2 * x2 * (Xsq) * x5 * (x2 - x5 - x6)
                    - 4 * ml1 * ml2 * Mn * x1 * x2 * D1 * x5 * (x2 - x5 - x6)
                    - 2 * ml1 * ml2 * x1**2 * (Xsq) * x5**2 * (x2 - x5 - x6)
                    - 4 * ml1 * ml2 * Mn * x1 * (-D1) * x5**2 * (x2 - x5 - x6)
                    - 4 * ml2**2 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x6
                    - 4 * Mn * x2**2 * D1 * x3 * (-x1 + x2 - x3 - x4) * x6
                    - 2 * x1**2 * x2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6
                    + 4 * Mn * x1 * x2 * D1 * (x1 - x2 + x3 + x4) * x5 * x6
                    + 2 * x1**2 * (Xsq) * (x1 - x2 + x3 + x4) * x5**2 * x6
                    + 4 * Mn * x1 * (-D1) * (x1 - x2 + x3 + x4) * x5**2 * x6
                    + x1**3 * (Xsq) * x5 * (x2 - x5 - x6) * x6
                    + 2 * Mn * x1**2 * (-D1) * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * x3 * x5 * (x2 - x5 - x6) * x6
                    + 4 * Mn * x1 * (-D1) * x3 * x5 * (x2 - x5 - x6) * x6
                    - 2
                    * ml2**2
                    * Mn
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + Mn
                    * x1
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + x1**2
                    * x2
                    * (Xsq)
                    * x5
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - x1**2
                    * (Xsq)
                    * x5**2
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5**2
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                )
                + V2
                * (
                    -2
                    * ml2**2
                    * Mn
                    * x2**3
                    * D1
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    - 2
                    * Mn
                    * x2**3
                    * D1
                    * x3
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    + 2
                    * ml2**2
                    * Mn
                    * x2**2
                    * D1
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    * x5
                    - x1**2
                    * x2**2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 3
                    * Mn
                    * x1
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 2
                    * x1**2
                    * x2
                    * (Xsq)
                    * x5**2
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 4
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5**2
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - x1**2
                    * (Xsq)
                    * x5**3
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5**3
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 4 * ml1 * ml2**3 * Mn * x2**2 * D1 * (x2 - x5 - x6)
                    - 4 * ml1 * ml2 * Mn * x1 * x2**2 * D1 * (x2 - x5 - x6)
                    - 4 * ml1 * ml2 * Mn * x2**2 * D1 * x3 * (x2 - x5 - x6)
                    - 2 * ml1 * ml2 * x1**2 * x2 * (Xsq) * x5 * (x2 - x5 - x6)
                    + 4 * ml1 * ml2 * Mn * x1 * x2 * D1 * x5 * (x2 - x5 - x6)
                    + 2 * ml1 * ml2 * x1**2 * (Xsq) * x5**2 * (x2 - x5 - x6)
                    + 4 * ml1 * ml2 * Mn * x1 * (-D1) * x5**2 * (x2 - x5 - x6)
                    - 4 * ml2**2 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x6
                    - 4 * Mn * x2**2 * D1 * x3 * (-x1 + x2 - x3 - x4) * x6
                    - 2 * x1**2 * x2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6
                    + 4 * Mn * x1 * x2 * D1 * (x1 - x2 + x3 + x4) * x5 * x6
                    + 2 * x1**2 * (Xsq) * (x1 - x2 + x3 + x4) * x5**2 * x6
                    + 4 * Mn * x1 * (-D1) * (x1 - x2 + x3 + x4) * x5**2 * x6
                    + x1**3 * (Xsq) * x5 * (x2 - x5 - x6) * x6
                    + 2 * Mn * x1**2 * (-D1) * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * x3 * x5 * (x2 - x5 - x6) * x6
                    + 4 * Mn * x1 * (-D1) * x3 * x5 * (x2 - x5 - x6) * x6
                    - 2
                    * ml2**2
                    * Mn
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + Mn
                    * x1
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + x1**2
                    * x2
                    * (Xsq)
                    * x5
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - x1**2
                    * (Xsq)
                    * x5**2
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5**2
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                )
                + 2
                * VA
                * (
                    -2
                    * ml2**2
                    * Mn
                    * x2**3
                    * D1
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    - 2
                    * Mn
                    * x2**3
                    * D1
                    * x3
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    + 2
                    * ml2**2
                    * Mn
                    * x2**2
                    * D1
                    * (-(ml1**2) + ml2**2 - x1 + 2 * x2 - 2 * x3 - 2 * x5)
                    * x5
                    - x1**2
                    * x2**2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 3
                    * Mn
                    * x1
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 2
                    * x1**2
                    * x2
                    * (Xsq)
                    * x5**2
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 4
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5**2
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - x1**2
                    * (Xsq)
                    * x5**3
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5**3
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 4 * ml2**2 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x6
                    + 4 * Mn * x2**2 * D1 * x3 * (-x1 + x2 - x3 - x4) * x6
                    + 2 * x1**2 * x2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6
                    - 4 * Mn * x1 * x2 * D1 * (x1 - x2 + x3 + x4) * x5 * x6
                    - 2 * x1**2 * (Xsq) * (x1 - x2 + x3 + x4) * x5**2 * x6
                    - 4 * Mn * x1 * (-D1) * (x1 - x2 + x3 + x4) * x5**2 * x6
                    - x1**3 * (Xsq) * x5 * (x2 - x5 - x6) * x6
                    - 2 * Mn * x1**2 * (-D1) * x5 * (x2 - x5 - x6) * x6
                    - 2 * x1**2 * (Xsq) * x3 * x5 * (x2 - x5 - x6) * x6
                    - 4 * Mn * x1 * (-D1) * x3 * x5 * (x2 - x5 - x6) * x6
                    + Mn
                    * x1
                    * x2**2
                    * (-D1)
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 2
                    * ml2**2
                    * Mn
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - x1**2
                    * x2
                    * (Xsq)
                    * x5
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 2
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + x1**2
                    * (Xsq)
                    * x5**2
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5**2
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                )
            )
            + Diag11
            * (x1 + 2 * x3) ** 2
            * (
                A2
                * (
                    -4 * ml1**2 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x5
                    - 4 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x4 * x5
                    - 2
                    * ml1**2
                    * Mn
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + Mn
                    * x1
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 2
                    * ml1**2
                    * Mn
                    * x2**3
                    * D1
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    - 2
                    * Mn
                    * x2**3
                    * D1
                    * x4
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    - 4 * ml1**3 * ml2 * Mn * x2**2 * D1 * (x2 - x5 - x6)
                    + 4 * ml1 * ml2 * Mn * x1 * x2**2 * D1 * (x2 - x5 - x6)
                    + 4 * ml1 * ml2 * Mn * x2**2 * D1 * x4 * (x2 - x5 - x6)
                    - 2 * x1**2 * x2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6
                    + 4 * Mn * x1 * x2 * D1 * (x1 - x2 + x3 + x4) * x5 * x6
                    + x1**2
                    * x2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6
                    - 2
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6
                    + 2
                    * ml1**2
                    * Mn
                    * x2**2
                    * D1
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    * x6
                    + 2 * ml1 * ml2 * x1**2 * x2 * (Xsq) * (x2 - x5 - x6) * x6
                    - 4 * ml1 * ml2 * Mn * x1 * x2 * D1 * (x2 - x5 - x6) * x6
                    + x1**3 * (Xsq) * x5 * (x2 - x5 - x6) * x6
                    + 2 * Mn * x1**2 * (-D1) * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * x4 * x5 * (x2 - x5 - x6) * x6
                    + 4 * Mn * x1 * (-D1) * x4 * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6**2
                    + 4 * Mn * x1 * (-D1) * (x1 - x2 + x3 + x4) * x5 * x6**2
                    - x1**2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6**2
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6**2
                    - 2 * ml1 * ml2 * x1**2 * (Xsq) * (x2 - x5 - x6) * x6**2
                    - 4 * ml1 * ml2 * Mn * x1 * (-D1) * (x2 - x5 - x6) * x6**2
                    - x1**2
                    * x2**2
                    * (Xsq)
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 3
                    * Mn
                    * x1
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 2
                    * x1**2
                    * x2
                    * (Xsq)
                    * x6**2
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 4
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x6**2
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - x1**2
                    * (Xsq)
                    * x6**3
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x6**3
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                )
                + V2
                * (
                    -4 * ml1**2 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x5
                    - 4 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x4 * x5
                    - 2
                    * ml1**2
                    * Mn
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + Mn
                    * x1
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    - 2
                    * ml1**2
                    * Mn
                    * x2**3
                    * D1
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    - 2
                    * Mn
                    * x2**3
                    * D1
                    * x4
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    + 4 * ml1**3 * ml2 * Mn * x2**2 * D1 * (x2 - x5 - x6)
                    - 4 * ml1 * ml2 * Mn * x1 * x2**2 * D1 * (x2 - x5 - x6)
                    - 4 * ml1 * ml2 * Mn * x2**2 * D1 * x4 * (x2 - x5 - x6)
                    - 2 * x1**2 * x2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6
                    + 4 * Mn * x1 * x2 * D1 * (x1 - x2 + x3 + x4) * x5 * x6
                    + x1**2
                    * x2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6
                    - 2
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6
                    + 2
                    * ml1**2
                    * Mn
                    * x2**2
                    * D1
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    * x6
                    - 2 * ml1 * ml2 * x1**2 * x2 * (Xsq) * (x2 - x5 - x6) * x6
                    + 4 * ml1 * ml2 * Mn * x1 * x2 * D1 * (x2 - x5 - x6) * x6
                    + x1**3 * (Xsq) * x5 * (x2 - x5 - x6) * x6
                    + 2 * Mn * x1**2 * (-D1) * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * x4 * x5 * (x2 - x5 - x6) * x6
                    + 4 * Mn * x1 * (-D1) * x4 * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6**2
                    + 4 * Mn * x1 * (-D1) * (x1 - x2 + x3 + x4) * x5 * x6**2
                    - x1**2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6**2
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6**2
                    + 2 * ml1 * ml2 * x1**2 * (Xsq) * (x2 - x5 - x6) * x6**2
                    + 4 * ml1 * ml2 * Mn * x1 * (-D1) * (x2 - x5 - x6) * x6**2
                    - x1**2
                    * x2**2
                    * (Xsq)
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 3
                    * Mn
                    * x1
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 2
                    * x1**2
                    * x2
                    * (Xsq)
                    * x6**2
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 4
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x6**2
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - x1**2
                    * (Xsq)
                    * x6**3
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x6**3
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                )
                + 2
                * VA
                * (
                    -4 * ml1**2 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x5
                    - 4 * Mn * x2**2 * D1 * (-x1 + x2 - x3 - x4) * x4 * x5
                    - 2
                    * ml1**2
                    * Mn
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + Mn
                    * x1
                    * x2**2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    + 2
                    * ml1**2
                    * Mn
                    * x2**3
                    * D1
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    + 2
                    * Mn
                    * x2**3
                    * D1
                    * x4
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    - 2 * x1**2 * x2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6
                    + 4 * Mn * x1 * x2 * D1 * (x1 - x2 + x3 + x4) * x5 * x6
                    + x1**2
                    * x2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6
                    - 2
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6
                    - 2
                    * ml1**2
                    * Mn
                    * x2**2
                    * D1
                    * (ml1**2 - ml2**2 - x1 + 2 * x2 - 2 * x4 - 2 * x6)
                    * x6
                    + x1**3 * (Xsq) * x5 * (x2 - x5 - x6) * x6
                    + 2 * Mn * x1**2 * (-D1) * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * x4 * x5 * (x2 - x5 - x6) * x6
                    + 4 * Mn * x1 * (-D1) * x4 * x5 * (x2 - x5 - x6) * x6
                    + 2 * x1**2 * (Xsq) * (x1 - x2 + x3 + x4) * x5 * x6**2
                    + 4 * Mn * x1 * (-D1) * (x1 - x2 + x3 + x4) * x5 * x6**2
                    - x1**2
                    * (Xsq)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6**2
                    - 2
                    * Mn
                    * x1
                    * (-D1)
                    * x5
                    * (ml1**2 - ml2**2 + x1 - 2 * x2 + 2 * x3 + 2 * x5)
                    * x6**2
                    + x1**2
                    * x2**2
                    * (Xsq)
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 3
                    * Mn
                    * x1
                    * x2**2
                    * D1
                    * x6
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    - 2
                    * x1**2
                    * x2
                    * (Xsq)
                    * x6**2
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 4
                    * Mn
                    * x1
                    * x2
                    * D1
                    * x6**2
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + x1**2
                    * (Xsq)
                    * x6**3
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                    + 2
                    * Mn
                    * x1
                    * (-D1)
                    * x6**3
                    * (-(ml1**2) + ml2**2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)
                )
            )
            + Diag12
            * (x1 + 2 * x3)
            * (x1 + 2 * x4)
            * (
                2
                * Enu2
                * Mn2
                * x1
                * (
                    -2 * V2 * x1 * x2**4
                    + 4 * V2 * x2**5
                    + V2 * x1 * x2**3 * x3
                    + 2 * VA * x1 * x2**3 * x3
                    - 6 * V2 * x2**4 * x3
                    - 4 * VA * x2**4 * x3
                    + 2 * V2 * x2**3 * x3**2
                    + 4 * VA * x2**3 * x3**2
                    + V2 * x1 * x2**3 * x4
                    - 2 * VA * x1 * x2**3 * x4
                    - 6 * V2 * x2**4 * x4
                    + 4 * VA * x2**4 * x4
                    + 4 * V2 * x2**3 * x3 * x4
                    + 2 * V2 * x2**3 * x4**2
                    - 4 * VA * x2**3 * x4**2
                    + 4 * V2 * x1**2 * x2**2 * x5
                    + 8 * VA * x1**2 * x2**2 * x5
                    - 6 * V2 * x1 * x2**3 * x5
                    - 12 * VA * x1 * x2**3 * x5
                    - 8 * V2 * x2**4 * x5
                    - 8 * VA * x2**4 * x5
                    + 8 * V2 * x1 * x2**2 * x3 * x5
                    + 16 * VA * x1 * x2**2 * x3 * x5
                    + 6 * V2 * x2**3 * x3 * x5
                    + 12 * VA * x2**3 * x3 * x5
                    + 8 * V2 * x1 * x2**2 * x4 * x5
                    + 16 * VA * x1 * x2**2 * x4 * x5
                    + 6 * V2 * x2**3 * x4 * x5
                    - 4 * VA * x2**3 * x4 * x5
                    + 2 * V2 * x2**2 * x3 * x4 * x5
                    + 4 * VA * x2**2 * x3 * x4 * x5
                    - 2 * V2 * x2**2 * x4**2 * x5
                    + 4 * VA * x2**2 * x4**2 * x5
                    - 3 * V2 * x1**2 * x2 * x5**2
                    - 6 * VA * x1**2 * x2 * x5**2
                    + 12 * V2 * x1 * x2**2 * x5**2
                    + 24 * VA * x1 * x2**2 * x5**2
                    + 8 * V2 * x2**3 * x5**2
                    + 16 * VA * x2**3 * x5**2
                    - 6 * V2 * x1 * x2 * x3 * x5**2
                    - 12 * VA * x1 * x2 * x3 * x5**2
                    - 4 * V2 * x2**2 * x3 * x5**2
                    - 8 * VA * x2**2 * x3 * x5**2
                    - 12 * V2 * x1 * x2 * x4 * x5**2
                    - 24 * VA * x1 * x2 * x4 * x5**2
                    - 6 * V2 * x1 * x2 * x5**3
                    - 12 * VA * x1 * x2 * x5**3
                    - 4 * V2 * x2**2 * x5**3
                    - 8 * VA * x2**2 * x5**3
                    + 6 * V2 * x1 * x4 * x5**3
                    + 12 * VA * x1 * x4 * x5**3
                    + 4 * V2 * x1**2 * x2**2 * x6
                    - 8 * VA * x1**2 * x2**2 * x6
                    - 6 * V2 * x1 * x2**3 * x6
                    + 12 * VA * x1 * x2**3 * x6
                    - 8 * V2 * x2**4 * x6
                    + 8 * VA * x2**4 * x6
                    + 8 * V2 * x1 * x2**2 * x3 * x6
                    - 16 * VA * x1 * x2**2 * x3 * x6
                    + 6 * V2 * x2**3 * x3 * x6
                    + 4 * VA * x2**3 * x3 * x6
                    - 2 * V2 * x2**2 * x3**2 * x6
                    - 4 * VA * x2**2 * x3**2 * x6
                    + 8 * V2 * x1 * x2**2 * x4 * x6
                    - 16 * VA * x1 * x2**2 * x4 * x6
                    + 6 * V2 * x2**3 * x4 * x6
                    - 12 * VA * x2**3 * x4 * x6
                    + 2 * V2 * x2**2 * x3 * x4 * x6
                    - 4 * VA * x2**2 * x3 * x4 * x6
                    + 16 * V2 * x1 * x2**2 * x5 * x6
                    + 8 * V2 * x2**3 * x5 * x6
                    - 6 * V2 * x1 * x2 * x3 * x5 * x6
                    - 12 * VA * x1 * x2 * x3 * x5 * x6
                    - 4 * V2 * x2**2 * x3 * x5 * x6
                    - 8 * VA * x2**2 * x3 * x5 * x6
                    - 6 * V2 * x1 * x2 * x4 * x5 * x6
                    + 12 * VA * x1 * x2 * x4 * x5 * x6
                    - 4 * V2 * x2**2 * x4 * x5 * x6
                    + 8 * VA * x2**2 * x4 * x5 * x6
                    - 18 * V2 * x1 * x2 * x5**2 * x6
                    - 36 * VA * x1 * x2 * x5**2 * x6
                    - 4 * V2 * x2**2 * x5**2 * x6
                    - 8 * VA * x2**2 * x5**2 * x6
                    + 6 * V2 * x1 * x3 * x5**2 * x6
                    + 12 * VA * x1 * x3 * x5**2 * x6
                    + 12 * V2 * x1 * x5**3 * x6
                    + 24 * VA * x1 * x5**3 * x6
                    - 3 * V2 * x1**2 * x2 * x6**2
                    + 6 * VA * x1**2 * x2 * x6**2
                    + 12 * V2 * x1 * x2**2 * x6**2
                    - 24 * VA * x1 * x2**2 * x6**2
                    + 8 * V2 * x2**3 * x6**2
                    - 16 * VA * x2**3 * x6**2
                    - 12 * V2 * x1 * x2 * x3 * x6**2
                    + 24 * VA * x1 * x2 * x3 * x6**2
                    - 6 * V2 * x1 * x2 * x4 * x6**2
                    + 12 * VA * x1 * x2 * x4 * x6**2
                    - 4 * V2 * x2**2 * x4 * x6**2
                    + 8 * VA * x2**2 * x4 * x6**2
                    - 18 * V2 * x1 * x2 * x5 * x6**2
                    + 36 * VA * x1 * x2 * x5 * x6**2
                    - 4 * V2 * x2**2 * x5 * x6**2
                    + 8 * VA * x2**2 * x5 * x6**2
                    + 6 * V2 * x1 * x4 * x5 * x6**2
                    - 12 * VA * x1 * x4 * x5 * x6**2
                    - 6 * V2 * x1 * x2 * x6**3
                    + 12 * VA * x1 * x2 * x6**3
                    - 4 * V2 * x2**2 * x6**3
                    + 8 * VA * x2**2 * x6**3
                    + 6 * V2 * x1 * x3 * x6**3
                    - 12 * VA * x1 * x3 * x6**3
                    + 12 * V2 * x1 * x5 * x6**3
                    - 24 * VA * x1 * x5 * x6**3
                    + 2 * ml1**3 * ml2 * V2 * x2**2 * (-x2 + x5 + x6)
                    + ml2**4 * x2**2 * (2 * VA * (x2 - x5 - x6) + V2 * (-x5 + x6))
                    + ml1**4 * x2**2 * (V2 * (x5 - x6) + 2 * VA * (-x2 + x5 + x6))
                    - 2
                    * ml1
                    * ml2
                    * V2
                    * (
                        ml2**2 * x2**2 * (x2 - x5 - x6)
                        + x1
                        * (
                            x2**3
                            - 6 * x2**2 * (x5 + x6)
                            - 6 * x5 * x6 * (x5 + x6)
                            + 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                        )
                        + x2**2
                        * (
                            4 * x2**2
                            + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6))
                            - x2 * (3 * x3 + 3 * x4 + 4 * (x5 + x6))
                        )
                    )
                    - ml2**2
                    * (
                        V2
                        * (
                            -2 * x2**4
                            + x2**3 * (x3 + x4 + 4 * x5)
                            + x2**2
                            * (-2 * x5 * (x3 - x4 + 2 * x5) + x1 * (3 * x5 - x6))
                            + 6 * x1 * x5 * (x5 - x6) * x6
                            + 3 * x1 * x2 * (-(x5**2) + x6**2)
                        )
                        + 2
                        * VA
                        * (
                            x2**2
                            * (
                                -2 * x2**2
                                - 2 * (x3 + 2 * x5) * (x5 + x6)
                                + x2 * (3 * x3 + x4 + 6 * x5 + 2 * x6)
                            )
                            + x1
                            * (
                                x2**3
                                + 3 * x2**2 * (x5 + x6)
                                + 6 * x5 * x6 * (x5 + x6)
                                - 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                        )
                    )
                    + ml1**2
                    * (
                        V2
                        * (
                            2 * x2**4
                            + 6 * x1 * x5 * (x5 - x6) * x6
                            - x2**3 * (x3 + x4 + 4 * x6)
                            + 3 * x1 * x2 * (-(x5**2) + x6**2)
                            + x2**2
                            * (x1 * (x5 - 3 * x6) + 2 * x6 * (-x3 + x4 + 2 * x6))
                        )
                        + 2
                        * VA
                        * (
                            x2**2
                            * (
                                -2 * x2**2
                                - 2 * (x5 + x6) * (x4 + 2 * x6)
                                + x2 * (x3 + 3 * x4 + 2 * x5 + 6 * x6)
                            )
                            + x1
                            * (
                                x2**3
                                + 3 * x2**2 * (x5 + x6)
                                + 6 * x5 * x6 * (x5 + x6)
                                - 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                        )
                    )
                )
                - 2
                * Enu
                * Mn
                * x1
                * x2
                * (
                    -2 * V2 * x1 * x2**4
                    + 4 * V2 * x2**5
                    + V2 * x1 * x2**3 * x3
                    + 2 * VA * x1 * x2**3 * x3
                    - 6 * V2 * x2**4 * x3
                    - 4 * VA * x2**4 * x3
                    + 2 * V2 * x2**3 * x3**2
                    + 4 * VA * x2**3 * x3**2
                    + V2 * x1 * x2**3 * x4
                    - 2 * VA * x1 * x2**3 * x4
                    - 6 * V2 * x2**4 * x4
                    + 4 * VA * x2**4 * x4
                    + 4 * V2 * x2**3 * x3 * x4
                    + 2 * V2 * x2**3 * x4**2
                    - 4 * VA * x2**3 * x4**2
                    + 4 * V2 * x1**2 * x2**2 * x5
                    + 8 * VA * x1**2 * x2**2 * x5
                    - 6 * V2 * x1 * x2**3 * x5
                    - 12 * VA * x1 * x2**3 * x5
                    - 8 * V2 * x2**4 * x5
                    - 8 * VA * x2**4 * x5
                    + 8 * V2 * x1 * x2**2 * x3 * x5
                    + 16 * VA * x1 * x2**2 * x3 * x5
                    + 6 * V2 * x2**3 * x3 * x5
                    + 12 * VA * x2**3 * x3 * x5
                    + 8 * V2 * x1 * x2**2 * x4 * x5
                    + 16 * VA * x1 * x2**2 * x4 * x5
                    + 6 * V2 * x2**3 * x4 * x5
                    - 4 * VA * x2**3 * x4 * x5
                    + 2 * V2 * x2**2 * x3 * x4 * x5
                    + 4 * VA * x2**2 * x3 * x4 * x5
                    - 2 * V2 * x2**2 * x4**2 * x5
                    + 4 * VA * x2**2 * x4**2 * x5
                    - 3 * V2 * x1**2 * x2 * x5**2
                    - 6 * VA * x1**2 * x2 * x5**2
                    + 12 * V2 * x1 * x2**2 * x5**2
                    + 24 * VA * x1 * x2**2 * x5**2
                    + 8 * V2 * x2**3 * x5**2
                    + 16 * VA * x2**3 * x5**2
                    - 6 * V2 * x1 * x2 * x3 * x5**2
                    - 12 * VA * x1 * x2 * x3 * x5**2
                    - 4 * V2 * x2**2 * x3 * x5**2
                    - 8 * VA * x2**2 * x3 * x5**2
                    - 12 * V2 * x1 * x2 * x4 * x5**2
                    - 24 * VA * x1 * x2 * x4 * x5**2
                    - 6 * V2 * x1 * x2 * x5**3
                    - 12 * VA * x1 * x2 * x5**3
                    - 4 * V2 * x2**2 * x5**3
                    - 8 * VA * x2**2 * x5**3
                    + 6 * V2 * x1 * x4 * x5**3
                    + 12 * VA * x1 * x4 * x5**3
                    + 4 * V2 * x1**2 * x2**2 * x6
                    - 8 * VA * x1**2 * x2**2 * x6
                    - 6 * V2 * x1 * x2**3 * x6
                    + 12 * VA * x1 * x2**3 * x6
                    - 8 * V2 * x2**4 * x6
                    + 8 * VA * x2**4 * x6
                    + 8 * V2 * x1 * x2**2 * x3 * x6
                    - 16 * VA * x1 * x2**2 * x3 * x6
                    + 6 * V2 * x2**3 * x3 * x6
                    + 4 * VA * x2**3 * x3 * x6
                    - 2 * V2 * x2**2 * x3**2 * x6
                    - 4 * VA * x2**2 * x3**2 * x6
                    + 8 * V2 * x1 * x2**2 * x4 * x6
                    - 16 * VA * x1 * x2**2 * x4 * x6
                    + 6 * V2 * x2**3 * x4 * x6
                    - 12 * VA * x2**3 * x4 * x6
                    + 2 * V2 * x2**2 * x3 * x4 * x6
                    - 4 * VA * x2**2 * x3 * x4 * x6
                    + 16 * V2 * x1 * x2**2 * x5 * x6
                    + 8 * V2 * x2**3 * x5 * x6
                    - 6 * V2 * x1 * x2 * x3 * x5 * x6
                    - 12 * VA * x1 * x2 * x3 * x5 * x6
                    - 4 * V2 * x2**2 * x3 * x5 * x6
                    - 8 * VA * x2**2 * x3 * x5 * x6
                    - 6 * V2 * x1 * x2 * x4 * x5 * x6
                    + 12 * VA * x1 * x2 * x4 * x5 * x6
                    - 4 * V2 * x2**2 * x4 * x5 * x6
                    + 8 * VA * x2**2 * x4 * x5 * x6
                    - 18 * V2 * x1 * x2 * x5**2 * x6
                    - 36 * VA * x1 * x2 * x5**2 * x6
                    - 4 * V2 * x2**2 * x5**2 * x6
                    - 8 * VA * x2**2 * x5**2 * x6
                    + 6 * V2 * x1 * x3 * x5**2 * x6
                    + 12 * VA * x1 * x3 * x5**2 * x6
                    + 12 * V2 * x1 * x5**3 * x6
                    + 24 * VA * x1 * x5**3 * x6
                    - 3 * V2 * x1**2 * x2 * x6**2
                    + 6 * VA * x1**2 * x2 * x6**2
                    + 12 * V2 * x1 * x2**2 * x6**2
                    - 24 * VA * x1 * x2**2 * x6**2
                    + 8 * V2 * x2**3 * x6**2
                    - 16 * VA * x2**3 * x6**2
                    - 12 * V2 * x1 * x2 * x3 * x6**2
                    + 24 * VA * x1 * x2 * x3 * x6**2
                    - 6 * V2 * x1 * x2 * x4 * x6**2
                    + 12 * VA * x1 * x2 * x4 * x6**2
                    - 4 * V2 * x2**2 * x4 * x6**2
                    + 8 * VA * x2**2 * x4 * x6**2
                    - 18 * V2 * x1 * x2 * x5 * x6**2
                    + 36 * VA * x1 * x2 * x5 * x6**2
                    - 4 * V2 * x2**2 * x5 * x6**2
                    + 8 * VA * x2**2 * x5 * x6**2
                    + 6 * V2 * x1 * x4 * x5 * x6**2
                    - 12 * VA * x1 * x4 * x5 * x6**2
                    - 6 * V2 * x1 * x2 * x6**3
                    + 12 * VA * x1 * x2 * x6**3
                    - 4 * V2 * x2**2 * x6**3
                    + 8 * VA * x2**2 * x6**3
                    + 6 * V2 * x1 * x3 * x6**3
                    - 12 * VA * x1 * x3 * x6**3
                    + 12 * V2 * x1 * x5 * x6**3
                    - 24 * VA * x1 * x5 * x6**3
                    + 2 * ml1**3 * ml2 * V2 * x2**2 * (-x2 + x5 + x6)
                    + ml2**4 * x2**2 * (2 * VA * (x2 - x5 - x6) + V2 * (-x5 + x6))
                    + ml1**4 * x2**2 * (V2 * (x5 - x6) + 2 * VA * (-x2 + x5 + x6))
                    - 2
                    * ml1
                    * ml2
                    * V2
                    * (
                        ml2**2 * x2**2 * (x2 - x5 - x6)
                        + x1
                        * (
                            x2**3
                            - 6 * x2**2 * (x5 + x6)
                            - 6 * x5 * x6 * (x5 + x6)
                            + 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                        )
                        + x2**2
                        * (
                            4 * x2**2
                            + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6))
                            - x2 * (3 * x3 + 3 * x4 + 4 * (x5 + x6))
                        )
                    )
                    - ml2**2
                    * (
                        V2
                        * (
                            -2 * x2**4
                            + x2**3 * (x3 + x4 + 4 * x5)
                            + x2**2
                            * (-2 * x5 * (x3 - x4 + 2 * x5) + x1 * (3 * x5 - x6))
                            + 6 * x1 * x5 * (x5 - x6) * x6
                            + 3 * x1 * x2 * (-(x5**2) + x6**2)
                        )
                        + 2
                        * VA
                        * (
                            x2**2
                            * (
                                -2 * x2**2
                                - 2 * (x3 + 2 * x5) * (x5 + x6)
                                + x2 * (3 * x3 + x4 + 6 * x5 + 2 * x6)
                            )
                            + x1
                            * (
                                x2**3
                                + 3 * x2**2 * (x5 + x6)
                                + 6 * x5 * x6 * (x5 + x6)
                                - 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                        )
                    )
                    + ml1**2
                    * (
                        V2
                        * (
                            2 * x2**4
                            + 6 * x1 * x5 * (x5 - x6) * x6
                            - x2**3 * (x3 + x4 + 4 * x6)
                            + 3 * x1 * x2 * (-(x5**2) + x6**2)
                            + x2**2
                            * (x1 * (x5 - 3 * x6) + 2 * x6 * (-x3 + x4 + 2 * x6))
                        )
                        + 2
                        * VA
                        * (
                            x2**2
                            * (
                                -2 * x2**2
                                - 2 * (x5 + x6) * (x4 + 2 * x6)
                                + x2 * (x3 + 3 * x4 + 2 * x5 + 6 * x6)
                            )
                            + x1
                            * (
                                x2**3
                                + 3 * x2**2 * (x5 + x6)
                                + 6 * x5 * x6 * (x5 + x6)
                                - 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                        )
                    )
                )
                + x2**2
                * (
                    4 * Mn2 * V2 * x1 * x2**4
                    - 8 * Mn2 * V2 * x2**5
                    - 2 * Mn2 * V2 * x1 * x2**3 * x3
                    - 4 * Mn2 * VA * x1 * x2**3 * x3
                    + 12 * Mn2 * V2 * x2**4 * x3
                    + 8 * Mn2 * VA * x2**4 * x3
                    - 4 * Mn2 * V2 * x2**3 * x3**2
                    - 8 * Mn2 * VA * x2**3 * x3**2
                    - 2 * Mn2 * V2 * x1 * x2**3 * x4
                    + 4 * Mn2 * VA * x1 * x2**3 * x4
                    + 12 * Mn2 * V2 * x2**4 * x4
                    - 8 * Mn2 * VA * x2**4 * x4
                    - 8 * Mn2 * V2 * x2**3 * x3 * x4
                    - 4 * Mn2 * V2 * x2**3 * x4**2
                    + 8 * Mn2 * VA * x2**3 * x4**2
                    - 4 * Mn2 * V2 * x1**2 * x2**2 * x5
                    - 8 * Mn2 * VA * x1**2 * x2**2 * x5
                    + V2 * x1**3 * x2**2 * x5
                    + 2 * VA * x1**3 * x2**2 * x5
                    + 4 * Mn2 * V2 * x1 * x2**3 * x5
                    + 8 * Mn2 * VA * x1 * x2**3 * x5
                    - 2 * V2 * x1**2 * x2**3 * x5
                    - 4 * VA * x1**2 * x2**3 * x5
                    + 16 * Mn2 * V2 * x2**4 * x5
                    + 16 * Mn2 * VA * x2**4 * x5
                    - 8 * Mn2 * V2 * x1 * x2**2 * x3 * x5
                    - 16 * Mn2 * VA * x1 * x2**2 * x3 * x5
                    + 2 * V2 * x1**2 * x2**2 * x3 * x5
                    + 4 * VA * x1**2 * x2**2 * x3 * x5
                    - 12 * Mn2 * V2 * x2**3 * x3 * x5
                    - 24 * Mn2 * VA * x2**3 * x3 * x5
                    - 8 * Mn2 * V2 * x1 * x2**2 * x4 * x5
                    - 16 * Mn2 * VA * x1 * x2**2 * x4 * x5
                    + 2 * V2 * x1**2 * x2**2 * x4 * x5
                    + 4 * VA * x1**2 * x2**2 * x4 * x5
                    - 12 * Mn2 * V2 * x2**3 * x4 * x5
                    + 8 * Mn2 * VA * x2**3 * x4 * x5
                    - 4 * Mn2 * V2 * x2**2 * x3 * x4 * x5
                    - 8 * Mn2 * VA * x2**2 * x3 * x4 * x5
                    + 4 * Mn2 * V2 * x2**2 * x4**2 * x5
                    - 8 * Mn2 * VA * x2**2 * x4**2 * x5
                    + 2 * Mn2 * V2 * x1**2 * x2 * x5**2
                    + 4 * Mn2 * VA * x1**2 * x2 * x5**2
                    - V2 * x1**3 * x2 * x5**2
                    - 2 * VA * x1**3 * x2 * x5**2
                    - 8 * Mn2 * V2 * x1 * x2**2 * x5**2
                    - 16 * Mn2 * VA * x1 * x2**2 * x5**2
                    + 4 * V2 * x1**2 * x2**2 * x5**2
                    + 8 * VA * x1**2 * x2**2 * x5**2
                    - 16 * Mn2 * V2 * x2**3 * x5**2
                    - 32 * Mn2 * VA * x2**3 * x5**2
                    + 4 * Mn2 * V2 * x1 * x2 * x3 * x5**2
                    + 8 * Mn2 * VA * x1 * x2 * x3 * x5**2
                    - 2 * V2 * x1**2 * x2 * x3 * x5**2
                    - 4 * VA * x1**2 * x2 * x3 * x5**2
                    + 8 * Mn2 * V2 * x2**2 * x3 * x5**2
                    + 16 * Mn2 * VA * x2**2 * x3 * x5**2
                    + 8 * Mn2 * V2 * x1 * x2 * x4 * x5**2
                    + 16 * Mn2 * VA * x1 * x2 * x4 * x5**2
                    - 4 * V2 * x1**2 * x2 * x4 * x5**2
                    - 8 * VA * x1**2 * x2 * x4 * x5**2
                    + 4 * Mn2 * V2 * x1 * x2 * x5**3
                    + 8 * Mn2 * VA * x1 * x2 * x5**3
                    - 2 * V2 * x1**2 * x2 * x5**3
                    - 4 * VA * x1**2 * x2 * x5**3
                    + 8 * Mn2 * V2 * x2**2 * x5**3
                    + 16 * Mn2 * VA * x2**2 * x5**3
                    - 4 * Mn2 * V2 * x1 * x4 * x5**3
                    - 8 * Mn2 * VA * x1 * x4 * x5**3
                    + 2 * V2 * x1**2 * x4 * x5**3
                    + 4 * VA * x1**2 * x4 * x5**3
                    + 4 * ml1**3 * ml2 * Mn2 * V2 * x2**2 * (x2 - x5 - x6)
                    - 4 * Mn2 * V2 * x1**2 * x2**2 * x6
                    + 8 * Mn2 * VA * x1**2 * x2**2 * x6
                    + V2 * x1**3 * x2**2 * x6
                    - 2 * VA * x1**3 * x2**2 * x6
                    + 4 * Mn2 * V2 * x1 * x2**3 * x6
                    - 8 * Mn2 * VA * x1 * x2**3 * x6
                    - 2 * V2 * x1**2 * x2**3 * x6
                    + 4 * VA * x1**2 * x2**3 * x6
                    + 16 * Mn2 * V2 * x2**4 * x6
                    - 16 * Mn2 * VA * x2**4 * x6
                    - 8 * Mn2 * V2 * x1 * x2**2 * x3 * x6
                    + 16 * Mn2 * VA * x1 * x2**2 * x3 * x6
                    + 2 * V2 * x1**2 * x2**2 * x3 * x6
                    - 4 * VA * x1**2 * x2**2 * x3 * x6
                    - 12 * Mn2 * V2 * x2**3 * x3 * x6
                    - 8 * Mn2 * VA * x2**3 * x3 * x6
                    + 4 * Mn2 * V2 * x2**2 * x3**2 * x6
                    + 8 * Mn2 * VA * x2**2 * x3**2 * x6
                    - 8 * Mn2 * V2 * x1 * x2**2 * x4 * x6
                    + 16 * Mn2 * VA * x1 * x2**2 * x4 * x6
                    + 2 * V2 * x1**2 * x2**2 * x4 * x6
                    - 4 * VA * x1**2 * x2**2 * x4 * x6
                    - 12 * Mn2 * V2 * x2**3 * x4 * x6
                    + 24 * Mn2 * VA * x2**3 * x4 * x6
                    - 4 * Mn2 * V2 * x2**2 * x3 * x4 * x6
                    + 8 * Mn2 * VA * x2**2 * x3 * x4 * x6
                    - 16 * Mn2 * V2 * x1 * x2**2 * x5 * x6
                    + 4 * V2 * x1**2 * x2**2 * x5 * x6
                    - 16 * Mn2 * V2 * x2**3 * x5 * x6
                    + 4 * Mn2 * V2 * x1 * x2 * x3 * x5 * x6
                    + 8 * Mn2 * VA * x1 * x2 * x3 * x5 * x6
                    - 2 * V2 * x1**2 * x2 * x3 * x5 * x6
                    - 4 * VA * x1**2 * x2 * x3 * x5 * x6
                    + 8 * Mn2 * V2 * x2**2 * x3 * x5 * x6
                    + 16 * Mn2 * VA * x2**2 * x3 * x5 * x6
                    + 4 * Mn2 * V2 * x1 * x2 * x4 * x5 * x6
                    - 8 * Mn2 * VA * x1 * x2 * x4 * x5 * x6
                    - 2 * V2 * x1**2 * x2 * x4 * x5 * x6
                    + 4 * VA * x1**2 * x2 * x4 * x5 * x6
                    + 8 * Mn2 * V2 * x2**2 * x4 * x5 * x6
                    - 16 * Mn2 * VA * x2**2 * x4 * x5 * x6
                    + 12 * Mn2 * V2 * x1 * x2 * x5**2 * x6
                    + 24 * Mn2 * VA * x1 * x2 * x5**2 * x6
                    - 6 * V2 * x1**2 * x2 * x5**2 * x6
                    - 12 * VA * x1**2 * x2 * x5**2 * x6
                    + 8 * Mn2 * V2 * x2**2 * x5**2 * x6
                    + 16 * Mn2 * VA * x2**2 * x5**2 * x6
                    - 4 * Mn2 * V2 * x1 * x3 * x5**2 * x6
                    - 8 * Mn2 * VA * x1 * x3 * x5**2 * x6
                    + 2 * V2 * x1**2 * x3 * x5**2 * x6
                    + 4 * VA * x1**2 * x3 * x5**2 * x6
                    - 8 * Mn2 * V2 * x1 * x5**3 * x6
                    - 16 * Mn2 * VA * x1 * x5**3 * x6
                    + 4 * V2 * x1**2 * x5**3 * x6
                    + 8 * VA * x1**2 * x5**3 * x6
                    + 2 * Mn2 * V2 * x1**2 * x2 * x6**2
                    - 4 * Mn2 * VA * x1**2 * x2 * x6**2
                    - V2 * x1**3 * x2 * x6**2
                    + 2 * VA * x1**3 * x2 * x6**2
                    - 8 * Mn2 * V2 * x1 * x2**2 * x6**2
                    + 16 * Mn2 * VA * x1 * x2**2 * x6**2
                    + 4 * V2 * x1**2 * x2**2 * x6**2
                    - 8 * VA * x1**2 * x2**2 * x6**2
                    - 16 * Mn2 * V2 * x2**3 * x6**2
                    + 32 * Mn2 * VA * x2**3 * x6**2
                    + 8 * Mn2 * V2 * x1 * x2 * x3 * x6**2
                    - 16 * Mn2 * VA * x1 * x2 * x3 * x6**2
                    - 4 * V2 * x1**2 * x2 * x3 * x6**2
                    + 8 * VA * x1**2 * x2 * x3 * x6**2
                    + 4 * Mn2 * V2 * x1 * x2 * x4 * x6**2
                    - 8 * Mn2 * VA * x1 * x2 * x4 * x6**2
                    - 2 * V2 * x1**2 * x2 * x4 * x6**2
                    + 4 * VA * x1**2 * x2 * x4 * x6**2
                    + 8 * Mn2 * V2 * x2**2 * x4 * x6**2
                    - 16 * Mn2 * VA * x2**2 * x4 * x6**2
                    + 12 * Mn2 * V2 * x1 * x2 * x5 * x6**2
                    - 24 * Mn2 * VA * x1 * x2 * x5 * x6**2
                    - 6 * V2 * x1**2 * x2 * x5 * x6**2
                    + 12 * VA * x1**2 * x2 * x5 * x6**2
                    + 8 * Mn2 * V2 * x2**2 * x5 * x6**2
                    - 16 * Mn2 * VA * x2**2 * x5 * x6**2
                    - 4 * Mn2 * V2 * x1 * x4 * x5 * x6**2
                    + 8 * Mn2 * VA * x1 * x4 * x5 * x6**2
                    + 2 * V2 * x1**2 * x4 * x5 * x6**2
                    - 4 * VA * x1**2 * x4 * x5 * x6**2
                    + 4 * Mn2 * V2 * x1 * x2 * x6**3
                    - 8 * Mn2 * VA * x1 * x2 * x6**3
                    - 2 * V2 * x1**2 * x2 * x6**3
                    + 4 * VA * x1**2 * x2 * x6**3
                    + 8 * Mn2 * V2 * x2**2 * x6**3
                    - 16 * Mn2 * VA * x2**2 * x6**3
                    - 4 * Mn2 * V2 * x1 * x3 * x6**3
                    + 8 * Mn2 * VA * x1 * x3 * x6**3
                    + 2 * V2 * x1**2 * x3 * x6**3
                    - 4 * VA * x1**2 * x3 * x6**3
                    - 8 * Mn2 * V2 * x1 * x5 * x6**3
                    + 16 * Mn2 * VA * x1 * x5 * x6**3
                    + 4 * V2 * x1**2 * x5 * x6**3
                    - 8 * VA * x1**2 * x5 * x6**3
                    + 2
                    * ml1**4
                    * Mn2
                    * x2**2
                    * (2 * VA * (x2 - x5 - x6) + V2 * (-x5 + x6))
                    + 2
                    * ml2**4
                    * Mn2
                    * x2**2
                    * (V2 * (x5 - x6) + 2 * VA * (-x2 + x5 + x6))
                    + 2
                    * ml1
                    * ml2
                    * V2
                    * (
                        2 * ml2**2 * Mn2 * x2**2 * (x2 - x5 - x6)
                        - x1**2
                        * (
                            x2**3
                            - 2 * x2**2 * (x5 + x6)
                            - 2 * x5 * x6 * (x5 + x6)
                            + x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                        )
                        + 2
                        * Mn2
                        * (
                            -(
                                x1
                                * (
                                    x2**3
                                    + 2 * x2**2 * (x5 + x6)
                                    + 2 * x5 * x6 * (x5 + x6)
                                    - x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                                )
                            )
                            + x2**2
                            * (
                                4 * x2**2
                                + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6))
                                - x2 * (3 * x3 + 3 * x4 + 4 * (x5 + x6))
                            )
                        )
                    )
                    + ml2**2
                    * (
                        x1**2
                        * (
                            -(V2 * (x5 - x6) * (x2**2 + 2 * x5 * x6 - x2 * (x5 + x6)))
                            - 2
                            * VA
                            * (
                                x2**2 * (x5 + x6)
                                + 2 * x5 * x6 * (x5 + x6)
                                - x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                        )
                        + 2
                        * Mn2
                        * (
                            V2
                            * (
                                -2 * x2**4
                                + x2**3 * (x3 + x4 + 4 * x5)
                                + 2 * x1 * x5 * (x5 - x6) * x6
                                + x1 * x2 * (-(x5**2) + x6**2)
                                + x2**2
                                * (-2 * x5 * (x3 - x4 + 2 * x5) + x1 * (x5 + x6))
                            )
                            + 2
                            * VA
                            * (
                                x2**2
                                * (
                                    -2 * x2**2
                                    - 2 * (x3 + 2 * x5) * (x5 + x6)
                                    + x2 * (3 * x3 + x4 + 6 * x5 + 2 * x6)
                                )
                                + x1
                                * (
                                    x2**3
                                    + x2**2 * (x5 + x6)
                                    + 2 * x5 * x6 * (x5 + x6)
                                    - x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                                )
                            )
                        )
                    )
                    + ml1**2
                    * (
                        x1**2
                        * (
                            V2 * (x5 - x6) * (x2**2 + 2 * x5 * x6 - x2 * (x5 + x6))
                            + 2
                            * VA
                            * (
                                x2**2 * (x5 + x6)
                                + 2 * x5 * x6 * (x5 + x6)
                                - x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                        )
                        - 2
                        * Mn2
                        * (
                            V2
                            * (
                                2 * x2**4
                                + 2 * x1 * x5 * (x5 - x6) * x6
                                - x2**3 * (x3 + x4 + 4 * x6)
                                + x1 * x2 * (-(x5**2) + x6**2)
                                - x2**2 * (2 * (x3 - x4 - 2 * x6) * x6 + x1 * (x5 + x6))
                            )
                            + 2
                            * VA
                            * (
                                x2**2
                                * (
                                    -2 * x2**2
                                    - 2 * (x5 + x6) * (x4 + 2 * x6)
                                    + x2 * (x3 + 3 * x4 + 2 * x5 + 6 * x6)
                                )
                                + x1
                                * (
                                    x2**3
                                    + x2**2 * (x5 + x6)
                                    + 2 * x5 * x6 * (x5 + x6)
                                    - x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                                )
                            )
                        )
                    )
                )
                + A2
                * (
                    2
                    * Enu2
                    * Mn2
                    * x1
                    * (
                        -2 * x1 * x2**4
                        + 4 * x2**5
                        + x1 * x2**3 * x3
                        - 6 * x2**4 * x3
                        + 2 * x2**3 * x3**2
                        + x1 * x2**3 * x4
                        - 6 * x2**4 * x4
                        + 4 * x2**3 * x3 * x4
                        + 2 * x2**3 * x4**2
                        + 4 * x1**2 * x2**2 * x5
                        - 6 * x1 * x2**3 * x5
                        - 8 * x2**4 * x5
                        + 8 * x1 * x2**2 * x3 * x5
                        + 6 * x2**3 * x3 * x5
                        + 8 * x1 * x2**2 * x4 * x5
                        + 6 * x2**3 * x4 * x5
                        + 2 * x2**2 * x3 * x4 * x5
                        - 2 * x2**2 * x4**2 * x5
                        - 3 * x1**2 * x2 * x5**2
                        + 12 * x1 * x2**2 * x5**2
                        + 8 * x2**3 * x5**2
                        - 6 * x1 * x2 * x3 * x5**2
                        - 4 * x2**2 * x3 * x5**2
                        - 12 * x1 * x2 * x4 * x5**2
                        - 6 * x1 * x2 * x5**3
                        - 4 * x2**2 * x5**3
                        + 6 * x1 * x4 * x5**3
                        + 2 * ml1**3 * ml2 * x2**2 * (x2 - x5 - x6)
                        + ml1**4 * x2**2 * (x5 - x6)
                        + 4 * x1**2 * x2**2 * x6
                        - 6 * x1 * x2**3 * x6
                        - 8 * x2**4 * x6
                        + 8 * x1 * x2**2 * x3 * x6
                        + 6 * x2**3 * x3 * x6
                        - 2 * x2**2 * x3**2 * x6
                        + 8 * x1 * x2**2 * x4 * x6
                        + 6 * x2**3 * x4 * x6
                        + 2 * x2**2 * x3 * x4 * x6
                        + 16 * x1 * x2**2 * x5 * x6
                        + 8 * x2**3 * x5 * x6
                        - 6 * x1 * x2 * x3 * x5 * x6
                        - 4 * x2**2 * x3 * x5 * x6
                        - 6 * x1 * x2 * x4 * x5 * x6
                        - 4 * x2**2 * x4 * x5 * x6
                        - 18 * x1 * x2 * x5**2 * x6
                        - 4 * x2**2 * x5**2 * x6
                        + 6 * x1 * x3 * x5**2 * x6
                        + 12 * x1 * x5**3 * x6
                        - 3 * x1**2 * x2 * x6**2
                        + 12 * x1 * x2**2 * x6**2
                        + 8 * x2**3 * x6**2
                        - 12 * x1 * x2 * x3 * x6**2
                        - 6 * x1 * x2 * x4 * x6**2
                        - 4 * x2**2 * x4 * x6**2
                        - 18 * x1 * x2 * x5 * x6**2
                        - 4 * x2**2 * x5 * x6**2
                        + 6 * x1 * x4 * x5 * x6**2
                        - 6 * x1 * x2 * x6**3
                        - 4 * x2**2 * x6**3
                        + 6 * x1 * x3 * x6**3
                        + 12 * x1 * x5 * x6**3
                        + ml2**4 * x2**2 * (-x5 + x6)
                        + ml2**2
                        * (
                            2 * x2**4
                            - x2**3 * (x3 + x4 + 4 * x5)
                            + 6 * x1 * x5 * x6 * (-x5 + x6)
                            + 3 * x1 * x2 * (x5**2 - x6**2)
                            + x2**2
                            * (2 * x5 * (x3 - x4 + 2 * x5) + x1 * (-3 * x5 + x6))
                        )
                        + ml1**2
                        * (
                            2 * x2**4
                            + 6 * x1 * x5 * (x5 - x6) * x6
                            - x2**3 * (x3 + x4 + 4 * x6)
                            + 3 * x1 * x2 * (-(x5**2) + x6**2)
                            + x2**2
                            * (x1 * (x5 - 3 * x6) + 2 * x6 * (-x3 + x4 + 2 * x6))
                        )
                        + 2
                        * ml1
                        * ml2
                        * (
                            ml2**2 * x2**2 * (x2 - x5 - x6)
                            + x1
                            * (
                                x2**3
                                - 6 * x2**2 * (x5 + x6)
                                - 6 * x5 * x6 * (x5 + x6)
                                + 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                            + x2**2
                            * (
                                4 * x2**2
                                + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6))
                                - x2 * (3 * x3 + 3 * x4 + 4 * (x5 + x6))
                            )
                        )
                    )
                    - 2
                    * Enu
                    * Mn
                    * x1
                    * x2
                    * (
                        -2 * x1 * x2**4
                        + 4 * x2**5
                        + x1 * x2**3 * x3
                        - 6 * x2**4 * x3
                        + 2 * x2**3 * x3**2
                        + x1 * x2**3 * x4
                        - 6 * x2**4 * x4
                        + 4 * x2**3 * x3 * x4
                        + 2 * x2**3 * x4**2
                        + 4 * x1**2 * x2**2 * x5
                        - 6 * x1 * x2**3 * x5
                        - 8 * x2**4 * x5
                        + 8 * x1 * x2**2 * x3 * x5
                        + 6 * x2**3 * x3 * x5
                        + 8 * x1 * x2**2 * x4 * x5
                        + 6 * x2**3 * x4 * x5
                        + 2 * x2**2 * x3 * x4 * x5
                        - 2 * x2**2 * x4**2 * x5
                        - 3 * x1**2 * x2 * x5**2
                        + 12 * x1 * x2**2 * x5**2
                        + 8 * x2**3 * x5**2
                        - 6 * x1 * x2 * x3 * x5**2
                        - 4 * x2**2 * x3 * x5**2
                        - 12 * x1 * x2 * x4 * x5**2
                        - 6 * x1 * x2 * x5**3
                        - 4 * x2**2 * x5**3
                        + 6 * x1 * x4 * x5**3
                        + 2 * ml1**3 * ml2 * x2**2 * (x2 - x5 - x6)
                        + ml1**4 * x2**2 * (x5 - x6)
                        + 4 * x1**2 * x2**2 * x6
                        - 6 * x1 * x2**3 * x6
                        - 8 * x2**4 * x6
                        + 8 * x1 * x2**2 * x3 * x6
                        + 6 * x2**3 * x3 * x6
                        - 2 * x2**2 * x3**2 * x6
                        + 8 * x1 * x2**2 * x4 * x6
                        + 6 * x2**3 * x4 * x6
                        + 2 * x2**2 * x3 * x4 * x6
                        + 16 * x1 * x2**2 * x5 * x6
                        + 8 * x2**3 * x5 * x6
                        - 6 * x1 * x2 * x3 * x5 * x6
                        - 4 * x2**2 * x3 * x5 * x6
                        - 6 * x1 * x2 * x4 * x5 * x6
                        - 4 * x2**2 * x4 * x5 * x6
                        - 18 * x1 * x2 * x5**2 * x6
                        - 4 * x2**2 * x5**2 * x6
                        + 6 * x1 * x3 * x5**2 * x6
                        + 12 * x1 * x5**3 * x6
                        - 3 * x1**2 * x2 * x6**2
                        + 12 * x1 * x2**2 * x6**2
                        + 8 * x2**3 * x6**2
                        - 12 * x1 * x2 * x3 * x6**2
                        - 6 * x1 * x2 * x4 * x6**2
                        - 4 * x2**2 * x4 * x6**2
                        - 18 * x1 * x2 * x5 * x6**2
                        - 4 * x2**2 * x5 * x6**2
                        + 6 * x1 * x4 * x5 * x6**2
                        - 6 * x1 * x2 * x6**3
                        - 4 * x2**2 * x6**3
                        + 6 * x1 * x3 * x6**3
                        + 12 * x1 * x5 * x6**3
                        + ml2**4 * x2**2 * (-x5 + x6)
                        + ml2**2
                        * (
                            2 * x2**4
                            - x2**3 * (x3 + x4 + 4 * x5)
                            + 6 * x1 * x5 * x6 * (-x5 + x6)
                            + 3 * x1 * x2 * (x5**2 - x6**2)
                            + x2**2
                            * (2 * x5 * (x3 - x4 + 2 * x5) + x1 * (-3 * x5 + x6))
                        )
                        + ml1**2
                        * (
                            2 * x2**4
                            + 6 * x1 * x5 * (x5 - x6) * x6
                            - x2**3 * (x3 + x4 + 4 * x6)
                            + 3 * x1 * x2 * (-(x5**2) + x6**2)
                            + x2**2
                            * (x1 * (x5 - 3 * x6) + 2 * x6 * (-x3 + x4 + 2 * x6))
                        )
                        + 2
                        * ml1
                        * ml2
                        * (
                            ml2**2 * x2**2 * (x2 - x5 - x6)
                            + x1
                            * (
                                x2**3
                                - 6 * x2**2 * (x5 + x6)
                                - 6 * x5 * x6 * (x5 + x6)
                                + 3 * x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                            + x2**2
                            * (
                                4 * x2**2
                                + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6))
                                - x2 * (3 * x3 + 3 * x4 + 4 * (x5 + x6))
                            )
                        )
                    )
                    + x2**2
                    * (
                        -4 * ml2**2 * Mn2 * x2**4
                        + 4 * Mn2 * x1 * x2**4
                        - 8 * Mn2 * x2**5
                        + 2 * ml2**2 * Mn2 * x2**3 * x3
                        - 2 * Mn2 * x1 * x2**3 * x3
                        + 12 * Mn2 * x2**4 * x3
                        - 4 * Mn2 * x2**3 * x3**2
                        + 2 * ml2**2 * Mn2 * x2**3 * x4
                        - 2 * Mn2 * x1 * x2**3 * x4
                        + 12 * Mn2 * x2**4 * x4
                        - 8 * Mn2 * x2**3 * x3 * x4
                        - 4 * Mn2 * x2**3 * x4**2
                        + 2 * ml2**4 * Mn2 * x2**2 * x5
                        + 2 * ml2**2 * Mn2 * x1 * x2**2 * x5
                        - ml2**2 * x1**2 * x2**2 * x5
                        - 4 * Mn2 * x1**2 * x2**2 * x5
                        + x1**3 * x2**2 * x5
                        + 8 * ml2**2 * Mn2 * x2**3 * x5
                        + 4 * Mn2 * x1 * x2**3 * x5
                        - 2 * x1**2 * x2**3 * x5
                        + 16 * Mn2 * x2**4 * x5
                        - 4 * ml2**2 * Mn2 * x2**2 * x3 * x5
                        - 8 * Mn2 * x1 * x2**2 * x3 * x5
                        + 2 * x1**2 * x2**2 * x3 * x5
                        - 12 * Mn2 * x2**3 * x3 * x5
                        + 4 * ml2**2 * Mn2 * x2**2 * x4 * x5
                        - 8 * Mn2 * x1 * x2**2 * x4 * x5
                        + 2 * x1**2 * x2**2 * x4 * x5
                        - 12 * Mn2 * x2**3 * x4 * x5
                        - 4 * Mn2 * x2**2 * x3 * x4 * x5
                        + 4 * Mn2 * x2**2 * x4**2 * x5
                        - 2 * ml2**2 * Mn2 * x1 * x2 * x5**2
                        + ml2**2 * x1**2 * x2 * x5**2
                        + 2 * Mn2 * x1**2 * x2 * x5**2
                        - x1**3 * x2 * x5**2
                        - 8 * ml2**2 * Mn2 * x2**2 * x5**2
                        - 8 * Mn2 * x1 * x2**2 * x5**2
                        + 4 * x1**2 * x2**2 * x5**2
                        - 16 * Mn2 * x2**3 * x5**2
                        + 4 * Mn2 * x1 * x2 * x3 * x5**2
                        - 2 * x1**2 * x2 * x3 * x5**2
                        + 8 * Mn2 * x2**2 * x3 * x5**2
                        + 8 * Mn2 * x1 * x2 * x4 * x5**2
                        - 4 * x1**2 * x2 * x4 * x5**2
                        + 4 * Mn2 * x1 * x2 * x5**3
                        - 2 * x1**2 * x2 * x5**3
                        + 8 * Mn2 * x2**2 * x5**3
                        - 4 * Mn2 * x1 * x4 * x5**3
                        + 2 * x1**2 * x4 * x5**3
                        - 2 * ml2**4 * Mn2 * x2**2 * x6
                        + 2 * ml2**2 * Mn2 * x1 * x2**2 * x6
                        + ml2**2 * x1**2 * x2**2 * x6
                        - 4 * Mn2 * x1**2 * x2**2 * x6
                        + x1**3 * x2**2 * x6
                        + 4 * Mn2 * x1 * x2**3 * x6
                        - 2 * x1**2 * x2**3 * x6
                        + 16 * Mn2 * x2**4 * x6
                        - 8 * Mn2 * x1 * x2**2 * x3 * x6
                        + 2 * x1**2 * x2**2 * x3 * x6
                        - 12 * Mn2 * x2**3 * x3 * x6
                        + 4 * Mn2 * x2**2 * x3**2 * x6
                        - 8 * Mn2 * x1 * x2**2 * x4 * x6
                        + 2 * x1**2 * x2**2 * x4 * x6
                        - 12 * Mn2 * x2**3 * x4 * x6
                        - 4 * Mn2 * x2**2 * x3 * x4 * x6
                        - 16 * Mn2 * x1 * x2**2 * x5 * x6
                        + 4 * x1**2 * x2**2 * x5 * x6
                        - 16 * Mn2 * x2**3 * x5 * x6
                        + 4 * Mn2 * x1 * x2 * x3 * x5 * x6
                        - 2 * x1**2 * x2 * x3 * x5 * x6
                        + 8 * Mn2 * x2**2 * x3 * x5 * x6
                        + 4 * Mn2 * x1 * x2 * x4 * x5 * x6
                        - 2 * x1**2 * x2 * x4 * x5 * x6
                        + 8 * Mn2 * x2**2 * x4 * x5 * x6
                        + 4 * ml2**2 * Mn2 * x1 * x5**2 * x6
                        - 2 * ml2**2 * x1**2 * x5**2 * x6
                        + 12 * Mn2 * x1 * x2 * x5**2 * x6
                        - 6 * x1**2 * x2 * x5**2 * x6
                        + 8 * Mn2 * x2**2 * x5**2 * x6
                        - 4 * Mn2 * x1 * x3 * x5**2 * x6
                        + 2 * x1**2 * x3 * x5**2 * x6
                        - 8 * Mn2 * x1 * x5**3 * x6
                        + 4 * x1**2 * x5**3 * x6
                        + 2 * ml2**2 * Mn2 * x1 * x2 * x6**2
                        - ml2**2 * x1**2 * x2 * x6**2
                        + 2 * Mn2 * x1**2 * x2 * x6**2
                        - x1**3 * x2 * x6**2
                        - 8 * Mn2 * x1 * x2**2 * x6**2
                        + 4 * x1**2 * x2**2 * x6**2
                        - 16 * Mn2 * x2**3 * x6**2
                        + 8 * Mn2 * x1 * x2 * x3 * x6**2
                        - 4 * x1**2 * x2 * x3 * x6**2
                        + 4 * Mn2 * x1 * x2 * x4 * x6**2
                        - 2 * x1**2 * x2 * x4 * x6**2
                        + 8 * Mn2 * x2**2 * x4 * x6**2
                        - 4 * ml2**2 * Mn2 * x1 * x5 * x6**2
                        + 2 * ml2**2 * x1**2 * x5 * x6**2
                        + 12 * Mn2 * x1 * x2 * x5 * x6**2
                        - 6 * x1**2 * x2 * x5 * x6**2
                        + 8 * Mn2 * x2**2 * x5 * x6**2
                        - 4 * Mn2 * x1 * x4 * x5 * x6**2
                        + 2 * x1**2 * x4 * x5 * x6**2
                        + 4 * Mn2 * x1 * x2 * x6**3
                        - 2 * x1**2 * x2 * x6**3
                        + 8 * Mn2 * x2**2 * x6**3
                        - 4 * Mn2 * x1 * x3 * x6**3
                        + 2 * x1**2 * x3 * x6**3
                        - 8 * Mn2 * x1 * x5 * x6**3
                        + 4 * x1**2 * x5 * x6**3
                        + 2 * ml1**4 * Mn2 * x2**2 * (-x5 + x6)
                        + 4 * ml1**3 * ml2 * Mn2 * x2**2 * (-x2 + x5 + x6)
                        + ml1**2
                        * (
                            x1**2 * (x5 - x6) * (x2**2 + 2 * x5 * x6 - x2 * (x5 + x6))
                            + 2
                            * Mn2
                            * (
                                -2 * x2**4
                                + 2 * x1 * x5 * x6 * (-x5 + x6)
                                + x2**3 * (x3 + x4 + 4 * x6)
                                + x1 * x2 * (x5**2 - x6**2)
                                + x2**2 * (2 * (x3 - x4 - 2 * x6) * x6 + x1 * (x5 + x6))
                            )
                        )
                        - 2
                        * ml1
                        * ml2
                        * (
                            2 * ml2**2 * Mn2 * x2**2 * (x2 - x5 - x6)
                            - x1**2
                            * (
                                x2**3
                                - 2 * x2**2 * (x5 + x6)
                                - 2 * x5 * x6 * (x5 + x6)
                                + x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                            )
                            + 2
                            * Mn2
                            * (
                                -(
                                    x1
                                    * (
                                        x2**3
                                        + 2 * x2**2 * (x5 + x6)
                                        + 2 * x5 * x6 * (x5 + x6)
                                        - x2 * (x5**2 + 4 * x5 * x6 + x6**2)
                                    )
                                )
                                + x2**2
                                * (
                                    4 * x2**2
                                    + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6))
                                    - x2 * (3 * x3 + 3 * x4 + 4 * (x5 + x6))
                                )
                            )
                        )
                    )
                )
            )
        )
        * Z2
    ) / (Enu2 * Mn2 * x1**2 * x2**4 * (x1 + 2 * x3) ** 2 * (x1 + 2 * x4) ** 2)
    return dsigma


def diffractive_dsigma(
    s,
    phi,
    theta,
    t,
    l,
    q,
    Enu,
    ml1,
    ml2,
    Vijk,
    Aijk,
):
    """
    Diffractive trident sigma (leptonic tensor factor).

    Translated exactly from dsigma_dPS_diff() in cross_sections.cxx.
    Returns the raw sigma factor WITHOUT kinematic prefactors. The caller
    (integrands.py) must multiply by:
        sin(theta) * alphaQED * Gf^2 / (16 * (4*pi)^3 * s^2)
        * beta * F_diffractive(q^2)^2 * 2/q/s
        * alphaQED/pi * GeV2_to_cm2

    Parameters
    ----------
    s : float
        Dilepton+neutrino CM energy squared [GeV^2].
    phi, theta : float
        Azimuthal and polar angles of the lepton in the dilepton rest frame [rad].
    t : float
        Lepton-pair invariant mass squared [GeV^2].
    l : float
        Argument l = l_true + t_true (see integrands.cxx).
    q : float
        Photon virtuality |q| [GeV].
    Enu : float
        Neutrino energy [GeV].
    ml1, ml2 : float
        Lepton masses [GeV].
    Vijk, Aijk : float
        Vector and axial-vector couplings.

    Returns
    -------
    float
        dsigma_dPS_diff [cm^2 / (GeV^5)], identical to the C++ function return value.
        Includes all prefactors: sin(theta), Gf^2/(16*(4pi)^3*s^2), beta,
        F_diffractive^2 * 2/q/s, GeV2_to_cm2 * alphaQED/pi.
    """
    from neptune.const import alphaQED, Gf, GeV2_to_cm2
    from neptune.nuclear_tools import F_diffractive

    mj = ml1
    mk = ml2
    mj2 = mj**2
    mk2 = mk**2
    Aijk2 = Aijk**2
    Vijk2 = Vijk**2
    beta_sq = 1.0 - 2.0 * (mj2 + mk2) / l + (mj2 - mk2) ** 2 / l**2
    beta = np.sqrt(np.maximum(beta_sq, 0.0))
    beta2 = beta**2
    l2 = l**2
    l3 = l**3
    s2 = s**2
    t2 = t**2
    mst2 = (-2 * s + t) ** 2
    costh = np.cos(theta)
    sinth = np.sin(theta)
    cosphi = np.cos(phi)
    costh2 = costh**2
    sinth2 = sinth**2
    cosphi2 = cosphi**2
    kin_arg = l * s * (l - t) * (-s + t)
    sqrtKIN = np.sqrt(np.maximum(kin_arg, 0.0))

    sigma = (
        128
        * (
            2
            * Aijk
            * l
            * t
            * Vijk
            * (
                (l - mj2 - mk2) * (mj2 - mk2) * (l - t) * t * (-2 * s + t)
                + beta
                * l
                * (l - t)
                * (-2 * s + t)
                * (2 * (l) ** 2 + (mj2 + mk2) * t - 2 * l * (mj2 + mk2 + t))
                * costh
                + beta
                * l
                * sqrtKIN
                * (
                    4 * (l) ** 2
                    + t * (4 * mj2 + 4 * mk2 + t)
                    - 2 * l * (2 * mj2 + 2 * mk2 + 3 * t)
                )
                * cosphi
                * sinth
            )
            + Aijk2
            * (
                t2
                * (
                    (l) ** 5
                    - (mj2 - mk2) ** 2 * (mj2 + mk2) * s * (s - t)
                    - (l) ** 4 * (mj2 + 4 * mj * mk + mk2 + 2 * t)
                    - l * (mj2 - mk2) ** 2 * (-s2 + (mj2 + mk2) * t + s * t)
                    + (l) ** 3
                    * (
                        -((mj) ** 4)
                        + 4 * (mj) ** 3 * mk
                        - (mk) ** 4
                        + s2
                        + 2 * mk2 * t
                        - s * t
                        + t2
                        + 2 * mj2 * (mk2 + t)
                        + mj * (4 * (mk) ** 3 + 6 * mk * t)
                    )
                    + (l) ** 2
                    * (
                        (mj) ** 6
                        - 4 * (mj) ** 3 * mk * t
                        + (mj) ** 4 * (-mk2 + t)
                        + mk2 * ((mk) ** 4 - s2 + mk2 * t + s * t - t2)
                        - 2 * mj * mk * (2 * s2 + 2 * mk2 * t - 2 * s * t + t2)
                        - mj2 * ((mk) ** 4 + s2 + 2 * mk2 * t - s * t + t2)
                    )
                )
                + beta2
                * (l) ** 2
                * (
                    -((mj2 + mk2) * s * (s - t) * t2)
                    + (l) ** 3 * (-2 * s + t) ** 2
                    - (l) ** 2 * (-2 * s + t) ** 2 * (mj2 + mk2 + 2 * t)
                    + l
                    * t
                    * (
                        mj2 * (-2 * s + t) ** 2
                        + mk2 * (-2 * s + t) ** 2
                        + t * (3 * s2 - 3 * s * t + t2)
                    )
                )
                * costh2
                - beta
                * l
                * sqrtKIN
                * cosphi
                * (
                    (mj2 - mk2) * t * (-2 * l + 2 * mj2 + 2 * mk2 + t) * (-2 * s + t)
                    + 2
                    * beta
                    * l
                    * s
                    * (
                        4 * (l) ** 2
                        + t * (2 * mj2 + 2 * mk2 + t)
                        - 2 * l * (2 * mj2 + 2 * mk2 + 3 * t)
                    )
                    * costh
                )
                * sinth
                - 4
                * beta2
                * (l) ** 3
                * s
                * (l - t)
                * (l - mj2 - mk2 - t)
                * (s - t)
                * cosphi2
                * sinth2
                + beta
                * l
                * t
                * costh
                * (
                    (mj2 - mk2)
                    * (
                        2 * (mj2 + mk2) * s * (s - t) * t
                        + (l) ** 2 * (-2 * s + t) ** 2
                        - l
                        * (
                            4 * mj2 * s * (s - t)
                            + 4 * mk2 * s * (s - t)
                            + t * (-2 * s + t) ** 2
                        )
                    )
                    + beta
                    * l
                    * sqrtKIN
                    * (
                        4 * (l) ** 2
                        + t * (2 * mj2 + 2 * mk2 + t)
                        - 2 * l * (2 * mj2 + 2 * mk2 + 3 * t)
                    )
                    * cosphi
                    * sinth
                )
            )
            + Vijk2
            * (
                t2
                * (
                    (l) ** 5
                    - (mj2 - mk2) ** 2 * (mj2 + mk2) * s * (s - t)
                    - (l) ** 4 * (mj2 - 4 * mj * mk + mk2 + 2 * t)
                    - l * (mj2 - mk2) ** 2 * (-s2 + (mj2 + mk2) * t + s * t)
                    - (l) ** 3
                    * (
                        (mj) ** 4
                        + 4 * (mj) ** 3 * mk
                        + (mk) ** 4
                        - s2
                        - 2 * mk2 * t
                        + s * t
                        - t2
                        - 2 * mj2 * (mk2 + t)
                        + mj * (4 * (mk) ** 3 + 6 * mk * t)
                    )
                    + (l) ** 2
                    * (
                        (mj) ** 6
                        + 4 * (mj) ** 3 * mk * t
                        + (mj) ** 4 * (-mk2 + t)
                        + mk2 * ((mk) ** 4 - s2 + mk2 * t + s * t - t2)
                        + 2 * mj * mk * (2 * s2 + 2 * mk2 * t - 2 * s * t + t2)
                        - mj2 * ((mk) ** 4 + s2 + 2 * mk2 * t - s * t + t2)
                    )
                )
                + beta2
                * (l) ** 2
                * (
                    -((mj2 + mk2) * s * (s - t) * t2)
                    + (l) ** 3 * (-2 * s + t) ** 2
                    - (l) ** 2 * (-2 * s + t) ** 2 * (mj2 + mk2 + 2 * t)
                    + l
                    * t
                    * (
                        mj2 * (-2 * s + t) ** 2
                        + mk2 * (-2 * s + t) ** 2
                        + t * (3 * s2 - 3 * s * t + t2)
                    )
                )
                * costh2
                - beta
                * l
                * sqrtKIN
                * cosphi
                * (
                    (mj2 - mk2) * t * (-2 * l + 2 * mj2 + 2 * mk2 + t) * (-2 * s + t)
                    + 2
                    * beta
                    * l
                    * s
                    * (
                        4 * (l) ** 2
                        + t * (2 * mj2 + 2 * mk2 + t)
                        - 2 * l * (2 * mj2 + 2 * mk2 + 3 * t)
                    )
                    * costh
                )
                * sinth
                - 4
                * beta2
                * (l) ** 3
                * s
                * (l - t)
                * (l - mj2 - mk2 - t)
                * (s - t)
                * cosphi2
                * sinth2
                + beta
                * l
                * t
                * costh
                * (
                    (mj2 - mk2)
                    * (
                        2 * (mj2 + mk2) * s * (s - t) * t
                        + (l) ** 2 * (-2 * s + t) ** 2
                        - l
                        * (
                            4 * mj2 * s * (s - t)
                            + 4 * mk2 * s * (s - t)
                            + t * (-2 * s + t) ** 2
                        )
                    )
                    + beta
                    * l
                    * sqrtKIN
                    * (
                        4 * (l) ** 2
                        + t * (2 * mj2 + 2 * mk2 + t)
                        - 2 * l * (2 * mj2 + 2 * mk2 + 3 * t)
                    )
                    * cosphi
                    * sinth
                )
            )
        )
    ) / (
        (t) ** 4
        * (-l - mj2 + mk2 + beta * l * costh)
        * (l - mj2 + mk2 + beta * l * costh)
    ) + (
        32
        * (
            (
                (s - t)
                * t
                * (Aijk + Vijk) ** 2
                * (4 * l * mj2 + t * (l + mj2 - mk2 - beta * l * costh))
                * (
                    (l - mj2 + mk2) * (l + s - t) * t
                    - beta * l * (2 * l * s - l * t - s * t + t2) * costh
                    + 2 * beta * l * sqrtKIN * cosphi * sinth
                )
                + 2
                * mj2
                * (Aijk + Vijk) ** 2
                * (
                    (l + mj2 - mk2) * (l - s) * t
                    - beta * l * (2 * l * s - l * t - s * t) * costh
                    + 2 * beta * l * sqrtKIN * cosphi * sinth
                )
                * (
                    (l - mj2 + mk2) * (l + s - t) * t
                    - beta * l * (2 * l * s - l * t - s * t + t2) * costh
                    + 2 * beta * l * sqrtKIN * cosphi * sinth
                )
                - (Aijk - Vijk)
                * (
                    s
                    * t
                    * (Aijk - Vijk)
                    * (4 * l * mj2 + t * (l + mj2 - mk2 - beta * l * costh))
                    * (
                        (l - mj2 + mk2) * (l - s) * t
                        + beta * l * (2 * l * s - l * t - s * t) * costh
                        - 2 * beta * l * sqrtKIN * cosphi * sinth
                    )
                    - 2
                    * mj
                    * (
                        -2
                        * l
                        * mk
                        * (l - t)
                        * t2
                        * (Aijk + Vijk)
                        * (4 * l * mj2 + t * (-l - mj2 + mk2 + beta * l * costh))
                        + mj
                        * (Aijk - Vijk)
                        * (
                            (l - mj2 + mk2) * (l - s) * t
                            + beta * l * (2 * l * s - l * t - s * t) * costh
                            - 2 * beta * l * sqrtKIN * cosphi * sinth
                        )
                        * (
                            (l + mj2 - mk2) * (l + s - t) * t
                            + beta * l * (2 * l * s - l * t - s * t + t2) * costh
                            - 2 * beta * l * sqrtKIN * cosphi * sinth
                        )
                    )
                )
            )
            / (l + mj2 - mk2 - beta * l * costh) ** 2
            + (
                (s - t)
                * t
                * (Aijk - Vijk) ** 2
                * (4 * l * mk2 + t * (l - mj2 + mk2 + beta * l * costh))
                * (
                    (l + mj2 - mk2) * (l + s - t) * t
                    + beta * l * (2 * l * s - l * t - s * t + t2) * costh
                    - 2 * beta * l * sqrtKIN * cosphi * sinth
                )
                - s
                * t
                * (Aijk + Vijk) ** 2
                * (4 * l * mk2 + t * (l - mj2 + mk2 + beta * l * costh))
                * (
                    (l + mj2 - mk2) * (l - s) * t
                    - beta * l * (2 * l * s - l * t - s * t) * costh
                    + 2 * beta * l * sqrtKIN * cosphi * sinth
                )
                + 2
                * mk
                * (
                    mk
                    * (Aijk + Vijk) ** 2
                    * (
                        (l + mj2 - mk2) * (l - s) * t
                        - beta * l * (2 * l * s - l * t - s * t) * costh
                        + 2 * beta * l * sqrtKIN * cosphi * sinth
                    )
                    * (
                        (l - mj2 + mk2) * (l + s - t) * t
                        - beta * l * (2 * l * s - l * t - s * t + t2) * costh
                        + 2 * beta * l * sqrtKIN * cosphi * sinth
                    )
                    + (Aijk - Vijk)
                    * (
                        -2
                        * l
                        * mj
                        * (l - t)
                        * t2
                        * (Aijk + Vijk)
                        * (4 * l * mk2 - t * (l - mj2 + mk2 + beta * l * costh))
                        + mk
                        * (Aijk - Vijk)
                        * (
                            (l - mj2 + mk2) * (l - s) * t
                            + beta * l * (2 * l * s - l * t - s * t) * costh
                            - 2 * beta * l * sqrtKIN * cosphi * sinth
                        )
                        * (
                            (l + mj2 - mk2) * (l + s - t) * t
                            + beta * l * (2 * l * s - l * t - s * t + t2) * costh
                            - 2 * beta * l * sqrtKIN * cosphi * sinth
                        )
                    )
                )
            )
            / (l - mj2 + mk2 + beta * l * costh) ** 2
        )
    ) / (
        t
    ) ** 4

    # Apply all prefactors exactly as in C++ dsigma_dPS_diff:
    #   sigma *= sin(theta) * alpha_QED*Gf*Gf/(16*pow(4.0*M_PI,3)*pow(s,2));
    #   sigma *= beta;
    #   sigma *= SQR(F_diffractive(q*q))*2/q/s;
    #   return sigma * GeV2_to_cm2 * alpha_QED / M_PI ;
    sigma *= np.sin(theta) * alphaQED * Gf**2 / (16.0 * (4.0 * np.pi) ** 3 * s**2)
    sigma *= beta
    sigma *= F_diffractive(q**2) ** 2 * 2.0 / q / s
    sigma *= GeV2_to_cm2 * alphaQED / np.pi

    return sigma


def coherent_epa_dsigma(
    s,
    phi,
    theta,
    t,
    l,
    q,
    Enu,
    ml1,
    ml2,
    Vijk,
    Aijk,
    Z,
    A,
):
    """
    Coherent (nuclear) trident dsigma in the Equivalent Photon Approximation.

    Same leptonic structure as diffractive_dsigma, but with the nuclear form
    factor FF_WS(q, A) and Z^2 instead of the nucleon dipole form factor.

    Returns dsigma [cm^2/GeV^5], including GeV2_to_cm2 conversion.
    Equivalent to the C++ dsigma_dPS function (coherent EPA variant).
    """
    from neptune.nuclear_tools import FF_WS, F_diffractive
    from neptune.const import alphaQED, Gf, GeV2_to_cm2

    sigma = diffractive_dsigma(s, phi, theta, t, l, q, Enu, ml1, ml2, Vijk, Aijk)

    # diffractive_dsigma includes F_diffractive(q^2)^2 * 2/q/s * GeV2_to_cm2 * alphaQED/pi
    # Replace with FF_WS(q,A)^2 * Z^2 * 2/q/s * GeV2_to_cm2 * alphaQED/pi
    # => divide by F_dif^2, multiply by FF_WS^2 * Z^2
    F_dif2 = F_diffractive(q**2) ** 2
    sigma = sigma / F_dif2 * FF_WS(q, A) ** 2 * float(Z) ** 2
    return sigma
