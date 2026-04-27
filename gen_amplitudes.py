"""
Script to generate amplitudes.py from C++ source.
Run from neptune root: python gen_amplitudes.py
"""

import re

CXX_FILE = "/Users/mhostert/Repos/bsm_trident/MH_Code/clean_code/cross_sections.cxx"
OUT_FILE = "/Users/mhostert/Repos/neptune/src/neptune/amplitudes.py"

with open(CXX_FILE) as f:
    content = f.read()


# --------------------------------------------------------------------------
# 1. Extract coherent dsigma expression
# --------------------------------------------------------------------------
idx = content.find("long double dsigma = (8*(alphaQED")
end_idx = content.find(";", idx)
coh_raw = content[idx:end_idx].replace("long double dsigma = ", "")

COH_SUBS = [
    ("(x1*x1*x1*x1)", "x1**4"),
    ("(x1*x1*x1)", "x1**3"),
    ("(x1*x1)", "x1**2"),
    ("x1*x1*x1", "x1**3"),
    ("x1*x1", "x1**2"),
    ("(x2*x2*x2*x2*x2)", "x2**5"),
    ("(x2*x2*x2*x2)", "x2**4"),
    ("(x2*x2*x2)", "x2**3"),
    ("(x2*x2)", "x2**2"),
    ("x2*x2*x2*x2*x2", "x2**5"),
    ("x2*x2*x2*x2", "x2**4"),
    ("x2*x2*x2", "x2**3"),
    ("x2*x2", "x2**2"),
    ("(x3*x3)", "x3**2"),
    ("x3*x3", "x3**2"),
    ("(x4*x4)", "x4**2"),
    ("x4*x4", "x4**2"),
    ("(x5*x5*x5)", "x5**3"),
    ("(x5*x5)", "x5**2"),
    ("x5*x5*x5", "x5**3"),
    ("x5*x5", "x5**2"),
    ("(x6*x6*x6)", "x6**3"),
    ("(x6*x6)", "x6**2"),
    ("x6*x6*x6", "x6**3"),
    ("x6*x6", "x6**2"),
    ("(Mn*Mn)", "Mn2"),
    ("Mn*Mn", "Mn2"),
    ("(Enu*Enu)", "Enu2"),
    ("Enu*Enu", "Enu2"),
    ("(ml1*ml1*ml1*ml1)", "ml1**4"),
    ("(ml1*ml1*ml1)", "ml1**3"),
    ("(ml1*ml1)", "ml1**2"),
    ("ml1*ml1*ml1*ml1", "ml1**4"),
    ("ml1*ml1*ml1", "ml1**3"),
    ("ml1*ml1", "ml1**2"),
    ("(ml2*ml2*ml2*ml2)", "ml2**4"),
    ("(ml2*ml2*ml2)", "ml2**3"),
    ("(ml2*ml2)", "ml2**2"),
    ("ml2*ml2*ml2*ml2", "ml2**4"),
    ("ml2*ml2*ml2", "ml2**3"),
    ("ml2*ml2", "ml2**2"),
    ("alphaQED*alphaQED", "alphaQED**2"),
    ("FormFactor*FormFactor", "FF2"),
    ("(-2*Enu*Mn + x2)*(-2*Enu*Mn + x2)", "Xsq"),
    ("(Enu2*Mn*x1 - Enu*x1*x2 - Mn*x2**2)", "(-D1)"),
    ("(-(Enu2*Mn*x1) + Enu*x1*x2 + Mn*x2**2)", "D1"),
    ("*(Z*Z)", "*Z2"),
    ("(Gf*Gf)", "Gf2"),
    ("Gf*Gf", "Gf2"),
    ("((x1 + 2*x3)*(x1 + 2*x3))", "(x1+2*x3)**2"),
    ("((x1 + 2*x4)*(x1 + 2*x4))", "(x1+2*x4)**2"),
    ("(x1 + 2*x4)*(x1 + 2*x4)", "(x1+2*x4)**2"),
    ("(x1 + 2*x3)*(x1 + 2*x3)", "(x1+2*x3)**2"),
]

for old, new in COH_SUBS:
    coh_raw = coh_raw.replace(old, new)

coh_expr = coh_raw.strip()


# --------------------------------------------------------------------------
# 2. Extract diffractive sigma expression
# --------------------------------------------------------------------------
dif_func = content.find("long double dsigma_dPS_diff(int nu_alpha,")
dif_sig_start = content.find("  sigma = //TA + TB", dif_func)
dif_sig_end = content.find("  /* EXTRA FACTOR OF SIN(THETA)", dif_func)
dif_raw = content[dif_sig_start:dif_sig_end].strip()

dif_raw = dif_raw.replace("sigma = //TA + TB", "sigma = ")

DIF_SUBS = [
    ("pow(mj,2)", "mj2"),
    ("pow(mk,2)", "mk2"),
    ("pow(Aijk,2)", "Aijk2"),
    ("pow(Vijk,2)", "Vijk2"),
    ("pow(beta,2)", "beta2"),
    ("pow(s,2)", "s2"),
    ("pow(t,2)", "t2"),
    ("pow(cos(theta),2)", "costh2"),
    ("pow(sin(theta),2)", "sinth2"),
    ("pow(cos(phi),2)", "cosphi2"),
    ("sqrt(l*s*(l - t)*(-s + t))", "sqrtKIN"),
    ("cos(theta)", "costh"),
    ("sin(theta)", "sinth"),
    ("cos(phi)", "cosphi"),
]
for old, new in DIF_SUBS:
    dif_raw = dif_raw.replace(old, new)

# pow(x, n) -> (x)**n
dif_raw = re.sub(r"pow\(([^,()]+),\s*(\d+)\)", r"(\1)**\2", dif_raw)
# Handle nested pow: pow((expr), n) where expr may contain parens
dif_raw = re.sub(r"pow\(\(([^()]+)\),\s*(\d+)\)", r"(\1)**\2", dif_raw)
# pow with simple identifiers
dif_raw = re.sub(r"pow\(([a-zA-Z0-9_]+),\s*(\d+)\)", r"\1**\2", dif_raw)

dif_raw = dif_raw.replace(";", "").replace("  // nu gamma part!", "")
dif_expr = dif_raw.strip()


# --------------------------------------------------------------------------
# 3. Write amplitudes.py
# --------------------------------------------------------------------------
def ind(text, n=4):
    pad = " " * n
    return "\n".join(pad + line for line in text.split("\n"))


lines = [
    '"""',
    "amplitudes.py",
    "=============",
    "Matrix elements for neutrino trident production.",
    "",
    "Two scattering regimes:",
    "",
    "Coherent (nuclear):  nu + N -> nu + l- + l+ + N",
    "  Phase space variables: x1=Q2, x2..x6 Lorentz invariants of lepton momenta.",
    "  Nuclear target scatters coherently; form factor is Woods-Saxon FF_WS(q,A).",
    "  Exact polynomial translated from cross_sections.cxx (CohFromIntToPhysical).",
    "",
    "Diffractive (nucleon):  nu + gamma* -> nu + l- + l+",
    "  Phase space variables: s, t, l (Lorentz invariants), theta, phi (dilepton angles).",
    "  Form factor is Dirac dipole F_diffractive(q^2).",
    "  Exact expression translated from dsigma_dPS_diff in cross_sections.cxx.",
    "",
    "References:",
    "  Czyz et al., Phys.Rev. 177 (1969) 2311",
    "  MH Hostert C++ implementation in bsm_trident/MH_Code/clean_code/",
    '"""',
    "",
    "import numpy as np",
    "",
    "",
    "def coherent_dsigma(",
    "    x1, x2, x3, x4, x5, x6,",
    "    Enu, Mn, ml1, ml2,",
    "    V2, A2, VA,",
    "    Z, A,",
    "    Diag11=1.0, Diag22=1.0, Diag12=1.0,",
    "):",
    '    """',
    "    Differential cross section integrand for coherent trident production.",
    "",
    "    Translated exactly from CohFromIntToPhysical() in cross_sections.cxx.",
    "    Includes nuclear form factor FF_WS, alphaQED^2, Gf^2, and Z^2 factors.",
    "    Returns dsigma BEFORE multiplying by the phase space Jacobian.",
    "",
    "    Parameters",
    "    ----------",
    "    x1..x6 : float or ndarray",
    "        Lorentz-invariant kinematic variables.",
    "        x1 = Q^2; x2..x6 defined in phase_space.py.",
    "    Enu, Mn : float",
    "        Neutrino energy and nucleon mass [GeV].",
    "    ml1, ml2 : float",
    "        Lepton masses: ml1 = m(l-), ml2 = m(l+) [GeV].",
    "    V2, A2, VA : float or ndarray",
    "        Effective coupling coefficients V^2, A^2, V*A.",
    "    Z, A : int",
    "        Atomic and mass numbers of the nucleus.",
    "    Diag11, Diag22, Diag12 : float",
    "        Diagram selector flags (1.0=include, 0.0=exclude).",
    "        Diag12=0 when l1 != l2 (no t-channel/u-channel interference).",
    "",
    "    Returns",
    "    -------",
    "    float or ndarray",
    "        dsigma [GeV^-2] before Jacobian multiplication.",
    '    """',
    "    from neptune.nuclear_tools import FF_WS",
    "    from neptune.const import alphaQED, Gf",
    "",
    "    FF2 = FF_WS(np.sqrt(np.maximum(x1, 0.0)), A)**2",
    "    Z2 = float(Z)**2",
    "    Gf2 = Gf**2",
    "    Mn2 = Mn**2",
    "    Enu2 = Enu**2",
    "    # D1 = -(Enu^2*Mn*x1) + Enu*x1*x2 + Mn*x2^2",
    "    D1 = -(Enu2*Mn*x1) + Enu*x1*x2 + Mn*x2**2",
    "    # Xsq = (-2*Enu*Mn + x2)^2",
    "    Xsq = (-2*Enu*Mn + x2)**2",
    "",
    "    dsigma = (",
]
# Add the coherent expression indented
for line in coh_expr.split("\n"):
    lines.append("    " + line)
lines += [
    "    )",
    "    return dsigma",
    "",
    "",
    "def diffractive_dsigma(",
    "    s, phi, theta, t, l, q,",
    "    Enu, ml1, ml2,",
    "    Vijk, Aijk,",
    "):",
    '    """',
    "    Diffractive trident sigma (leptonic tensor factor).",
    "",
    "    Translated exactly from dsigma_dPS_diff() in cross_sections.cxx.",
    "    Returns the raw sigma factor WITHOUT kinematic prefactors. The caller",
    "    (integrands.py) must multiply by:",
    "        sin(theta) * alphaQED * Gf^2 / (16 * (4*pi)^3 * s^2)",
    "        * beta * F_diffractive(q^2)^2 * 2/q/s",
    "        * alphaQED/pi * GeV2_to_cm2",
    "",
    "    Parameters",
    "    ----------",
    "    s : float",
    "        Dilepton+neutrino CM energy squared [GeV^2].",
    "    phi, theta : float",
    "        Azimuthal and polar angles of the lepton in the dilepton rest frame [rad].",
    "    t : float",
    "        Lepton-pair invariant mass squared [GeV^2].",
    "    l : float",
    "        Argument l = l_true + t_true (see integrands.cxx).",
    "    q : float",
    "        Photon virtuality |q| [GeV].",
    "    Enu : float",
    "        Neutrino energy [GeV].",
    "    ml1, ml2 : float",
    "        Lepton masses [GeV].",
    "    Vijk, Aijk : float",
    "        Vector and axial-vector couplings.",
    "",
    "    Returns",
    "    -------",
    "    float",
    "        sigma leptonic factor [GeV^n].",
    '    """',
    "    mj = ml1",
    "    mk = ml2",
    "    mj2 = mj**2",
    "    mk2 = mk**2",
    "    Aijk2 = Aijk**2",
    "    Vijk2 = Vijk**2",
    "    beta_sq = 1.0 - 2.0*(mj2 + mk2)/l + (mj2 - mk2)**2/l**2",
    "    beta = np.sqrt(np.maximum(beta_sq, 0.0))",
    "    beta2 = beta**2",
    "    l2 = l**2",
    "    l3 = l**3",
    "    s2 = s**2",
    "    t2 = t**2",
    "    mst2 = (-2*s + t)**2",
    "    costh = np.cos(theta)",
    "    sinth = np.sin(theta)",
    "    cosphi = np.cos(phi)",
    "    costh2 = costh**2",
    "    sinth2 = sinth**2",
    "    cosphi2 = cosphi**2",
    "    kin_arg = l * s * (l - t) * (-s + t)",
    "    sqrtKIN = np.sqrt(np.maximum(kin_arg, 0.0))",
    "",
]
dif_lines = dif_expr.split("\n")
# sigma = \n    (... needs to be sigma = (\n    ...
if dif_lines[0].strip() == "sigma =":
    dif_lines[0] = "sigma = ("
    # find the last non-empty line and append closing paren
    last_nonempty = len(dif_lines) - 1
    while last_nonempty > 0 and not dif_lines[last_nonempty].strip():
        last_nonempty -= 1
    dif_lines[last_nonempty] = dif_lines[last_nonempty] + ")"
for line in dif_lines:
    lines.append("    " + line)
lines += [
    "",
    "    return sigma",
    "",
]

output = "\n".join(lines)

with open(OUT_FILE, "w") as f:
    f.write(output)

print(f"Wrote {OUT_FILE}")
print(f"  Lines: {len(lines)}")
print(f"  Chars: {len(output)}")
