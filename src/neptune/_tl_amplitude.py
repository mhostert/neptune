"""
Transverse / longitudinal photon decomposition of trident matrix elements.

The expressions are CForm exports from ``mathematica/BSM_trident.nb`` produced
by ``mathematica/export_TL_pieces.wl`` (paste-into-notebook script). The
decomposition is::

    dΣ = ( h_T(regime; x1, x2, ...) · σT_lep(x1..x6, ...)
            + h_L(regime; x1, x2, ...) · σL_lep(x1..x6, ...) )
            / (64 π² x1 x2)

with the same x1..x6 invariant phase space used everywhere in the
``BSM_trident.nb`` formalism (see ``phase_space.map_unit_to_physical``).
``σT_lep`` and ``σL_lep`` are purely leptonic and identical for both
regimes; the choice of ``h_T`` and ``h_L`` is what selects coherent vs.
diffractive.

Three modes are exposed:

* ``"full"``           — keep both T and L pieces with full x1 dependence.
* ``"improved-epa"``   — drop ``h_L σL``; keep ``σT(x1, x2, ...)``.
* ``"epa"``            — drop ``h_L σL``; evaluate ``σT`` at x1 = 0.

Files expected in ``data/matrixelements/`` (produced by the .wl script):

    sigmaT_lep_DIAG_AV.dat
    sigmaL_lep_DIAG_AV.dat
    sigmaT_lep_q0_DIAG_AV.dat
    hT_coh.dat,  hL_coh.dat
    hT_dif.dat,  hL_dif.dat

Until the export script has been run, calling any of the loaders raises
``TLExportsMissing`` with instructions.
"""

from functools import lru_cache
from importlib import resources
import re

import numpy as np

from neptune import const


_FORMULA_DIR = "data/matrixelements"

_FILES = {
    "sigma_T_lep":    "sigmaT_lep_DIAG_AV.dat",
    "sigma_L_lep":    "sigmaL_lep_DIAG_AV.dat",
    "sigma_T_lep_q0": "sigmaT_lep_q0_DIAG_AV.dat",
    "h_T_coh":        "hT_coh.dat",
    "h_L_coh":        "hL_coh.dat",
    "h_T_dif":        "hT_dif.dat",
    "h_L_dif":        "hL_dif.dat",
}

# Symbols permitted in each .dat file. Anything outside these sets raises
# during compilation; this guards against a stray Mathematica symbol leaking
# through CForm.
_LEP_SYMS = {
    "x1", "x2", "x3", "x4", "x5", "x6",
    "Enu", "ml1", "ml2",
    "V2", "A2", "VA",
    "Diag11", "Diag22", "Diag12",
    "alphaQED", "Gf",
    "Pi", "pow", "sqrt",
}

_HCOH_SYMS = {
    "x1", "x2", "Enu", "Mn", "Z", "FormFactor",
    "alphaQED", "Pi", "pow", "sqrt",
}

_HDIF_SYMS = {
    "x1", "x2", "Enu", "Mn", "HH1", "HH2",
    "alphaQED", "Pi", "pow", "sqrt",
}

_ALLOWED = {
    "sigma_T_lep":    _LEP_SYMS,
    "sigma_L_lep":    _LEP_SYMS,
    "sigma_T_lep_q0": _LEP_SYMS,
    "h_T_coh":        _HCOH_SYMS,
    "h_L_coh":        _HCOH_SYMS,
    "h_T_dif":        _HDIF_SYMS,
    "h_L_dif":        _HDIF_SYMS,
}


class TLExportsMissing(FileNotFoundError):
    """Raised when the T/L .dat exports have not been generated yet."""


def _resource_path(filename: str):
    return resources.files("neptune").joinpath(_FORMULA_DIR, filename)


@lru_cache(maxsize=None)
def _compiled(piece: str):
    filename = _FILES[piece]
    path = _resource_path(filename)
    if not path.is_file():
        raise TLExportsMissing(
            f"T/L matrix-element export {filename!r} not found at "
            f"{path}. Run mathematica/export_TL_pieces.wl from "
            "BSM_trident.nb to generate it."
        )
    expr = path.read_text(encoding="utf-8").strip()
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
    unknown = names - _ALLOWED[piece]
    if unknown:
        raise ValueError(
            f"Unexpected symbol(s) in {filename}: {', '.join(sorted(unknown))}"
        )
    return compile(expr, f"<neptune:{filename}>", "eval")


_GLOBALS = {
    "__builtins__": {},
    "pow":  np.power,
    "sqrt": np.sqrt,
    "Pi":   np.pi,
}


def _eval(piece: str, env: dict):
    code = _compiled(piece)
    return eval(code, _GLOBALS, env)


# ---------------------------------------------------------------------------
# Leptonic cross sections (identical for coherent and diffractive)
# ---------------------------------------------------------------------------

def sigma_T_lep(
    x1, x2, x3, x4, x5, x6,
    Enu, ml1, ml2,
    V2, A2, VA,
    Diag11=1.0, Diag22=1.0, Diag12=1.0,
):
    """Leptonic transverse σT_{γν}(x1, x2, x3, x4, x5, x6)."""
    return _eval("sigma_T_lep", {
        "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6,
        "Enu": Enu, "ml1": ml1, "ml2": ml2,
        "V2": V2, "A2": A2, "VA": VA,
        "Diag11": Diag11, "Diag22": Diag22, "Diag12": Diag12,
        "alphaQED": const.alphaQED, "Gf": const.Gf,
    })


def sigma_L_lep(
    x1, x2, x3, x4, x5, x6,
    Enu, ml1, ml2,
    V2, A2, VA,
    Diag11=1.0, Diag22=1.0, Diag12=1.0,
):
    """Leptonic longitudinal σL_{γν}(x1, x2, x3, x4, x5, x6)."""
    return _eval("sigma_L_lep", {
        "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6,
        "Enu": Enu, "ml1": ml1, "ml2": ml2,
        "V2": V2, "A2": A2, "VA": VA,
        "Diag11": Diag11, "Diag22": Diag22, "Diag12": Diag12,
        "alphaQED": const.alphaQED, "Gf": const.Gf,
    })


def sigma_T_lep_q0(
    x2, x3, x4, x5, x6,
    Enu, ml1, ml2,
    V2, A2, VA,
    Diag11=1.0, Diag22=1.0, Diag12=1.0,
):
    """Leptonic σT_{γν} evaluated at q² = 0 (used by the strict EPA mode)."""
    return _eval("sigma_T_lep_q0", {
        # x1 = 0 in the substituted expression; pass 0.0 so any residual
        # x1 in the exported polynomial evaluates to 0 (it should not appear).
        "x1": 0.0,
        "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6,
        "Enu": Enu, "ml1": ml1, "ml2": ml2,
        "V2": V2, "A2": A2, "VA": VA,
        "Diag11": Diag11, "Diag22": Diag22, "Diag12": Diag12,
        "alphaQED": const.alphaQED, "Gf": const.Gf,
    })


# ---------------------------------------------------------------------------
# Hadronic flux functions (the only thing that differs between regimes)
# ---------------------------------------------------------------------------

def h_T_coh(x1, x2, Enu, Mn, Z, FormFactor):
    return _eval("h_T_coh", {
        "x1": x1, "x2": x2, "Enu": Enu, "Mn": Mn,
        "Z": float(Z), "FormFactor": FormFactor,
        "alphaQED": const.alphaQED,
    })


def h_L_coh(x1, x2, Enu, Mn, Z, FormFactor):
    return _eval("h_L_coh", {
        "x1": x1, "x2": x2, "Enu": Enu, "Mn": Mn,
        "Z": float(Z), "FormFactor": FormFactor,
        "alphaQED": const.alphaQED,
    })


def h_T_dif(x1, x2, Enu, Mn, HH1, HH2):
    return _eval("h_T_dif", {
        "x1": x1, "x2": x2, "Enu": Enu, "Mn": Mn,
        "HH1": HH1, "HH2": HH2,
        "alphaQED": const.alphaQED,
    })


def h_L_dif(x1, x2, Enu, Mn, HH1, HH2):
    return _eval("h_L_dif", {
        "x1": x1, "x2": x2, "Enu": Enu, "Mn": Mn,
        "HH1": HH1, "HH2": HH2,
        "alphaQED": const.alphaQED,
    })


# ---------------------------------------------------------------------------
# Unified differential cross section
# ---------------------------------------------------------------------------

_VALID_REGIMES = {"coherent", "diffractive"}
_VALID_MODES = {"full", "improved-epa", "epa"}


def differential_cross_section_tl(
    x1, x2, x3, x4, x5, x6,
    Enu, Mn, ml1, ml2,
    V2, A2, VA,
    *,
    regime,
    mode="full",
    Diag11=1.0, Diag22=1.0, Diag12=1.0,
    Z=None, FormFactor=None,
    HH1=None, HH2=None,
):
    """
    Combined dSigma in the T/L decomposition::

        full          : (h_T σT(x1,...) + h_L σL(x1,...)) / (64 π² x1 x2)
        improved-epa  :  h_T σT(x1,...)                      / (64 π² x1 x2)
        epa           :  h_T σT(0, x2,...)                   / (64 π² x1 x2)

    Parameters
    ----------
    x1..x6 : array_like
        Lorentz invariants (see ``phase_space.map_unit_to_physical``).
    regime : {"coherent", "diffractive"}
        Selects which hadronic flux functions ``h_T``, ``h_L`` to use.
    mode : {"full", "improved-epa", "epa"}
        See module docstring.
    Z, FormFactor : required if regime == "coherent".
    HH1, HH2     : required if regime == "diffractive". These are the
        nucleon hadronic-tensor combinations
        ``HH1 = F1² + x1/(4Mn²) F2²``,
        ``HH2 = (F1+F2)²`` (matches ``_elastic_amplitude``).

    Returns
    -------
    dsigma : ndarray
        Differential cross section before the phase-space Jacobian; multiply
        by ``map_unit_to_physical()['Jacob']`` and ``GeV2_to_cm2`` for cm².
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(f"regime must be one of {_VALID_REGIMES}")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}")

    if regime == "coherent":
        if Z is None or FormFactor is None:
            raise ValueError("coherent regime requires Z and FormFactor")
        h_T = h_T_coh(x1, x2, Enu, Mn, Z, FormFactor)
        h_L = h_L_coh(x1, x2, Enu, Mn, Z, FormFactor) if mode == "full" else None
    else:
        if HH1 is None or HH2 is None:
            raise ValueError("diffractive regime requires HH1 and HH2")
        h_T = h_T_dif(x1, x2, Enu, Mn, HH1, HH2)
        h_L = h_L_dif(x1, x2, Enu, Mn, HH1, HH2) if mode == "full" else None

    if mode == "epa":
        sT = sigma_T_lep_q0(
            x2, x3, x4, x5, x6, Enu, ml1, ml2, V2, A2, VA,
            Diag11=Diag11, Diag22=Diag22, Diag12=Diag12,
        )
        numerator = h_T * sT
    elif mode == "improved-epa":
        sT = sigma_T_lep(
            x1, x2, x3, x4, x5, x6, Enu, ml1, ml2, V2, A2, VA,
            Diag11=Diag11, Diag22=Diag22, Diag12=Diag12,
        )
        numerator = h_T * sT
    else:  # full
        sT = sigma_T_lep(
            x1, x2, x3, x4, x5, x6, Enu, ml1, ml2, V2, A2, VA,
            Diag11=Diag11, Diag22=Diag22, Diag12=Diag12,
        )
        sL = sigma_L_lep(
            x1, x2, x3, x4, x5, x6, Enu, ml1, ml2, V2, A2, VA,
            Diag11=Diag11, Diag22=Diag22, Diag12=Diag12,
        )
        numerator = h_T * sT + h_L * sL

    return numerator / (64.0 * np.pi ** 2 * x1 * x2)
