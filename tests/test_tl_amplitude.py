"""
Smoke tests for the T/L decomposition harness in ``neptune._tl_amplitude``.

The Mathematica .dat exports may not yet exist on disk (they are produced by
``mathematica/export_TL_pieces.wl``), so the tests inject stub expressions
via the module's compiled-formula cache instead of touching the real files.
This isolates what we actually want to verify here: argument routing and
the algebraic combination formula

    dσ = (h_T σT + h_L σL) / (64 π² x1 x2)

with the regime and mode dispatch behaving as documented.
"""

import numpy as np
import pytest

from neptune import _tl_amplitude as tl


@pytest.fixture
def stubbed(monkeypatch):
    """Replace `_eval` with one that consults a per-piece stub table."""
    stubs = {}

    def fake_eval(piece, env):
        if piece not in stubs:
            raise KeyError(f"no stub registered for {piece}")
        # Mirror real evaluation: same globals, real env so numpy ops work.
        return eval(stubs[piece], tl._GLOBALS, env)

    monkeypatch.setattr(tl, "_eval", fake_eval)
    return stubs


def test_missing_export_raises_clear_error(tmp_path, monkeypatch):
    """If a .dat file is missing, the loader points at the .wl script."""
    tl._compiled.cache_clear()

    # Point the resource resolution at an empty directory.
    class _EmptyPath:
        def joinpath(self, *parts):
            return tmp_path / "/".join(parts)
    monkeypatch.setattr(tl.resources, "files", lambda pkg: _EmptyPath())

    with pytest.raises(tl.TLExportsMissing) as exc:
        tl._compiled("sigma_T_lep")
    assert "export_TL_pieces.wl" in str(exc.value)


def test_full_combines_T_and_L_coherent(stubbed):
    stubbed["h_T_coh"] = "x1 + x2"          # arbitrary
    stubbed["h_L_coh"] = "x1 * x2"
    stubbed["sigma_T_lep"] = "V2 + A2"
    stubbed["sigma_L_lep"] = "V2 - A2"

    x1, x2 = 0.05, 1.2
    expected = (
        ((x1 + x2) * (3.0 + 4.0)) + ((x1 * x2) * (3.0 - 4.0))
    ) / (64.0 * np.pi ** 2 * x1 * x2)

    got = tl.differential_cross_section_tl(
        x1, x2, 0.1, 0.2, 0.3, 0.4,
        Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
        V2=3.0, A2=4.0, VA=5.0,
        regime="coherent", mode="full",
        Z=18, FormFactor=0.9,
    )
    assert got == pytest.approx(expected)


def test_improved_epa_drops_longitudinal(stubbed):
    """improved-EPA: keep h_T σT(x1,...), drop h_L σL."""
    stubbed["h_T_coh"] = "1.0 + 0.0*x1*x2"   # scalar 1, with stub args used
    stubbed["sigma_T_lep"] = "x1 + x2"
    stubbed["h_L_coh"] = "1e30"              # large stub: must NOT be used
    stubbed["sigma_L_lep"] = "1e30"

    x1, x2 = 0.05, 1.2
    expected = (x1 + x2) / (64.0 * np.pi ** 2 * x1 * x2)

    got = tl.differential_cross_section_tl(
        x1, x2, 0.1, 0.2, 0.3, 0.4,
        Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
        V2=3.0, A2=4.0, VA=5.0,
        regime="coherent", mode="improved-epa",
        Z=18, FormFactor=0.9,
    )
    assert got == pytest.approx(expected)


def test_epa_evaluates_sigmaT_at_q2_zero(stubbed):
    """EPA: σT_lep_q0 must be invoked, σT_lep must NOT be."""
    stubbed["h_T_coh"] = "1.0 + 0.0*x1*x2"
    stubbed["sigma_T_lep_q0"] = "x2"             # only x2, NOT x1
    stubbed["sigma_T_lep"] = "1e30"              # poison: must NOT be called
    stubbed["h_L_coh"] = "1e30"                  # poison
    stubbed["sigma_L_lep"] = "1e30"              # poison

    x1, x2 = 0.05, 1.2
    expected = x2 / (64.0 * np.pi ** 2 * x1 * x2)

    got = tl.differential_cross_section_tl(
        x1, x2, 0.1, 0.2, 0.3, 0.4,
        Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
        V2=3.0, A2=4.0, VA=5.0,
        regime="coherent", mode="epa",
        Z=18, FormFactor=0.9,
    )
    assert got == pytest.approx(expected)


def test_diffractive_uses_HH1_HH2(stubbed):
    """Same kinematics, but routes to h_T_dif / h_L_dif and HH1, HH2."""
    stubbed["h_T_dif"] = "HH1 + HH2 + 0.0*x1*x2*Enu*Mn"
    stubbed["h_L_dif"] = "HH1 - HH2 + 0.0*x1*x2*Enu*Mn"
    stubbed["sigma_T_lep"] = "1.0 + 0.0*x1*x2"
    stubbed["sigma_L_lep"] = "2.0 + 0.0*x1*x2"

    x1, x2 = 0.05, 1.2
    HH1, HH2 = 0.7, 0.3
    expected = (
        (HH1 + HH2) * 1.0 + (HH1 - HH2) * 2.0
    ) / (64.0 * np.pi ** 2 * x1 * x2)

    got = tl.differential_cross_section_tl(
        x1, x2, 0.1, 0.2, 0.3, 0.4,
        Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
        V2=3.0, A2=4.0, VA=5.0,
        regime="diffractive", mode="full",
        HH1=HH1, HH2=HH2,
    )
    assert got == pytest.approx(expected)


def test_coherent_requires_Z_and_FormFactor():
    with pytest.raises(ValueError, match="Z and FormFactor"):
        tl.differential_cross_section_tl(
            0.05, 1.2, 0.1, 0.2, 0.3, 0.4,
            Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
            V2=1.0, A2=1.0, VA=1.0,
            regime="coherent", mode="full",
        )


def test_diffractive_requires_HH1_HH2():
    with pytest.raises(ValueError, match="HH1 and HH2"):
        tl.differential_cross_section_tl(
            0.05, 1.2, 0.1, 0.2, 0.3, 0.4,
            Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
            V2=1.0, A2=1.0, VA=1.0,
            regime="diffractive", mode="full",
        )


def test_invalid_regime_or_mode_rejected():
    with pytest.raises(ValueError, match="regime"):
        tl.differential_cross_section_tl(
            0.05, 1.2, 0.1, 0.2, 0.3, 0.4,
            Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
            V2=1.0, A2=1.0, VA=1.0,
            regime="bogus", mode="full",
            Z=18, FormFactor=0.9,
        )
    with pytest.raises(ValueError, match="mode"):
        tl.differential_cross_section_tl(
            0.05, 1.2, 0.1, 0.2, 0.3, 0.4,
            Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
            V2=1.0, A2=1.0, VA=1.0,
            regime="coherent", mode="bogus",
            Z=18, FormFactor=0.9,
        )


def test_vectorised_call(stubbed):
    """Combination should broadcast cleanly with numpy arrays."""
    stubbed["h_T_coh"] = "x1 + x2"
    stubbed["h_L_coh"] = "x1 * x2"
    stubbed["sigma_T_lep"] = "V2 + A2 + 0.0*x1"
    stubbed["sigma_L_lep"] = "V2 - A2 + 0.0*x1"

    x1 = np.array([0.05, 0.06, 0.07])
    x2 = np.array([1.2, 1.3, 1.4])
    expected = (
        (x1 + x2) * 7.0 + (x1 * x2) * (-1.0)
    ) / (64.0 * np.pi ** 2 * x1 * x2)

    got = tl.differential_cross_section_tl(
        x1, x2, 0.1, 0.2, 0.3, 0.4,
        Enu=10.0, Mn=37.2, ml1=0.106, ml2=0.106,
        V2=3.0, A2=4.0, VA=5.0,
        regime="coherent", mode="full",
        Z=18, FormFactor=0.9,
    )
    np.testing.assert_allclose(got, expected)
