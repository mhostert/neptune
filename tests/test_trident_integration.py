"""
Integration tests for the trident integrands and ``TridentProcess``.

These tests exercise the full Vegas integration through the T/L
integrands. They depend on the .dat exports produced by
``mathematica/BSM_trident.nb``; if any are missing the whole module
is skipped.
"""

import numpy as np
import pytest

from neptune import _tl_amplitude as tl
from neptune.integrands import (
    CoherentTridentIntegrand,
    DiffractiveTridentIntegrand,
    DISTridentIntegrand,
)
from neptune.model import TridentSMModel
from neptune.processes import TridentProcess
from neptune.dis import ConstantStructureFunctions
from neptune.const import CHANNEL_FLAGS


def _exports_present():
    return all(tl._resource_path(fn).is_file() for fn in tl._FILES.values())


pytestmark = pytest.mark.skipif(
    not _exports_present(),
    reason="T/L .dat exports not generated; evaluate mathematica/BSM_trident.nb",
)


def _proc(mode="full"):
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    return TridentProcess(m, Z=18, A=40, Enu=10.0,
                          mode=mode, nitn=4, neval=4_000)


# ---------------------------------------------------------------------------
# Vegas-level: each mode produces a finite, positive cross section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["full", "improved-epa", "epa"])
def test_coherent_modes_finite_and_positive(mode):
    mean, sdev = _proc(mode).sigma_coherent()
    assert np.isfinite(mean) and np.isfinite(sdev)
    assert mean > 0


@pytest.mark.parametrize("mode", ["full", "improved-epa", "epa"])
def test_diffractive_modes_finite_and_positive(mode):
    mean, sdev = _proc(mode).sigma_diffractive()
    assert np.isfinite(mean) and np.isfinite(sdev)
    assert mean > 0


# ---------------------------------------------------------------------------
# Mode dispatch and validation
# ---------------------------------------------------------------------------

def test_default_mode_is_full():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0)
    assert proc.mode == "full"
    assert proc._make_coherent_integrand().mode == "full"
    assert proc._make_diffractive_integrand().mode == "full"


def test_per_call_mode_override():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, mode="full")
    assert proc._make_coherent_integrand(mode="epa").mode == "epa"
    assert proc._make_diffractive_integrand(mode="improved-epa").mode == "improved-epa"


def test_integrand_classes_returned():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0)
    assert isinstance(proc._make_coherent_integrand(), CoherentTridentIntegrand)
    assert isinstance(proc._make_diffractive_integrand(), DiffractiveTridentIntegrand)


def test_tau_channels_integrands_are_finite():
    sf = ConstantStructureFunctions(
        F1_value=1.0,
        F2_value=0.2,
        q2_min=1.0,
        q2_max=50.0,
        x_min=1e-4,
    )
    rng = np.random.default_rng(12345)

    for nu, l1, l2 in CHANNEL_FLAGS:
        if "tau" not in (nu, l1, l2):
            continue

        model = TridentSMModel(nu_flavor=nu, l1_flavor=l1, l2_flavor=l2)
        integrands = [
            CoherentTridentIntegrand(
                nu, l1, l2, model, model.ml1, model.ml2, 18, 40, Enu=100.0,
            ),
            DiffractiveTridentIntegrand(
                nu, l1, l2, model, model.ml1, model.ml2, Enu=100.0,
            ),
            DISTridentIntegrand(
                nu, l1, l2, model, model.ml1, model.ml2, sf, Enu=100.0,
            ),
        ]

        for integrand in integrands:
            with np.errstate(over="ignore", invalid="ignore"):
                vals = integrand(rng.random((32, integrand.ndim))).ravel()
            assert np.all(np.isfinite(vals))


def test_invalid_mode_rejected():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    with pytest.raises(ValueError, match="mode"):
        TridentProcess(m, Z=18, A=40, Enu=10.0, mode="nonsense")


def test_mode_aliases_normalised():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    assert TridentProcess(m, Z=18, A=40, Enu=10.0, mode="iepa").mode == "improved-epa"
    assert TridentProcess(m, Z=18, A=40, Enu=10.0, mode="improved_epa").mode == "improved-epa"


# ---------------------------------------------------------------------------
# Sanity: σ_total combines target diffractive and coherent
# ---------------------------------------------------------------------------

def test_sigma_diffractive_sums_target_nucleons(monkeypatch):
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=2, A=5, Enu=10.0)

    def fake_sigma_nucleon(nucleon, **kwargs):
        if nucleon == "proton":
            return 10.0, 0.5
        if nucleon == "neutron":
            return 1.0, 0.25
        raise AssertionError(nucleon)

    monkeypatch.setattr(proc, "sigma_diffractive_nucleon", fake_sigma_nucleon)

    mean, sdev = proc.sigma_diffractive()
    assert mean == pytest.approx(2 * 10.0 + 3 * 1.0)
    assert sdev == pytest.approx(np.sqrt((2 * 0.5) ** 2 + (3 * 0.25) ** 2))


def test_sigma_total_combines_components():
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    proc = TridentProcess(m, Z=18, A=40, Enu=10.0, nitn=4, neval=4_000)
    out = proc.sigma_total()
    assert set(out) == {"diffractive", "coherent", "total_per_nucleus"}
    dif_mean, _ = out["diffractive"]
    coh_mean, _ = out["coherent"]
    tot_mean, _ = out["total_per_nucleus"]
    assert tot_mean == pytest.approx(dif_mean + coh_mean, rel=1e-9, abs=0)


# ---------------------------------------------------------------------------
# Event generator
# ---------------------------------------------------------------------------

def _generator(**kw):
    from neptune.MC import TridentGenerator
    m = TridentSMModel(nu_flavor="mu", l1_flavor="mu", l2_flavor="mu")
    base = dict(Z=18, A=40, Enu=10.0, n_events=3_000, nitn=5, neval=8_000, seed=11)
    base.update(kw)
    return TridentGenerator(m, **base)


def test_generator_weights_reproduce_process_cross_sections():
    """Coherent and diffractive (p + n) weight sums match TridentProcess."""
    ev = _generator().generate()
    proc = _proc()
    proc.neval = 20_000
    for regime, (ref, ref_sd) in (("coherent", proc.sigma_coherent()),
                                  ("diffractive", proc.sigma_diffractive())):
        w = ev["weight"][ev["mode"] == regime].sum()
        assert w == pytest.approx(ref, rel=0.03, abs=0), regime


def test_generator_seed_is_reproducible():
    a = _generator().generate()
    b = _generator().generate()
    np.testing.assert_array_equal(a["weight"], b["weight"])
    c = _generator(seed=12).generate()
    assert not np.array_equal(a["weight"][:100], c["weight"][:100])


def test_generator_forwards_nuclear_model():
    from neptune.nuclear_models import FreeNucleon
    gen = _generator(nuclear_model=FreeNucleon())
    f = gen._make_diffractive_integrand("neutron")
    assert isinstance(f.nuclear_model, FreeNucleon)
    assert f.nucleon == "neutron"
