"""Kinematic checks on the inclusive DIS trident phase-space map."""

import numpy as np
import pytest

from neptune import const
from neptune.integrands import _qmag
from neptune.phase_space import (
    dis_trident_xB_limits,
    map_dis_unit_to_trident_physical,
    map_unit_to_physical,
)


@pytest.fixture(scope="module")
def dis_ps():
    rng = np.random.default_rng(12345)
    x = rng.random((20_000, 9))
    return map_dis_unit_to_trident_physical(
        x, Enu=20.0, ml1=const.m_mu, ml2=const.m_mu,
        Q2_min=1.0, W2_min=4.0, xB_min=1e-6,
    )


def test_dis_map_has_valid_points(dis_ps):
    assert dis_ps["valid"].mean() > 0.05


def test_dis_map_respects_cuts(dis_ps):
    v = dis_ps["valid"]
    assert np.all(dis_ps["x1"][v] >= 1.0 * (1 - 1e-12))
    assert np.all(dis_ps["W2"][v] >= 4.0 * (1 - 1e-12))
    assert np.all((dis_ps["xB"][v] > 0.0) & (dis_ps["xB"][v] <= 1.0))
    assert np.all(dis_ps["Jacob"][v] > 0.0)


def test_dis_W2_definition(dis_ps):
    v = dis_ps["valid"]
    x1, xB = dis_ps["x1"][v], dis_ps["xB"][v]
    W2 = const.m_AVG**2 + x1 * (1.0 / xB - 1.0)
    assert np.allclose(dis_ps["W2"][v], W2, rtol=1e-12)


def test_dis_map_x2_below_elastic_bound(dis_ps):
    """At fixed Q², x2 = p_nu.q is bounded by the DIS limit, which exceeds
    the elastic (xB = 1) limit only through the smaller recoil."""
    v = dis_ps["valid"]
    Enu = 20.0
    x1, x2, xB = dis_ps["x1"][v], dis_ps["x2"][v], dis_ps["xB"][v]
    Mn = const.m_AVG
    x2_u = Enu * (np.sqrt(x1 + x1**2 / (4 * Mn**2 * xB**2)) - x1 / (2 * Mn * xB))
    assert np.all(x2 <= x2_u * (1 + 1e-10))


def test_xB_upper_limit_enforces_W2_min():
    x1 = np.array([1.0, 2.0, 5.0])
    _, xB_u = dis_trident_xB_limits(x1, 20.0, const.m_mu, const.m_mu, const.m_AVG,
                                    W2_min=4.0)
    W2_at_edge = const.m_AVG**2 + x1 * (1.0 / xB_u - 1.0)
    assert np.allclose(W2_at_edge, 4.0)


def test_qmag_elastic_and_inelastic():
    Q2, Mn = 0.5, const.m_AVG
    assert _qmag(Q2, Mn) == pytest.approx(np.sqrt(Q2 * (1 + Q2 / (4 * Mn**2))))
    # Inelastic: larger energy transfer at the same Q² → larger |q|.
    assert _qmag(Q2, Mn, 0.3) > _qmag(Q2, Mn, 1.0)


@pytest.mark.parametrize("bsm_mode", [const.SM_ONLY, const.INTERFERENCE,
                                      const.BSM_ONLY, const.SM_AND_BSM])
def test_elastic_map_m6_within_bounds_all_bsm_modes(bsm_mode):
    rng = np.random.default_rng(7)
    ps = map_unit_to_physical(
        rng.random((4000, 8)), Enu=5.0, ml1=const.m_mu, ml2=const.m_mu,
        Mn=const.m_proton, mzprime=0.05, bsm_mode=bsm_mode,
    )
    v = ps["valid"]
    assert v.mean() > 0.5
    assert np.all(ps["m6"][v] >= 0.0)
    assert np.all(ps["m6"][v] <= 2.0 * ps["E1"][v] * ps["E2"][v] * (1 + 1e-12))
