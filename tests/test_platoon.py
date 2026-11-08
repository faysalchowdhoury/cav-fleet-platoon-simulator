from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from platoon.api.app import create_app
from platoon.models import ACCParams, CACCParams, W99Params, acc_accel, cacc_vref, safe_accel, w99_accel
from platoon.sim import Fleet, FleetConfig, perturbation, queue_discharge

RNG = np.random.default_rng(0)
P = W99Params()


def w(v, vl, dx, vdes=30.0, a_prev=0.0, al=0.0):  # type: ignore[no-untyped-def]
    arr = lambda x: np.atleast_1d(np.asarray(x, float))  # noqa: E731
    return float(w99_accel(arr(v), arr(vl), arr(al), arr(dx), arr(vdes), arr(a_prev), P, RNG)[0])


def test_w99_regimes() -> None:
    assert w(0.0, 0.0, np.inf) == pytest.approx(P.cc8, abs=0.2)  # free start
    assert w(29.9, 30.0, np.inf) <= 0.11  # capped by desired speed
    assert w(20.0, 10.0, 15.0) < -3.0  # closing fast on a short gap: strong braking
    a_follow = w(20.0, 20.0, P.cc0 + P.cc1 * 20 + 1.0, a_prev=0.1)
    assert abs(a_follow) == pytest.approx(P.cc7, abs=1e-9)  # in the following band: oscillate at +/- CC7


def test_acc_cacc_equilibrium_and_safety() -> None:
    v = np.array([25.0])
    dx = np.array([2.0 + 1.1 * 25.0])
    assert acc_accel(v, v, dx, np.array([30.0]), ACCParams())[0] == pytest.approx(0.0, abs=1e-9)
    dxc = np.array([2.0 + 0.6 * 25.0])
    assert cacc_vref(v, v, np.zeros(1), dxc, np.array([30.0]), CACCParams())[0] == pytest.approx(25.0)
    assert safe_accel(np.array([25.0]), np.array([0.0]), np.array([5.0]), 0.1)[0] < -50  # must brake hard


def test_human_queue_discharge_tracks_cc1() -> None:
    f = [queue_discharge(FleetConfig(n=50, w99=W99Params(cc1=c), seed=1))["sat_flow_vph"] for c in (0.7, 1.2)]
    assert f[0] > f[1] > 1500


def test_full_cacc_raises_capacity_and_is_string_stable() -> None:
    h = queue_discharge(FleetConfig(n=50, seed=2))
    c = queue_discharge(FleetConfig(n=50, penetration=1.0, seed=2))
    assert c["sat_flow_vph"] > 1.4 * h["sat_flow_vph"] and c["collisions"] == 0
    s = perturbation(FleetConfig(n=30, penetration=1.0, seed=2))
    assert s["speed_drop_amplification"] < 1.5 and s["cacc_share"] > 0.9


def test_packet_loss_reduces_cacc_time() -> None:
    a = perturbation(FleetConfig(n=30, penetration=1.0, loss=0.0, seed=3))["cacc_share"]
    b = perturbation(FleetConfig(n=30, penetration=1.0, loss=0.6, seed=3))["cacc_share"]
    assert b < a


def test_penetration_assignment_and_no_collisions() -> None:
    f = Fleet.create(FleetConfig(n=40, penetration=0.25, seed=4), -np.arange(40) * 10.0, np.zeros(40))
    assert f.is_cav.sum() == 10
    assert perturbation(FleetConfig(n=30, penetration=0.5, loss=0.2, seed=4))["collisions"] == 0


def test_api() -> None:
    r = TestClient(create_app()).post("/v1/experiment", json={"penetration": 0.5, "v2v_loss": 0.1})
    assert r.status_code == 200 and r.json()["discharge_flow_vph"] > 1000
