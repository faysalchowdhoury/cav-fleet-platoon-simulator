"""Car-following models, vectorised over vehicles.

Wiedemann 99 (human drivers)
----------------------------
Re-implementation of the W99 regime logic, following the pseudocode published by Higgs, Abbas
and Medina (2011), with Vissim's parameter meanings. ``dv = v_leader - v`` (negative = closing), ``dx`` = net gap.

=====  ===============================================  ===========  =========
param  meaning                                          default      unit
=====  ===============================================  ===========  =========
CC0    standstill distance                              1.50         m
CC1    headway time                                     0.90         s
CC2    following variation (longitudinal oscillation)   4.00         m
CC3    threshold for entering "following"               -8.00        s
CC4    negative speed difference threshold              -0.35        m/s
CC5    positive speed difference threshold              0.35         m/s
CC6    speed dependency of oscillation                  11.44        1/(m s)
CC7    oscillation acceleration                         0.25         m/s^2
CC8    desired acceleration from standstill             3.50         m/s^2
CC9    desired acceleration at 80 km/h                  1.50         m/s^2
=====  ===============================================  ===========  =========

Thresholds: ``SDXc = CC0 + CC1 v_slower``, ``SDXo = SDXc + CC2``, ``SDXv = SDXo + CC3 (dv - CC4)``,
``SDV = CC6 dx^2 / 1e4``, ``CLDV = CC4 - SDV``, ``OPDV = CC5 + SDV``. Regimes: (1) decelerate to increase
distance, (2) decelerate to decrease distance (approaching), (3) keep distance (oscillate at +/-CC7),
(4) free acceleration with CC8 -> CC9 interpolated over 0-80 km/h. **This is not Vissim's proprietary code**;
regime structure and parameter roles match, but trajectories are not bit-identical.

ACC / CACC (automated vehicles), Milanes & Shladover (2014), PATH field-tested gains
------------------------------------------------------------------------------------
* ACC  ``a = k1 e + k2 (v_l - v)`` with ``e = dx - t_ACC v``, k1 = 0.23 s^-2, k2 = 0.07 s^-1, t_ACC = 1.1 s
* CACC ``v_ref(k) = v(k-1) + kp e(k) + kd de/dt(k)`` per 0.1-s control step (the discrete form used with
  these gains by Liu, Kan & Shladover 2018), ``e = dx - t_CACC v``, kp = 0.45, kd = 0.25, t_CACC = 0.6 s.
  It needs the leader's V2V message (speed); after ``fallback_steps`` consecutive lost messages, or
  behind a human leader, it degrades to ACC.

Both run under a Gipps safe-speed ceiling (:func:`safe_accel`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KMH80 = 80.0 / 3.6


@dataclass(frozen=True)
class W99Params:
    cc0: float = 1.5
    cc1: float = 0.9
    cc2: float = 4.0
    cc3: float = -8.0
    cc4: float = -0.35
    cc5: float = 0.35
    cc6: float = 11.44
    cc7: float = 0.25
    cc8: float = 3.5
    cc9: float = 1.5


@dataclass(frozen=True)
class ACCParams:
    k1: float = 0.23
    k2: float = 0.07
    t_gap: float = 1.1
    standstill: float = 2.0


@dataclass(frozen=True)
class CACCParams:
    kp: float = 0.45
    kd: float = 0.25
    t_gap: float = 0.6
    standstill: float = 2.0
    tau: float = 0.5
    fallback_steps: int = 3


A_MIN, A_MAX = -8.0, 4.0  # physical envelope; must not clip W99's CC8 = 3.5 m/s^2 standstill acceleration


def w99_accel(v: np.ndarray, v_l: np.ndarray, a_l: np.ndarray, dx: np.ndarray, v_des: np.ndarray, a_prev: np.ndarray,
              p: W99Params, rng: np.random.Generator) -> np.ndarray:
    """Vectorised W99 acceleration for followers (``dx`` = net gap to leader, ``inf`` = no leader)."""
    dv = v_l - v
    has_l = np.isfinite(dx)
    dx_ = np.where(has_l, dx, 1e6)
    u = rng.random(v.shape)
    v_slow = np.where((dv < 0) | (a_l < -1.0), v, v_l - dv * (u - 0.5))
    sdxc = np.where(v_l > 0.1, p.cc0 + p.cc1 * v_slow, p.cc0)
    sdxo = sdxc + p.cc2
    sdxv = sdxo + p.cc3 * (dv - p.cc4)
    sdv = p.cc6 * dx_ ** 2 / 1e4
    cldv = np.where(v_l > 0.1, p.cc4 - sdv, 0.0)
    opdv = np.where(v > p.cc5, p.cc5 + sdv, sdv)
    a = a_prev.copy()

    # regime 1: too close -> decelerate to increase distance
    r1 = has_l & (dv < opdv) & (dx_ <= sdxc)
    a1 = np.where((v > 0) & (dv < 0),
                  np.where(dx_ > p.cc0, a_l + dv ** 2 / np.minimum(p.cc0 - dx_, -0.1), a_l + 0.5 * (dv - opdv)),
                  a)
    a1 = np.where(a1 > -p.cc7, -p.cc7, np.maximum(a1, -10.0 + 0.5 * np.sqrt(np.maximum(v, 0.0))))
    # regime 2: approaching -> decelerate to decrease closing speed
    r2 = has_l & ~r1 & (dv < cldv) & (dx_ < sdxv)
    a2 = np.maximum(0.5 * dv ** 2 / np.minimum(-dx_ + sdxc - 0.1, -0.1), -10.0 + np.sqrt(np.maximum(v, 0.0)))
    # regime 3: following -> oscillate at +/- CC7
    r3 = has_l & ~r1 & ~r2 & (dv < opdv) & (dx_ < sdxo)
    a3 = np.where(a <= 0, np.minimum(a, -p.cc7), np.maximum(a, p.cc7))
    a3 = np.minimum(a3, v_des - v)
    # regime 4: free / relax
    r4 = ~(r1 | r2 | r3)
    a_max = p.cc8 + (p.cc9 - p.cc8) * np.minimum(v, KMH80) / KMH80 + 0.25 * (u - 0.5)
    a4 = np.where(dx_ < sdxo, np.minimum(dv ** 2 / np.maximum(sdxo - dx_, 0.1), a_max), a_max)
    a4 = np.minimum(a4, v_des - v)
    out = np.select([r1, r2, r3, r4], [a1, a2, a3, a4])
    return np.clip(out, A_MIN, A_MAX)


def safe_accel(v: np.ndarray, v_l: np.ndarray, dx: np.ndarray, dt: float, *, b: float = 6.0, tau: float = 0.3,
               s0: float = 1.0) -> np.ndarray:
    """Collision-avoidance ceiling for automated vehicles (Gipps safe speed).

    The published ACC/CACC gains were tuned for cruising, not emergency braking. Simulation
    studies with these controllers (e.g. Xiao, Wang & van Arem 2017) add a safety layer, and so
    does this one: speed may not exceed what still allows a stop behind a leader braking at ``b``.
    """
    gap = np.where(np.isfinite(dx), dx - s0, 1e6)
    v_safe = -b * tau + np.sqrt(np.maximum((b * tau) ** 2 + v_l ** 2 + 2.0 * b * np.maximum(gap, 0.0), 0.0))
    return (v_safe - v) / dt


def acc_accel(v: np.ndarray, v_l: np.ndarray, dx: np.ndarray, v_des: np.ndarray, p: ACCParams) -> np.ndarray:
    e = dx - p.standstill - p.t_gap * v
    a = p.k1 * e + p.k2 * (v_l - v)
    free = 0.4 * (v_des - v)  # cruise control towards the set speed
    a = np.where(np.isfinite(dx), np.minimum(a, free), free)
    return np.clip(a, A_MIN, A_MAX)


def cacc_vref(v: np.ndarray, v_l: np.ndarray, a: np.ndarray, dx: np.ndarray, v_des: np.ndarray, p: CACCParams) -> np.ndarray:
    """Speed command for the next 0.1-s control step."""
    e = np.where(np.isfinite(dx), dx - p.standstill - p.t_gap * v, 0.0)
    de = (v_l - v) - p.t_gap * a
    return np.clip(v + p.kp * e + p.kd * de, 0.0, v_des)
