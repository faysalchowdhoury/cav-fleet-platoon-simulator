"""Single-lane mixed-autonomy simulation (vectorised over vehicles, 0.1-s steps).

Vehicle 0 is the platoon head; vehicle ``i`` follows ``i - 1``. The head either drives freely
(queue discharge) or follows a prescribed speed profile (perturbation). V2V messages from a CAV
leader reach a CAV follower with probability ``1 - loss``, one step late (100 ms latency).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import A_MAX, A_MIN, ACCParams, CACCParams, W99Params, acc_accel, cacc_vref, safe_accel, w99_accel


@dataclass
class FleetConfig:
    n: int = 40
    penetration: float = 0.0
    loss: float = 0.0
    w99: W99Params = field(default_factory=W99Params)
    acc: ACCParams = field(default_factory=ACCParams)
    cacc: CACCParams = field(default_factory=CACCParams)
    v_des_mean: float = 30.0
    v_des_sd: float = 1.5
    car_length: float = 4.5
    dt: float = 0.1
    actuator_lag_s: float = 0.3
    seed: int = 0


@dataclass
class Fleet:
    cfg: FleetConfig
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    is_cav: np.ndarray
    v_des: np.ndarray
    v_ref: np.ndarray
    lost: np.ndarray
    known_vl: np.ndarray
    rng: np.random.Generator
    collisions: int = 0
    cacc_steps: int = 0
    cav_steps: int = 0

    @classmethod
    def create(cls, cfg: FleetConfig, x0: np.ndarray, v0: np.ndarray) -> "Fleet":
        rng = np.random.default_rng(cfg.seed)
        n = cfg.n
        k = int(round(cfg.penetration * n))
        is_cav = np.zeros(n, bool)
        is_cav[rng.choice(n, size=k, replace=False)] = True
        v_des = np.clip(rng.normal(cfg.v_des_mean, cfg.v_des_sd, n), 20.0, 40.0)
        v_des[is_cav] = cfg.v_des_mean
        return cls(cfg, x0.astype(float), v0.astype(float), np.zeros(n), is_cav, v_des, v0.astype(float).copy(),
                   np.zeros(n, int), v0.astype(float).copy(), rng)

    def gaps(self) -> np.ndarray:
        g = np.full(self.cfg.n, np.inf)
        g[1:] = self.x[:-1] - self.x[1:] - self.cfg.car_length
        return g

    def step(self, head_speed: float | None) -> None:
        c = self.cfg
        dt = c.dt
        dx = self.gaps()
        v_l = np.r_[self.v[0], self.v[:-1]]
        a_l = np.r_[0.0, self.a[:-1]]
        # V2V reception (CAV follower behind CAV leader)
        cav_pair = self.is_cav & np.r_[False, self.is_cav[:-1]]
        got = cav_pair & (self.rng.random(c.n) >= c.loss)
        self.lost = np.where(got, 0, self.lost + 1)
        self.known_vl = np.where(got, np.r_[self.v[0], self.v[:-1]], self.known_vl)  # 1-step-late value
        cacc_mode = cav_pair & (self.lost < c.cacc.fallback_steps)
        a_h = w99_accel(self.v, v_l, a_l, dx, self.v_des, self.a, c.w99, self.rng)
        a_acc = acc_accel(self.v, v_l, dx, self.v_des, c.acc)  # ACC uses on-board radar: no V2V needed
        v_cmd = cacc_vref(self.v, self.known_vl, self.a, dx, self.v_des, c.cacc)
        a_cacc = np.clip((v_cmd - self.v) / dt, A_MIN, A_MAX)
        a_cmd = np.where(cacc_mode, a_cacc, a_acc)
        # first-order powertrain/brake lag for automated vehicles (tau_a), then the safety ceiling
        a_lag = self.a + (a_cmd - self.a) * min(dt / c.actuator_lag_s, 1.0)
        a_cav = np.minimum(a_lag, safe_accel(self.v, v_l, dx, dt))
        a = np.where(self.is_cav, np.clip(a_cav, A_MIN, A_MAX), a_h)
        if head_speed is not None:
            a[0] = np.clip((head_speed - self.v[0]) / dt, -6.0, 2.5)
        self.cacc_steps += int(cacc_mode.sum())
        self.cav_steps += int(self.is_cav.sum())
        v_new = np.maximum(self.v + a * dt, 0.0)
        dxs = 0.5 * (self.v + v_new) * dt
        self.x = self.x + dxs
        self.a = (v_new - self.v) / dt
        self.v = v_new
        # collision guard: count and resolve overlaps (should not happen with sane parameters)
        g = self.gaps()
        bad = np.flatnonzero(g < 0.0)
        if bad.size:
            self.collisions += bad.size
            for i in bad:
                self.x[i] = self.x[i - 1] - c.car_length - 0.01
                self.v[i] = min(self.v[i], self.v[i - 1])


def queue_discharge(cfg: FleetConfig, *, detector_m: float = 150.0, t_max: float = 180.0, skip: int = 10) -> dict[str, float]:
    """Release a standing queue; saturation flow from headways at a downstream detector."""
    n = cfg.n
    rng = np.random.default_rng(cfg.seed + 1)
    gap = np.full(n, 2.0) + rng.uniform(-0.2, 0.2, n)
    x0 = -np.cumsum(np.r_[0.0, (cfg.car_length + gap)[1:]])
    fleet = Fleet.create(cfg, x0, np.zeros(n))
    cross = np.full(n, np.nan)
    steps = int(t_max / cfg.dt)
    for k in range(steps):
        old = fleet.x.copy()
        fleet.step(None)
        hit = np.isnan(cross) & (old < detector_m) & (fleet.x >= detector_m)
        cross[hit] = (k + 1) * cfg.dt
        if not np.isnan(cross[-1]):
            break
    h = np.diff(cross[skip:])
    h = h[np.isfinite(h)]
    return {"sat_flow_vph": float(3600.0 / h.mean()) if h.size else float("nan"),
            "collisions": fleet.collisions,
            "cacc_share": fleet.cacc_steps / max(fleet.cav_steps, 1)}


def perturbation(cfg: FleetConfig, *, v0: float = 25.0, v_low: float = 15.0, warmup_s: float = 60.0,
                 t_after: float = 80.0) -> dict[str, float]:
    """Head brakes 25 -> 15 m/s at 3 m/s^2, holds 10 s, recovers at 1 m/s^2. Measures string (in)stability."""
    n = cfg.n
    rng = np.random.default_rng(cfg.seed + 2)
    eq_gap = np.full(n, 2.0 + 1.0 * v0) + rng.uniform(-1, 1, n)
    x0 = -np.cumsum(np.r_[0.0, (cfg.car_length + eq_gap)[1:]])
    fleet = Fleet.create(cfg, x0, np.full(n, v0))
    fleet.v_des = np.maximum(fleet.v_des, v0 + 2.0)  # nobody wants to go slower than the platoon speed
    dt = cfg.dt
    for _ in range(int(warmup_s / dt)):
        fleet.step(v0)
    profile = np.r_[np.linspace(v0, v_low, int((v0 - v_low) / 3.0 / dt) + 1), np.full(int(10 / dt), v_low),
                    np.linspace(v_low, v0, int((v0 - v_low) / 1.0 / dt) + 1)]
    profile = np.r_[profile, np.full(int(t_after / dt), v0)]
    vmin = fleet.v.copy()
    amin = np.zeros(n)
    ttc_min = np.full(n, np.inf)
    for s in profile:
        fleet.step(float(s))
        vmin = np.minimum(vmin, fleet.v)
        amin = np.minimum(amin, fleet.a)
        closing = np.r_[0.0, fleet.v[1:] - fleet.v[:-1]]
        g = fleet.gaps()
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing > 0.05, g / closing, np.inf)
        ttc_min = np.minimum(ttc_min, ttc)
    drop = v0 - vmin
    tail = drop[-5:].mean()
    return {
        "speed_drop_amplification": float(tail / max(drop[0], 1e-6)),  # > 1: string-unstable
        "max_decel_tail": float(-amin[-5:].mean()),
        "min_ttc_s": float(np.min(ttc_min[1:])),
        "share_ttc_below_1_5": float(np.mean(ttc_min[1:] < 1.5)),
        "collisions": fleet.collisions,
        "cacc_share": fleet.cacc_steps / max(fleet.cav_steps, 1),
    }
