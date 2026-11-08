"""Penetration x packet-loss sweep and W99 CC1/CC2 sensitivity, with report and figures."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import W99Params
from .sim import FleetConfig, perturbation, queue_discharge

PENETRATION = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
LOSS = (0.0, 0.2, 0.5)
CC1 = (0.6, 0.9, 1.2, 1.5)
CC2 = (2.0, 4.0, 6.0)


def _one(args: tuple[str, float, float, float, float, int]) -> dict[str, Any]:
    kind, p, loss, cc1, cc2, seed = args
    w = replace(W99Params(), cc1=cc1, cc2=cc2)
    q = queue_discharge(FleetConfig(n=60, penetration=p, loss=loss, w99=w, seed=seed))
    s = perturbation(FleetConfig(n=40, penetration=p, loss=loss, w99=w, seed=seed))
    return {"kind": kind, "penetration": p, "loss": loss, "cc1": cc1, "cc2": cc2, "seed": seed,
            "sat_flow_vph": q["sat_flow_vph"], "amplification": s["speed_drop_amplification"],
            "tail_decel": s["max_decel_tail"], "min_ttc_s": s["min_ttc_s"], "share_ttc_lt_1_5": s["share_ttc_below_1_5"],
            "cacc_share": s["cacc_share"], "collisions": q["collisions"] + s["collisions"]}


def run(out_dir: str | Path, seeds: int = 10, workers: int = 2) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    jobs = [("mix", p, l, 0.9, 4.0, s) for p in PENETRATION for l in (LOSS if p > 0 else (0.0,)) for s in range(seeds)]
    jobs += [("w99", 0.0, 0.0, c1, c2, s) for c1 in CC1 for c2 in CC2 for s in range(seeds)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_one, jobs, chunksize=4))
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    g = df.groupby(["kind", "penetration", "loss", "cc1", "cc2"]).agg(
        sat=("sat_flow_vph", "mean"), sat_sd=("sat_flow_vph", "std"), amp=("amplification", "mean"),
        tail_decel=("tail_decel", "mean"), min_ttc=("min_ttc_s", "min"), ttc15=("share_ttc_lt_1_5", "mean"),
        cacc=("cacc_share", "mean"), collisions=("collisions", "sum"), n=("seed", "count")).reset_index()
    g.to_csv(out / "summary_table.csv", index=False)
    summary = {"runs": len(df), "wall_s": wall, "collisions_total": int(df["collisions"].sum()),
               "table": g.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _plot(g, out / "figures")
    return summary


def _plot(g: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mix = g[g["kind"] == "mix"]
    base = mix[mix["penetration"] == 0.0].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for loss, col in zip(LOSS, ("#1c7ed6", "#f59f00", "#e03131")):
        sub = pd.concat([pd.DataFrame([base]), mix[(mix["loss"] == loss) & (mix["penetration"] > 0)]]).sort_values("penetration")
        axes[0].errorbar(100 * sub["penetration"], sub["sat"], yerr=sub["sat_sd"], marker="o", color=col, capsize=3, label=f"V2V loss {loss:.0%}")
        axes[1].plot(100 * sub["penetration"], sub["amp"], marker="o", color=col)
        axes[2].plot(100 * sub["penetration"], 100 * sub["cacc"], marker="o", color=col)
    axes[0].set_ylabel("queue-discharge flow (veh/h/ln)")
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].set_ylabel("speed-drop amplification, tail / head")
    axes[2].set_ylabel("CAV time in CACC mode (%)")
    for ax in axes:
        ax.set_xlabel("CAV penetration (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "penetration_sweep.png", dpi=130)
    plt.close(fig)

    w = g[g["kind"] == "w99"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for c2, col in zip(CC2, ("#2b8a3e", "#1c7ed6", "#862e9c")):
        sub = w[w["cc2"] == c2].sort_values("cc1")
        axes[0].plot(sub["cc1"], sub["sat"], marker="o", color=col, label=f"CC2 = {c2:.0f} m")
        axes[1].plot(sub["cc1"], sub["amp"], marker="o", color=col)
    axes[0].set_ylabel("queue-discharge flow (veh/h/ln)")
    axes[1].set_ylabel("speed-drop amplification")
    for ax in axes:
        ax.set_xlabel("W99 CC1 headway time (s)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "w99_sensitivity.png", dpi=130)
    plt.close(fig)


def write_report(s: dict[str, Any], out: Path) -> Path:
    g = pd.DataFrame(s["table"])
    mix = g[g["kind"] == "mix"].sort_values(["loss", "penetration"])
    L = ["# Mixed-autonomy results", "", f"{s['runs']} runs (10 seeds per cell) in {s['wall_s']:.0f} s; collisions across all runs: {s['collisions_total']}.", "",
         "## Penetration x V2V loss", "", "| CAV % | V2V loss | Discharge flow (veh/h/ln) | Amplification | Tail decel (m/s²) | Min TTC (s) | Vehicles with TTC < 1.5 s | Time in CACC |",
         "|---|---|---|---|---|---|---|---|"]
    for r in mix.itertuples():
        L.append(f"| {100 * r.penetration:.0f}% | {100 * r.loss:.0f}% | {r.sat:.0f} ± {r.sat_sd:.0f} | {r.amp:.2f} | {r.tail_decel:.2f} | {r.min_ttc:.2f} | "
                 f"{100 * r.ttc15:.1f}% | {100 * r.cacc:.0f}% |")
    w = g[g["kind"] == "w99"].sort_values(["cc1", "cc2"])
    L += ["", "## W99 sensitivity (all human, 0% CAV)", "", "| CC1 (s) | CC2 (m) | Discharge flow (veh/h/ln) | Amplification | Min TTC (s) |", "|---|---|---|---|---|"]
    for r in w.itertuples():
        L.append(f"| {r.cc1:.1f} | {r.cc2:.0f} | {r.sat:.0f} ± {r.sat_sd:.0f} | {r.amp:.2f} | {r.min_ttc:.2f} |")
    L += ["", "![Penetration](figures/penetration_sweep.png)", "", "![W99](figures/w99_sensitivity.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
