"""``POST /v1/experiment``: run one discharge + perturbation experiment synchronously in a worker thread."""

from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..models import W99Params
from ..sim import FleetConfig, perturbation, queue_discharge


class ExperimentRequest(BaseModel):
    penetration: float = Field(0.0, ge=0.0, le=1.0)
    v2v_loss: float = Field(0.0, ge=0.0, lt=1.0)
    cc1: float = Field(0.9, gt=0.2, le=3.0)
    cc2: float = Field(4.0, ge=0.0, le=15.0)
    seed: int = Field(0, ge=0)


def create_app() -> FastAPI:
    app = FastAPI(title="Mixed-autonomy platoon lab", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/experiment")
    async def experiment(req: ExperimentRequest) -> dict[str, float]:
        w = replace(W99Params(), cc1=req.cc1, cc2=req.cc2)

        def work() -> dict[str, float]:
            q = queue_discharge(FleetConfig(n=60, penetration=req.penetration, loss=req.v2v_loss, w99=w, seed=req.seed))
            s = perturbation(FleetConfig(n=40, penetration=req.penetration, loss=req.v2v_loss, w99=w, seed=req.seed))
            return {"discharge_flow_vph": q["sat_flow_vph"], **s}

        return await run_in_threadpool(work)

    return app
