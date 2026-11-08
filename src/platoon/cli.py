"""``platoon study`` | ``platoon serve``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="platoon")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study")
    s.add_argument("--out", default="results/mixed_autonomy")
    s.add_argument("--seeds", type=int, default=10)
    s.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "study":
        from .study import run, write_report

        print(write_report(run(a.out, a.seeds, a.workers), Path(a.out)).read_text(encoding="utf-8"))
    else:
        import uvicorn

        from .api.app import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
