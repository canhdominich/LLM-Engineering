"""Freeze P1 predictions (module 01 project, m01/project-eta) for the active orders in orders.csv into eta_snapshot.json."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[2] / "m01" / "project-eta" / "solution"
sys.path.insert(0, str(P1))

from anhti_eta.config import load_settings  # noqa: E402
from anhti_eta.predict import Predictor  # noqa: E402
from anhti_eta.train import train_and_save  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        settings = load_settings(P1 / "eta.toml", artifacts_dir=tmp)
        metrics = train_and_save(settings)
        predictor = Predictor.load(settings.model_path)
    orders = pd.read_csv(HERE / "orders.csv", parse_dates=["created_at"], keep_default_na=False)
    active = orders[orders["status"].isin(["created", "picked_up", "out_for_delivery"])].copy()
    pred = predictor.predict_frame(active)
    out = {"model_version": predictor.version, "test_mae": round(metrics["mae"], 3),
           "eta_minutes": {oid: float(v) for oid, v in zip(active["order_id"], pred["eta_minutes"])}}
    (HERE / "eta_snapshot.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(predictor.version, "mae", out["test_mae"], len(out["eta_minutes"]), "orders")


if __name__ == "__main__":
    main()
