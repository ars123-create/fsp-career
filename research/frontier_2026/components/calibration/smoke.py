"""Save probability extraction diagnostics; no benchmark truth or fitting."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .predict import common_query_ids, expected_scores, prediction_report
from ...policies import Config as PConfig, ParticleSession
from ..constrained_gp.adapter import Config as GConfig, Session as Constrained
from ..distillation.adapter import Session as Distillation
from ..legacy_controls.adapter import Equal, GaussianTopK, LinearEVOI
from ..monne.adapter import Config as MConfig, MoNNE
from ..risk_decisions.adapter import Config as RConfig, Session as Risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    destination = parser.parse_args().output
    if destination.exists():
        raise SystemExit("Refusing to overwrite previous probability audit")
    values = np.array([[.1,.9],[.3,.7],[.7,.3],[.9,.1],[.6,.6],[.2,.2],[.8,.8]])
    ids = common_query_ids(values)
    sessions = [(name, ParticleSession(values, PConfig(policy=name, m=2)), None)
                for name in ("kg", "fractional", "latent_noise", "answerability")]
    sessions += [
        ("P06", Constrained(values, GConfig(particles=128, proposal_batch=128, minimum_ess=2)), None),
        ("P20", Distillation(values), None), ("legacy_linear", LinearEVOI(values), None),
        ("legacy_gp", GaussianTopK(values), None),
        ("MoNNE", MoNNE(values, MConfig(ensemble_size=2)), None),
        ("risk_bayes", Risk(values, RConfig(m=2)), "kl_dro"),
        ("risk_mmr", Risk(values, RConfig(m=2)), "mmr"),
        ("equal", Equal(values), None),
    ]
    rows = []
    for name, session, method in sessions:
        report = prediction_report(session, ((ids[0], 2),), ids, risk_method=method)
        row = asdict(report)
        row["method"] = name
        if report.probabilities is not None:
            row["probabilities"] = report.probabilities.tolist()
            row["max_row_sum_error"] = float(np.max(np.abs(report.probabilities.sum(axis=1)-1)))
        rows.append(row)
    scores = expected_scores([[.6,.3,.1]], [[.2,.5,.3]])
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Probability adapter engineering smoke only, not benchmark calibration results",
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "values": values.tolist(), "common_queries": ids, "history": [[ids[0],2]],
              "rows": rows, "analytic_score_example": {k: v.tolist() for k,v in scores.items()},
              "final_benchmark": "NOT RUN"}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(destination), "statuses": {r["method"]: r["status"] for r in rows}}, indent=2))


if __name__ == "__main__":
    main()
