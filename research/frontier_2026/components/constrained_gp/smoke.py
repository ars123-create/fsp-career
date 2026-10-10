"""Fixed engineering smoke; independent of hidden utility and heldout ranking."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import ndtr

from .adapter import Config, Session


HERE = Path(__file__).resolve().parent


def analytic_checks():
    cfg = replace(Config(), particles=16384, max_proposals=65536,
                  amplitude=1., lengthscale=1., seed=2026100842)
    s = Session([[0.], [1.]], cfg)
    difference = s.utilities[:, 1]-s.utilities[:, 0]
    variance = 2*(cfg.amplitude**2*(1-np.exp(-.5))+cfg.covariance_jitter)
    sd = np.sqrt(variance)
    exact = 1+sd*np.exp(-.5/sd**2)/np.sqrt(2*np.pi)/ndtr(1/sd)
    mcse = difference.std(ddof=1)/np.sqrt(len(difference))
    first = {"case": "one hard dominance constraint", "particles": cfg.particles,
             "seed": cfg.seed, "analytical_mean_difference": float(exact),
             "mc_mean_difference": float(difference.mean()), "mcse": float(mcse),
             "absolute_error_in_mcse": float(abs(difference.mean()-exact)/mcse)}
    cfg = replace(Config(), particles=16384, max_proposals=16384,
                  amplitude=.4, lapse=0., seed=2026100843)
    s = Session([[.2, .8], [.8, .2]], cfg)
    a, b = s.pairs[0]
    difference = s.utilities[:, a]-s.utilities[:, b]
    variance = 2*(cfg.amplitude**2*(1-np.exp(-.72/(2*cfg.lengthscale**2)))+cfg.covariance_jitter)
    exact = variance/np.sqrt(variance+cfg.probit_noise**2)*np.sqrt(2/np.pi)
    weights = s.posterior_weights(((0, 0),))
    estimate = weights @ difference
    mcse = np.sqrt(len(weights)/(len(weights)-1)*np.sum(weights**2*(difference-estimate)**2))
    second = {"case": "one probit comparison", "particles": cfg.particles,
              "seed": cfg.seed, "analytical_mean_difference": float(exact),
              "mc_mean_difference": float(estimate), "mcse": float(mcse),
              "absolute_error_in_mcse": float(abs(estimate-exact)/mcse),
              "posterior_ess": float(1/np.sum(weights**2))}
    return [first, second]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    destination = parser.parse_args().output
    if destination.exists():
        raise SystemExit("Choose a new output path; previous evidence is preserved")
    rng = np.random.default_rng(2026100844)
    pools = [
        ("six_profiles", np.array([[.9,.1],[.1,.9],[.5,.5],[.8,.8],[.2,.2],[.3,.7]])),
        ("uniform_30_three_skills", rng.uniform(size=(30, 3))),
        ("uniform_100_three_skills", rng.uniform(size=(100, 3))),
        ("ordered_100_one_skill", np.linspace(0, 1, 100)[:, None]),
    ]
    runs = []
    for name, values in pools:
        started = time.perf_counter()
        try:
            session = Session(values)
            sampling_seconds = time.perf_counter()-started
            history, steps = [], []
            for outcome in (0, 2, 1):
                before = time.perf_counter()
                selected, query = session.recommend(tuple(history))
                steps.append({"history": list(history), "selected": selected, "query": query,
                              "recommend_seconds": time.perf_counter()-before,
                              "diagnostics": session.diagnostics(tuple(history), query=query)})
                if query is None:
                    break
                history.append((query, outcome))
            final_selected, next_query = session.recommend(tuple(history))
            runs.append({"case": name, "status": "PASS", "pool": values.tolist(),
                         "profiles": len(values), "sampling_seconds": sampling_seconds,
                         "seconds": time.perf_counter()-started, "steps": steps,
                         "final_history": history, "final_selection": final_selected,
                         "query_after_history": next_query})
        except Exception as exc:
            runs.append({"case": name, "status": "FAIL", "pool": values.tolist(),
                         "profiles": len(values), "error_type": type(exc).__name__,
                         "error": str(exc), "diagnostics": getattr(exc, "diagnostics", None),
                         "seconds": time.perf_counter()-started})
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Engineering development smoke, no hidden utility, not final heldout",
        "heldout_status": "NOT RUN", "config": asdict(Config()), "pool_seed": 2026100844,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "runs": runs, "analytical_checks": analytic_checks(),
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                          for path in (HERE/"adapter.py", HERE/"smoke.py", HERE/"test_adapter.py")},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(destination),
                      "runs": [{k: v for k, v in run.items()
                                if k in {"case", "status", "seconds", "error"}} for run in runs],
                      "analytical_checks": result["analytical_checks"]}, indent=2))
    if not all(run["status"] == "PASS" for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
