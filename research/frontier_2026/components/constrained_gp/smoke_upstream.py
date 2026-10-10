"""Minimal preference prediction with the pinned authors' original SkewGP code.

The kernel/parameter setup follows notebooks/Preference_learning.ipynb.
Copyright (c) 2021 Alessio Benavoli, Dario Azzimonti, Dario Piga; BSD-3-Clause.
Unmodified upstream source and license are retained in upstream/.
This is not a reproduction of all notebook experiments or the 2024 paper.
"""
from contextlib import redirect_stdout
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
import commoncode
from SkewGP import SkewGP


def kernel(x1, x2, params, diag_=False):
    length = params["lengthscale"]["value"]
    variance = params["variance"]["value"]
    if not diag_:
        distance = cdist(np.atleast_2d(x1)/length, np.atleast_2d(x2)/length,
                         metric="sqeuclidean")
    else:
        distance = np.sum((np.atleast_2d(x1)/length-np.atleast_2d(x2)/length)**2, axis=1)
    return variance * np.exp(-.5 * distance)


def delta(x, params):
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "upstream-smoke.json")
    destination = parser.parse_args().output
    if destination.exists():
        raise SystemExit("Refusing to overwrite original upstream evidence")
    params = {
        "lengthscale": {"value": np.array([.5]), "range": np.array([[.01, 2.]]),
                        "transform": commoncode.logexp()},
        "variance": {"value": np.array([1.]), "range": np.array([[.01, 2.]]),
                     "transform": commoncode.logexp()},
        "noise_variance": {"value": np.array([.04]), "range": np.array([[.001, 1.]]),
                           "transform": commoncode.logexp()},
    }
    x = np.array([[0.], [.5], [1.]])
    rows = []
    output = io.StringIO()
    original_rng_state = np.random.get_state()
    try:
        for count in (2, 0):
            w = np.array([[-1., 1., 0.], [0., -1., 1.]])[:count]
            np.random.seed(2026100840)
            model = SkewGP(x, kernel, delta, params, W=w, Z=np.zeros((count, 1)),
                           latent_dim=0, type_y="affine", jitter=1e-8, num_cores=1)
            started = time.perf_counter()
            try:
                with redirect_stdout(output):
                    draws = model.predict(x, nsamples=2048, tune=256, progress=False)
                rows.append({"comparisons": count, "status": "PASS",
                             "shape": list(draws.shape), "finite": bool(np.isfinite(draws).all()),
                             "means": draws.mean(axis=1).tolist(),
                             "seconds": time.perf_counter()-started})
            except Exception as exc:
                rows.append({"comparisons": count, "status": "FAIL",
                             "error_type": type(exc).__name__, "error": str(exc),
                             "seconds": time.perf_counter()-started})
    finally:
        np.random.set_state(original_rng_state)
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "70512acf624adf489517d745fc803d6f1f9d15d0",
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "scope": "Pinned original model prediction, fixed kernel; no hyperparameter fitting or 2024 benchmark reproduction",
        "runs": rows, "log": output.getvalue(),
        "smoke_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if rows[0]["status"] != "PASS" or not rows[0]["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
