"""Run the pinned authors' functions unchanged; record gaps before adaptation."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
from bope.model import (  # noqa: E402 -- pinned original module
    PairwiseRankingNet, Swish, scale_neural_network,
    utility_model_training_noiseless,
)


def main():
    destination = HERE / "upstream-smoke.json"
    if destination.exists():
        raise SystemExit("Refusing to overwrite upstream evidence")
    torch.set_default_dtype(torch.double)
    torch.set_num_threads(2)
    original = torch.tensor([-3., -1., 0., 1., 3.])
    argument = original.clone()
    output = Swish()(argument)
    activation = {
        "input": original.tolist(), "output": output.tolist(),
        "argument_after_call": argument.tolist(),
        "matches_silu": bool(torch.allclose(output, original * torch.sigmoid(original))),
        "matches_squared_sigmoid": bool(torch.allclose(output, torch.sigmoid(original)**2)),
        "input_mutated": bool(not torch.equal(argument, original)),
    }
    training_log = io.StringIO()
    results = []
    values = torch.tensor([[.9, .4, .8], [.5, .9, .7], [.6, .7, .9],
                           [.8, .6, .5], [.4, .8, .9], [.9, .5, .6]])
    utility = values @ torch.tensor([.5, .2, .3])
    pairs = [(0, 1), (2, 3), (4, 5)]
    for count in (0, 1, 3):
        info = {"Y": [values[[a, b]] for a, b in pairs[:count]],
                "prefer": [1 if utility[a] >= utility[b] else -1
                           for a, b in pairs[:count]]}
        started = perf_counter()
        try:
            with redirect_stdout(training_log):
                model, unused_index = utility_model_training_noiseless(
                    3, info, monotonic=True, seed=123, Bayesian=False)
            with torch.no_grad():
                scores = model(values, values)[0].flatten()
            result = {"observations": count, "status": "PASS",
                      "scores": scores.tolist(),
                      "finite": bool(torch.isfinite(scores).all()),
                      "training_seconds": perf_counter()-started}
        except Exception as exc:
            result = {"observations": count, "status": "FAIL",
                      "error_type": type(exc).__name__, "error": str(exc),
                      "training_seconds": perf_counter()-started}
        results.append(result)
    torch.manual_seed(99)
    untrained = PairwiseRankingNet(3, True)
    duplicate = torch.ones((2, 3)) * .5
    raw = untrained(duplicate, duplicate)[0]
    std = raw.std().item()
    if std == 0:
        std = (raw.max()-raw.min()).item()
    normalized = scale_neural_network(untrained, raw.max().item(), std)
    scaled = normalized(duplicate, duplicate)[0]
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": json.loads((HERE/'upstream/SOURCE.json').read_text())['commit'],
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "threads": torch.get_num_threads(),
        "scope": "Original model training function only; not paper benchmark reproduction",
        "activation_audit": activation, "training": results,
        "degenerate_scale_audit": {"original_std_after_fallback": std,
                                    "finite": bool(torch.isfinite(scaled).all())},
        "smoke_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE/'upstream-training.log').write_text(training_log.getvalue())
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
