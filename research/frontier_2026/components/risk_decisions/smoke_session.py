"""Feature-only n=18/n=100 execution probe, never a quality benchmark."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import scipy

from research.frontier_2026.components.risk_decisions import Config, Session


def run():
    root = Path(__file__).resolve().parent
    config = Config()
    output = {"utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "constructed feature-only execution and timing smoke; no truth, no quality scores",
              "feature_fixture_seed": 6100829, "config": asdict(config),
              "numpy": np.__version__, "scipy": scipy.__version__,
              "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                for p in [root/'adapter.py',root/'decisions.py']},
              "runs": []}
    for n in (18,100):
        x = np.random.default_rng(6100829+n).uniform(0.,1.,(n,3))
        session = Session(x,config)
        for method in ('kl_dro','cvar','mmr','baseline_safe','edgecut'):
            start = time.perf_counter()
            record = {"n": n,"d": 3,"method": method,"questions": len(session.pairs)}
            try:
                selected,query = session.recommend(method=method)
                assert len(selected) == len(set(selected)) == 5
                assert query is None or 0 <= query < len(session.pairs)
                record.update(status="PASS",selection=selected,query=query,
                              diagnostics=session.last_diagnostics)
            except Exception as error:
                record.update(status="FAIL",error_type=type(error).__name__,error=str(error))
            record['seconds'] = time.perf_counter()-start
            output['runs'].append(record)
            print(json.dumps({key:record[key] for key in ('n','method','status','seconds')}),flush=True)
    output['status'] = 'PASS' if all(r['status']=='PASS' for r in output['runs']) else 'FAIL'
    return output


if __name__ == '__main__':
    result = run()
    with Path(__file__).with_name('session-smoke-v1.json').open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
