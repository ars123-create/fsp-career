"""Run sealed quality, then isolated research timing, then captured analysis.

No adaptive stopping, rerun, retuning or removal of method failures. If a phase
fails, preserve all its outputs and stop the orchestration with nonzero status.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from . import benchmark


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality',type=Path,required=True)
    parser.add_argument('--timing',type=Path,required=True);args=parser.parse_args()
    q,t=args.quality.resolve(),args.timing.resolve()
    qp,tp=benchmark.read_protocol(q/'protocol.json'),benchmark.read_protocol(t/'protocol.json')
    if qp['plan'].get('study_role')!='QUALITY' or tp['plan'].get('study_role')!='TIMING':
        raise ValueError('Wrong study roles')
    if benchmark.WORKSPACE!=q/'snapshot':raise RuntimeError('Orchestrator must run from captured quality snapshot')
    for key in ('source_sha256','versions','methods','primary_reference'):
        if qp[key]!=tp[key]:raise ValueError('Quality/timing frozen provenance differs')
    benchmark.verify_snapshot(qp,q/'snapshot');benchmark.verify_snapshot(tp,t/'snapshot')
    if qp['versions']!=benchmark.versions():raise ValueError('Runtime versions changed')
    with (q/'orchestration.jsonl').open('x') as log:
        def event(phase):
            value={'phase':phase,'host_utc':datetime.now(timezone.utc).isoformat()}
            log.write(json.dumps(value)+'\n');log.flush();print(json.dumps(value),flush=True)
        event('QUALITY_START')
        benchmark.launch(q/'protocol.json')
        event('QUALITY_ALL_WORKERS_AND_MERGE_COMPLETE')
        event('TIMING_SINGLE_WORKER_START')
        benchmark.launch(t/'protocol.json')
        event('TIMING_COMPLETE')
        event('CAPTURED_ANALYSIS_START')
        subprocess.run([sys.executable,'-m','research.frontier_2026.domain_transfer.analyze',
                        '--quality',str(q),'--timing',str(t)],cwd=q/'snapshot',check=True)
        event('STUDY_COMPLETE')


if __name__=='__main__':main()
