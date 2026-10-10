"""Execute the prespecified descriptive stress suite; never select a winner."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from .benchmark import HERE, WORKSPACE, write_json


def main():
    path = HERE/'STRESS_PROTOCOL.json'
    if sha256(path.read_bytes()).hexdigest() != path.with_suffix('.sha256').read_text().split()[0]:
        raise RuntimeError('Stress plan changed')
    plan = json.loads(path.read_text())
    out = HERE/'evidence/stress-comparison-v1'
    out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes(path.read_bytes())
    failures = []
    for arm in plan['arms']:
        command = [sys.executable,'-m','research.frontier_2026.benchmark',
                   '--out',str(out/arm['name']),'--stage','STRESS',
                   '--seed-start',str(arm['seed_start']),'--seeds',str(arm['seeds']),
                   '--n',str(arm['n']),'--d',str(arm['d']),'--variant',arm['variant'],
                   '--families',*plan['families'],'--methods',*plan['methods']]
        with (out/f"{arm['name']}.log").open('x') as stream:
            result = subprocess.run(command,cwd=WORKSPACE,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append({'arm':arm['name'],'exit_code':result.returncode})
        print(json.dumps({'arm':arm['name'],'exit_code':result.returncode}),flush=True)
    write_json(out/'completion.json', {'status':'COMPLETE' if not failures else 'FAIL',
               'arms':len(plan['arms']),'failed_arms':failures,
               'scope':'Prespecified descriptive stress; no final winner selection'})


if __name__ == '__main__':
    main()
