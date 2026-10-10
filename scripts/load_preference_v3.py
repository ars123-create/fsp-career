"""Read-only load of actual preference searches on the local synthetic demo.

Reuses the v2 httpx workload; every worker sends ten sequential requests, keeps
its own answer history and alternates left/right/skip. No database writes.
"""
import asyncio
import json
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
from sqlmodel import Session, select
from app.core.db import engine
from app.core.security import issue_access
from app.models import User


async def main():
    target = Path("output/validation/load-preference-v3.json")
    if target.exists():
        raise SystemExit("Refusing to overwrite load results.")
    with Session(engine) as db:
        users = db.exec(select(User).where(User.is_demo == True, User.role == "employer")).all()
        assert users, "Local synthetic employer required"
        tokens = [issue_access(u) for u in users]
    rows = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8018", timeout=40.) as client:
        health = (await client.get("/api/health")).json()
        assert health["version"] == "0.3.0-dev", health["version"]
        async def worker(index):
            payload = None
            for step in range(10):
                if payload is None:
                    payload = {"specialization":"backend", "grade":"Middle", "ranking":{"mode":"preference"}}
                start = perf_counter()
                try:
                    response = await client.post("/api/search", headers={"Authorization":f"Bearer {tokens[index % len(tokens)]}"}, json=payload)
                    row = {"client":index, "step":step, "seconds":perf_counter()-start, "status":response.status_code}
                    if response.status_code == 200:
                        result = response.json()
                        report = result["preference"]
                        row.update(algorithm=result["algorithm"], considered=report["considered"],
                                   answered=report["answered"], skipped=report["skipped"], changed=report["changed_count"])
                        if report["question"]:
                            q = report["question"]
                            payload = result["normalized"]
                            payload["ranking"]["comparisons"].append({"left":q["left"],"right":q["right"],"choice":("left","right","skip")[index % 3]})
                        else:
                            payload = None
                    else:
                        payload = None
                    rows.append(row)
                except httpx.HTTPError as exc:
                    rows.append({"client":index,"step":step,"seconds":perf_counter()-start,"status":type(exc).__name__})
                    payload = None
        await asyncio.gather(*(worker(i) for i in range(30)))
    result = {"scenario":"30 concurrent clients, 10 sequential searches each, alternating roots and independently answered/skipped queries; read-only",
              "platform":"local macOS, SQLite, uvicorn; 8 eligible synthetic profiles; not a production or 100-profile load test",
              "version":health["version"], "requests":len(rows), "sessions":len(tokens),
              "http_200":sum(r["status"]==200 for r in rows),
              "p95_ms":float(np.quantile([r["seconds"] for r in rows],.95)*1000),
              "max_ms":max(r["seconds"] for r in rows)*1000,
              "skip_invariant_failures":sum(r.get("skipped",0)>0 and r.get("changed",0)!=0 for r in rows), "rows":rows}
    with target.open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__ == "__main__":
    asyncio.run(main())
