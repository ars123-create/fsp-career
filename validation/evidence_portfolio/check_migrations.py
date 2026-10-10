"""Upgrade only fresh isolated SQLite files / unique PostgreSQL schemas."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from dotenv import dotenv_values
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import URL

from app.assessment_bank import build_test, score_test
from app.models import Attempt, Candidate, Evidence, GradeAward, User

ROOT = Path(__file__).resolve().parents[2]


def migrate(url, target):
    env = os.environ.copy();env["FSP_DATABASE_URL"] = url
    result = subprocess.run([sys.executable, "-m", "alembic", *target], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    if result.returncode:
        # Never print the connection URL or environment.
        raise RuntimeError(f"Alembic {target[0]} failed; exit={result.returncode}")


def run(url, populated):
    migrate(url, ["upgrade", "20261007_02"])
    engine = create_engine(url)
    meta = MetaData();meta.reflect(engine)
    with engine.begin() as conn:
        if populated:
            user=User(id="migration-candidate",email="migration@example.org",full_name="Fictional migration",hashed_password="not-a-login-hash",role="candidate")
            candidate=Candidate(user_id=user.id,specialization="backend",grade="Junior",
                                skills={"SQL":{"score":100,"families":2,"checked_at":200,"source":"test"}})
            questions=build_test("backend","Junior",19,version="2026-10-v1")
            answers={q["id"]:q["_correct"] for q in questions}
            attempt=Attempt(id="migration-attempt",candidate_id=user.id,specialization="backend",grade="Junior",
                seed=19,questions=questions,answers=answers,result=score_test(questions,answers),
                status="completed",created_at=100,finished_at=200,expires_at=300)
            for table, values in [("user",user.model_dump()),("candidate",candidate.model_dump(exclude={"skill_portfolio"})),
                                  ("attempt",attempt.model_dump())]:
                conn.execute(meta.tables[table].insert().values(**values))
            for q in questions:
                row=Evidence(candidate_id=user.id,attempt_id=attempt.id,skill=q["skill"],family=q["family"],score=100)
                conn.execute(meta.tables["evidence"].insert().values(**row.model_dump()))
            grant=GradeAward(candidate_id=user.id,attempt_id=attempt.id,specialization="backend",grade="Junior")
            conn.execute(meta.tables["gradeaward"].insert().values(**grant.model_dump()))
        before={name:conn.execute(select(table)).fetchall() for name,table in meta.tables.items() if name!="alembic_version"}
    migrate(url,["upgrade","head"])
    migrate(url,["check"])
    with engine.connect() as conn:
        preserved=all(conn.execute(select(meta.tables[name])).fetchall()==rows for name,rows in before.items())
        new=Table("candidate",MetaData(),autoload_with=engine)
        portfolios=conn.execute(select(new.c.skill_portfolio)).scalars().all()
        revision=conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        assert preserved and all(value=={} for value in portfolios) and revision=="20261007_03"
    engine.dispose()
    return {"populated":populated,"old_rows_and_columns_preserved":preserved,
            "empty_portfolios":len(portfolios),"revision":revision,"alembic_check":"PASS"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("backend",choices=["sqlite","postgresql"])
    args=parser.parse_args();results=[]
    if args.backend=="sqlite":
        for populated in (False,True):
            directory=Path(tempfile.mkdtemp(prefix="fsp-portfolio-migration-"))
            results.append(run(f"sqlite:///{directory/'test.sqlite'}",populated))
    else:
        config=dotenv_values(ROOT/".env")
        url=URL.create("postgresql+psycopg",username="fsp",password=config["FSP_POSTGRES_PASSWORD"],host="127.0.0.1",port=5434,database="fsp")
        admin=create_engine(url)
        for populated in (False,True):
            schema="fsp_migration_"+uuid.uuid4().hex
            with admin.begin() as conn:conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            try:
                isolated=url.update_query_dict({"options":f"-csearch_path={schema}"}).render_as_string(hide_password=False)
                results.append(run(isolated,populated))
            finally:
                with admin.begin() as conn:conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
    output=ROOT/f"output/validation/evidence-portfolio-migrations-{args.backend}.json"
    output.write_text(json.dumps({"backend":args.backend,"results":results},indent=2)+"\n")
    print(output.relative_to(ROOT),"PASS",len(results),"isolated schemas/files")


if __name__=="__main__":main()
