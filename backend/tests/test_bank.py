"""Independent executable oracles for the released demo task families."""
import ast
import contextlib
import io
import re
import sqlite3
import statistics
from collections import defaultdict
from hypothesis import given, settings, strategies as st
from app.assessment_bank import build_test_v1 as build_test, public_questions, score_test


def reference(q, grade):
    family = q["family"]; nums = [int(n) for n in re.findall(r"\d+", q["prompt"])]
    if family.startswith("python_"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            # The source is produced by our local fixed generator, never by a user.
            exec(q["code"], {"__builtins__": {"print": print, "sum": sum, "enumerate": enumerate}})
        return float(output.getvalue().strip())
    if family == "sql_null_distinct":
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE events(value INTEGER)")
        db.executemany("INSERT INTO events VALUES(?)", [(v,) for v in ast.literal_eval(q["code"].splitlines()[0][8:])])
        result = db.execute(q["code"].splitlines()[1]).fetchone()[0]; db.close(); return result
    if family == "sql_join_cardinality":
        left, matches, extra = nums[:3]
        db = sqlite3.connect(":memory:")
        db.executescript("CREATE TABLE users(id INTEGER); CREATE TABLE orders(id INTEGER, user_id INTEGER);")
        db.executemany("INSERT INTO users VALUES(?)", [(i,) for i in range(left)])
        db.executemany("INSERT INTO orders VALUES(?,?)", [(i, i) for i in range(matches)] + [(matches+i, 0) for i in range(extra)])
        result = len(db.execute(q["code"]).fetchall()); db.close(); return result
    if family == "api_idempotency":
        balance, amount, repeats = nums[:3]; processed = set()
        for _ in range(1 + repeats):
            key = "same-operation"
            if key not in processed: balance -= amount; processed.add(key)
        return balance
    if family == "api_concurrency":
        if grade == "Junior": return len(["initial"] + ["retry"] * nums[0])
        db = sqlite3.connect(":memory:"); db.execute("CREATE TABLE stock(id, count, version)")
        db.execute("INSERT INTO stock VALUES(1, ?, 7)", (nums[0],))
        for _ in range(nums[1]): db.execute(q["code"])
        result = db.execute("SELECT count FROM stock").fetchone()[0]; db.close(); return result
    if family == "analytics_median": return statistics.median(ast.literal_eval(q["prompt"].split(": ")[1].rstrip(".")))
    if family == "analytics_conversion":
        if grade == "Junior": return round(nums[1] / nums[0] * 100, 2)
        records = [1] * nums[1] + [0] * (nums[0] - nums[1]) + [1] * nums[3] + [0] * (nums[2] - nums[3])
        return round(statistics.mean(records) * 100, 2)
    if family == "quality_dedup":
        rows, missing, duplicated = nums[:3]
        keys = [None] * missing + ["duplicate"] * (duplicated + 1) + list(range(rows - missing - duplicated - 1))
        return len(set(k for k in keys if k is not None))
    if family == "quality_pipeline":
        if grade == "Junior": return len(list(range(nums[0]))[nums[1]:])
        return len(set(list(range(nums[0])) + list(range(nums[1]))))
    raise AssertionError(f"Missing independent oracle: {family}")


@settings(max_examples=80, derandomize=True, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31-1), grade=st.sampled_from(["Junior", "Middle"]), spec=st.sampled_from(["backend", "data"]))
def test_generated_questions_against_independent_execution(seed, grade, spec):
    questions = build_test(spec, grade, seed)
    assert len(questions) == 6 and len({q["family"] for q in questions}) == 6
    assert len({q["skill"] for q in questions}) == 3
    for q in questions:
        assert len(q["choices"]) == len(set(q["choices"])) == 4
        assert float(q["choices"][q["_correct"]]) == reference(q, grade)
        assert "_correct" not in str(public_questions([q]))


def test_forms_change_both_answers_and_positions():
    for grade in ("Junior", "Middle"):
        values = defaultdict(set); forms = set()
        for seed in range(100):
            questions = build_test("backend", grade, seed)
            forms.add(tuple(q["family"] for q in questions))
            for q in questions: values[q["family"]].add(q["choices"][q["_correct"]])
        assert len(forms) > 20
        assert all(len(v) >= 3 for v in values.values()), values


def test_competency_floor_cannot_be_hidden_by_average():
    questions = build_test("backend", "Junior", 57)
    answers = {q["id"]: q["_correct"] if q["skill"] != "SQL" else (q["_correct"] + 1) % 4 for q in questions}
    result = score_test(questions, answers)
    assert result["skills"]["SQL"] == 0 and not result["passed"]
