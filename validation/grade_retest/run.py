"""Fixed, explicitly hypothetical test-retest simulation of the product rubric."""
import argparse
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.assessment_bank import RUBRIC, SKILLS, answer_correct, build_test, score_test


def kappa(a, b):
    matrix = np.zeros((2, 2), dtype=int)
    for x, y in zip(a, b):
        matrix[int(x), int(y)] += 1
    n = int(matrix.sum())
    observed = float(np.trace(matrix)/n)
    expected = float(matrix.sum(0) @ matrix.sum(1)/(n*n))
    return {"agreement": observed, "kappa": (observed-expected)/(1-expected) if expected < 1 else None,
            "matrix_A_rows_B_columns_fail_pass": matrix.tolist(), "pairs": n}


def run(destination, *, smoke=False):
    destination.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    sources = [Path(__file__), protocol_path] + [ROOT / "backend/app" / p for p in
              ("assessment_bank.py", "practical_bank.py", "practical_bank_v2.py", "retry_item.py")]
    pins = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources}
    (destination / "frozen_inputs.json").write_text(json.dumps({"sha256": pins, "rubric": RUBRIC,
        "scope": "SMOKE_ONLY" if smoke else "FIXED_FULL_SIMULATION"}, ensure_ascii=False, indent=2))
    # Analytic checks before generating any outcomes; this is not bank calibration.
    assert float(expit(0)) == .5
    assert abs(kappa([0,0,0,1,1,1], [0,0,1,0,1,1])["kappa"]-1/3) < 1e-12
    rng = np.random.default_rng(protocol["seed"] + (999999 if smoke else 0))
    count = 1 if smoke else protocol["profiles_per_ability"]
    summaries, family_results, total = [], [], 0
    with (destination / "pairs.jsonl").open("x") as output:
        for scenario, options in protocol["scenarios"].items():
            for specialization in protocol["specializations"]:
                for grade in protocol["grades"]:
                    pairs, by_ability, families = [], defaultdict(list), defaultdict(lambda: defaultdict(list))
                    for theta in protocol["abilities"]:
                        for person in range(count):
                            abilities = theta + rng.normal(0, .35, 3)
                            abilities[0] += options["weak_skill_offset"]
                            forms = []
                            for form in ("A", "B"):
                                seed = int(rng.integers(1_000_000_000, 2_000_000_000))
                                tasks = build_test(specialization, grade, seed)
                                assert len(tasks) == 12
                                shift = rng.normal(0, options["form_shift_sd"])
                                answers, bits, item_rows = {}, {}, []
                                for task in tasks:
                                    practical = task.get("answer_type") == "fields"
                                    ability = abilities[SKILLS[specialization].index(task["skill"])]
                                    offset = int(sha256(task["family"].encode()).hexdigest()[:8], 16)/(2**32-1)*.8-.4
                                    difficulty = (.7 if grade == "Middle" else 0) + (.3 if practical else 0) + offset
                                    guess = 0 if practical else 1/len(task["choices"])
                                    probability = guess + (.98-guess)*float(expit((1.5 if practical else 1.2)*(ability-difficulty-shift)))
                                    correct = bool(rng.random() < probability)
                                    if practical:
                                        answer = dict(task["_correct"])
                                        if not correct:
                                            key = next(iter(answer)); answer[key] = str(Decimal(answer[key])+1)
                                    else:
                                        answer = task["_correct"] if correct else (task["_correct"]+1)%len(task["choices"])
                                    assert answer_correct(task, answer) == correct
                                    answers[task["id"]], bits[task["id"]] = answer, correct
                                    if theta >= 1: families[task["family"]]["upper"].append(int(correct))
                                    if theta <= -1: families[task["family"]]["lower"].append(int(correct))
                                    item_rows.append({"family": task["family"], "probability": probability, "correct": correct})
                                result = score_test(tasks, answers)
                                by_skill, practice = defaultdict(list), defaultdict(list)
                                for task in tasks:
                                    by_skill[task["skill"]].append(bits[task["id"]])
                                    if task.get("answer_type") == "fields": practice[task["skill"]].append(bits[task["id"]])
                                independent_pass = (sum(bits.values()) >= 9 and all(sum(v)*2 >= len(v) for v in by_skill.values())
                                                    and all(sum(practice[s])*2 >= len(practice[s]) for s in by_skill))
                                assert result["correct"] == sum(bits.values()) and result["passed"] == independent_pass
                                forms.append({"form": form, "seed": seed, "score": result["score"], "passed": result["passed"], "items": item_rows})
                            assert set(x["family"] for x in forms[0]["items"]) == set(x["family"] for x in forms[1]["items"])
                            row = {"scenario": scenario, "specialization": specialization, "claimed_grade": grade,
                                   "theta": theta, "profile": person, "skill_abilities": abilities.tolist(), "forms": forms}
                            output.write(json.dumps(row, ensure_ascii=False)+"\n")
                            pairs.append(forms); by_ability[theta].append(forms); total += 1
                    stats = kappa([p[0]["passed"] for p in pairs], [p[1]["passed"] for p in pairs])
                    stats.update({"scenario": scenario, "specialization": specialization, "claimed_grade": grade,
                        "mean_absolute_score_difference": float(np.mean([abs(p[0]["score"]-p[1]["score"]) for p in pairs])),
                        "by_ability": {str(t): {**kappa([p[0]["passed"] for p in ps], [p[1]["passed"] for p in ps]),
                            "pass_rate_A": float(np.mean([p[0]["passed"] for p in ps])), "pass_rate_B": float(np.mean([p[1]["passed"] for p in ps]))} for t,ps in by_ability.items()}})
                    summaries.append(stats)
                    for family, groups in sorted(families.items()):
                        family_results.append({"scenario": scenario, "specialization": specialization, "grade": grade, "family": family,
                            "upper_n": len(groups["upper"]), "lower_n": len(groups["lower"]),
                            "upper_minus_lower": float(np.mean(groups["upper"])-np.mean(groups["lower"]))})
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p,digest in pins.items())
    summary = {"status": "PASS_SIMULATION_EXECUTED", "human_validation": "NOT_RUN", "profiles_pairs": total,
               "forms": total*2, "responses": total*24, "rubric": RUBRIC, "summaries": summaries, "family_discrimination": family_results,
               "limitation": protocol["decision"], "input_sha256": pins,
               "rows_sha256": sha256((destination/"pairs.jsonl").read_bytes()).hexdigest()}
    (destination/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:summary[k] for k in ("status", "profiles_pairs", "forms", "responses")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.destination, smoke=args.smoke)
