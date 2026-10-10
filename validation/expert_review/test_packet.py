"""Technical fixtures only; no test response represents an independent person."""
import copy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from pydantic import ValidationError
from app.assessment_bank import public_questions
from validation.expert_review.packet import FIELDS, Rating, Review, build, collect, encoded


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "packet"
    key = build(path)
    return path, key


def template(path, name="backend-A"):
    return json.loads((path / name / "review.json").read_text())


def filled(path, name="backend-A", origin="synthetic_fixture", reviewer="fixture-A"):
    review = template(path, name)
    questions = json.loads((path / name / "questions.json").read_text())
    tasks = {c["case_id"]: c["task"] for c in questions["cases"]}
    review.update(origin=origin, reviewer_id=reviewer, relevant_background="Technical test fixture only",
                  independent=True, used_ai=False, saw_reference_answers=False)
    for r in review["ratings"]:
        q = tasks[r["case_id"]]
        r.update(status="solved", clarity="clear", difficulty="basic", work_relevance=1)
        if q["answer_type"] == "choice":
            r["choice"] = 1
        else:
            r["values"] = {f['name']: "0" for f in q['fields']}
    return review


def test_packets_keep_complete_context_without_answer_or_grade(bundle):
    path, key = bundle
    assert len(key["cases"]) == 48
    assert len(key["packets"]) == 4
    expected = {"questions.json", "review.json", "QUESTIONS_RU.txt", "INSTRUCTIONS_RU.txt", "review.schema.json"}
    for meta in key["packets"].values():
        name = meta["directory"]
        packet = json.loads((path / name / "questions.json").read_text())
        assert len(packet["cases"]) == 24
        for case in packet["cases"]:
            original = key["cases"][case["case_id"]]["task"]
            public = public_questions([original])[0]
            assert case["task"] == {k: v for k, v in public.items() if k in FIELDS}
            assert case["task"]["code"] in (path / name / "QUESTIONS_RU.txt").read_text()
            assert not {"grade", "seed", "family", "id", "_correct", "_explanation"} & set(case["task"])
        with ZipFile(path / (name + ".zip")) as z:
            assert set(z.namelist()) == expected
            assert z.testzip() is None
            for filename in expected:
                assert z.read(filename) == (path / name / filename).read_bytes()
            text = z.read("questions.json").decode()
            assert all(f'"{s}"' not in text for s in ["grade", "seed", "family", "_correct", "_explanation", "correct_values"])


def test_repeatable_archives_and_different_order(bundle, tmp_path):
    path, key = bundle
    other = tmp_path / "second"
    key2 = build(other)
    assert key2 == key
    for p in path.glob("*.zip"):
        assert p.read_bytes() == (other / p.name).read_bytes()
    for spec in ["backend", "data"]:
        a = template(path, spec + "-A")["ratings"]
        b = template(path, spec + "-B")["ratings"]
        assert a != b
        assert {r["case_id"] for r in a} == {r["case_id"] for r in b}


def test_never_overwrites_existing_reviews(bundle):
    path, _ = bundle
    target = path / "backend-A" / "review.json"
    target.write_text("preserve user's draft")
    with pytest.raises(FileExistsError):
        build(path)
    assert target.read_text() == "preserve user's draft"


def test_empty_reviews_are_not_run(bundle):
    path, _ = bundle
    for reviews in ([], [template(path)]):
        result = collect(path, reviews)
        assert result["content_review"] == "NOT RUN"
        assert result["human_validation"] == "NOT RUN"
        assert not result["complete_declared_blind_reviews"]


def test_fixture_cannot_be_reported_as_human_evidence(bundle):
    path, _ = bundle
    result = collect(path, [filled(path)])
    assert result["content_review"] == "TECHNICAL_SMOKE"
    assert result["human_validation"] == "NOT RUN"
    assert result["reviews"][0]["completed"] == 24
    assert not result["complete_declared_blind_reviews"]


def test_human_declaration_still_requires_manual_verification(bundle):
    path, _ = bundle
    reviews = [filled(path, spec + "-" + label, "human", "fixture-" + label)
               for spec in ["backend", "data"] for label in ["A", "B"]]
    result = collect(path, reviews)
    assert result["content_review"] == "DECLARED REVIEWS RECEIVED"
    assert result["grade_calibration"] == result["human_validation"] == "NOT RUN"
    assert result["manual_provenance_and_content_review_required"]


@pytest.mark.parametrize("field,value", [("used_ai", True), ("independent", False), ("saw_reference_answers", True), ("relevant_background", " ")])
def test_unblinded_or_undocumented_does_not_count_as_blind(bundle, field, value):
    path, _ = bundle
    review = filled(path, origin="human")
    review[field] = value
    result = collect(path, [review])
    assert result["content_review"] == "PARTIAL"
    assert not result["complete_declared_blind_reviews"]


@pytest.mark.parametrize("value", [True, "2", 4, -1])
def test_strict_relevance_rejects_coercion_and_out_of_range(value):
    with pytest.raises(ValidationError):
        Rating(case_id="Q001", work_relevance=value)


def test_unknown_fields_are_not_silently_dropped(bundle):
    path, _ = bundle
    review = template(path)
    review["ratings"][0]["invented_score"] = 99
    with pytest.raises(ValidationError):
        collect(path, [review])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign"])
def test_exact_assignment_required(bundle, mutation):
    path, _ = bundle
    review = template(path)
    if mutation == "missing": review["ratings"].pop()
    elif mutation == "duplicate": review["ratings"][-1] = copy.deepcopy(review["ratings"][0])
    else: review["ratings"][0]["case_id"] = "Q999"
    with pytest.raises(ValueError, match="case IDs"):
        collect(path, [review])


def test_same_person_and_duplicate_packet_rejected(bundle):
    path, _ = bundle
    a = filled(path, reviewer="Fixture")
    b = filled(path, "backend-B", reviewer=" fixture ")
    with pytest.raises(ValueError, match="Same reviewer"):
        collect(path, [a, b])
    with pytest.raises(ValueError, match="duplicate packet"):
        collect(path, [a, a])


def test_questions_cannot_change_after_review_assignment(bundle):
    path, _ = bundle
    p = path / "backend-A" / "questions.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Questions changed"):
        collect(path, [template(path)])


def test_partial_observation_requires_origin(bundle):
    path, _ = bundle
    review = template(path)
    review["ratings"][0]["clarity"] = "clear"
    with pytest.raises(ValueError, match="declare their origin"):
        collect(path, [review])


def test_bad_numeric_answer_and_incomplete_form(bundle):
    path, _ = bundle
    review = filled(path)
    field = next(r for r in review["ratings"] if r["values"] is not None)
    field["values"][next(iter(field["values"]))] = "NaN"
    with pytest.raises(ValueError): collect(path, [review])
    review = filled(path, origin="human")
    review["ratings"][0]["clarity"] = None
    result = collect(path, [review])
    assert result["content_review"] == "PARTIAL"
    assert result["reviews"][0]["completed"] == 23


def test_limitations_need_reason_and_cannot_mix_origins(bundle):
    path, _ = bundle
    review = filled(path)
    review["ratings"][0]["clarity"] = "ambiguous"
    with pytest.raises(ValueError, match="comments"):
        collect(path, [review])
    with pytest.raises(ValueError, match="mix synthetic"):
        collect(path, [filled(path), filled(path, "backend-B", "human", "fixture-B")])
