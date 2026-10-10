"""Offline content-review preparation. Never awards grades or certifies human validity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, Field
from app.assessment_bank import RUBRIC, SPECIALIZATIONS, build_test, public_questions
from app.practical_bank import validate_values

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "fsp-content-review-v1"
FIELDS = ("skill", "title", "prompt", "code", "tables", "choices", "answer_type", "fields")
SEEDS = {("backend", "Junior"): 947001, ("backend", "Middle"): 947002,
         ("data", "Junior"): 947003, ("data", "Middle"): 947004}
EXPECTED_BANKS = {
    "backend/app/assessment_bank.py": "955289281d1c765ec91c520d1a080343ccc22921581323bada057c6da2a0b23d",
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Rating(StrictModel):
    case_id: str
    status: Literal["solved", "not_solved", "ambiguous", "outside_expertise"] | None = None
    choice: int | None = Field(default=None, ge=1, le=4)
    values: dict[str, str] | None = None
    clarity: Literal["clear", "ambiguous", "incorrect", "cannot_rate"] | None = None
    work_relevance: int | None = Field(default=None, ge=0, le=3)
    difficulty: Literal["basic", "junior", "middle", "advanced", "cannot_rate"] | None = None
    comments: str = ""


class Review(StrictModel):
    format: Literal["fsp-content-review-v1"] = FORMAT
    packet_id: str
    origin: Literal["unfilled", "human", "synthetic_fixture"] = "unfilled"
    reviewer_id: str | None = None
    relevant_background: str | None = None
    independent: bool | None = None
    used_ai: bool | None = None
    saw_reference_answers: bool | None = None
    ratings: list[Rating]


INSTRUCTIONS = """Предметная проверка заданий ФСП / Карьера

Это проверка содержания демонстрационного банка. Вы не оцениваете реальных
соискателей и не принимаете решение о найме. Не указывайте чужие персональные
данные. Для reviewer_id достаточно собственного псевдонима.

Откройте QUESTIONS_RU.txt. В нём 24 задания двух форм одной специализации.
Текущие названия уровней и эталоны скрыты. Оцените задания самостоятельно,
не обсуждая ответы со вторым оценщиком. До возврата формы не открывайте код
банка, эталоны и координаторский ключ. Если использовали ИИ или уже видели
ответы, честно укажите это: такой отзыв сохранится отдельно от слепого.

Заполняйте review.json, сохранив packet_id и case_id. origin замените на
human; укажите псевдоним, relevant_background (опыт, относящийся к заданиям),
independent, used_ai и saw_reference_answers как true/false. Это самоотчёт,
не технически подтверждённая идентификация. null означает отсутствие ответа.

Для каждого задания:
status: solved / not_solved / ambiguous / outside_expertise.
При solved заполните либо choice (номер варианта от 1 до 4), либо values:
словарь имён полей и строковых чисел. Например: {"entities": "2", "total": "7"}.
При другом status оставьте choice/values равными null и поясните причину.
clarity: clear / ambiguous / incorrect / cannot_rate.
work_relevance: 0 — связи с работой нет, 1 — слабая, 2 — существенная,
3 — непосредственно полезно в задачах указанной специализации.
Если не можете оценить релевантность, оставьте null и поясните в comments.
difficulty: basic / junior / middle / advanced / cannot_rate.
Уровень одной задачи не равен грейду человека. Полная форма может включать
простые задания. comments: неоднозначность, ошибка, недостающее условие,
нереалистичность или причина невозможности оценить. Пропуск не считается нулём.

Не меняйте QUESTIONS_RU.txt и questions.json. Если нашли ошибку, опишите её
в comments. Возвращайте только заполненный review.json координатору тем
способом, который вы с ним согласовали. Этот пакет сам ничего не отправляет.

Формат решения может быть неверным или не совпасть с авторским эталоном — это
повод проверить задание, а не объявить оценщика неквалифицированным. Этот этап
не устанавливает проходные пороги и не заменяет исследование ответов кандидатов.
"""


def render_questions(packet):
    lines = ["СЛЕПАЯ ПРЕДМЕТНАЯ ПРОВЕРКА", packet["specialization_name"],
             "24 задания. Заданные уровни и эталоны не включены.", ""]
    for case in packet["cases"]:
        q = case["task"]
        lines += [f"{case['case_id']} / форма {case['form_id']} / {q['skill']}",
                  q["title"], q["prompt"]]
        if q.get("code"):
            lines += ["Код / запрос:", q["code"]]
        for table in q.get("tables", []):
            lines += ["Таблица " + table["name"], " | ".join(table["columns"])]
            lines += [" | ".join("NULL" if v is None else str(v) for v in row)
                      for row in table["rows"]]
        if q["answer_type"] == "choice":
            lines += [f"{i}. {choice}" for i, choice in enumerate(q["choices"], 1)]
        else:
            lines += ["Поля ответа:"]
            lines += [f"{field['name']}: {field['label']} (знаков после запятой: {field['places']})"
                      for field in q["fields"]]
        lines += ["", "-" * 60, ""]
    return "\n".join(lines) + "\n"


def build(output: Path):
    """Create a new packet; never overwrite an existing review or coordinator key."""
    version = json.loads((ROOT / "VERSION.json").read_text())
    if version["version"] != "0.4.4-dev":
        raise ValueError("This protocol is fixed to 0.4.4-dev; create a new protocol for changed code")
    snapshot = json.loads((ROOT / "output/validation/source-v0.4.4-dev.json").read_text())
    hashes = {f["path"]: f["sha256"] for f in snapshot["files"]}
    paths = ("backend/app/assessment_bank.py", "backend/app/practical_bank.py")
    for path in paths:
        if digest((ROOT / path).read_bytes()) != hashes[path]:
            raise ValueError("Assessment source changed after the fixed application snapshot")
    for path, expected in EXPECTED_BANKS.items():
        if hashes[path] != expected:
            raise ValueError("Unexpected bank protocol")
    output.mkdir(parents=True, exist_ok=False)
    coordinator = {"format": FORMAT, "version": version, "rubric_version": RUBRIC["version"],
                   "source_hashes": {p: hashes[p] for p in paths}, "cases": {}, "packets": {}}
    grouped = {spec: [] for spec in SPECIALIZATIONS}
    sequence = list(SEEDS.items())
    random.Random(947010).shuffle(sequence)
    case_number = 1
    for form_number, ((spec, grade), seed) in enumerate(sequence, 1):
        tasks = build_test(spec, grade, seed)
        public = public_questions(tasks)
        for task, visible in zip(tasks, public, strict=True):
            case_id = f"Q{case_number:03d}"
            case_number += 1
            case = {"case_id": case_id, "form_id": f"F{form_number}",
                    "task": {k: visible[k] for k in FIELDS if k in visible}}
            grouped[spec].append(case)
            coordinator["cases"][case_id] = {"specialization": spec, "grade": grade,
                                              "seed": seed, "task": task}
    for spec_index, (spec, cases) in enumerate(grouped.items()):
        for label in ("A", "B"):
            ordered = list(cases)
            random.Random(947100 + spec_index * 10 + ord(label)).shuffle(ordered)
            packet = {"format": FORMAT, "source_version": version["version"],
                      "specialization": spec, "specialization_name": SPECIALIZATIONS[spec],
                      "assignment": label, "cases": ordered}
            packet_id = digest(encoded(packet))
            packet["packet_id"] = packet_id
            review = Review(packet_id=packet_id, ratings=[Rating(case_id=c["case_id"]) for c in ordered])
            name = f"{spec}-{label}"
            folder = output / name
            folder.mkdir()
            content = {"questions.json": encoded(packet), "review.json": encoded(review.model_dump()),
                       "QUESTIONS_RU.txt": render_questions(packet).encode(),
                       "INSTRUCTIONS_RU.txt": INSTRUCTIONS.encode(),
                       "review.schema.json": encoded(Review.model_json_schema())}
            for filename, data in content.items():
                (folder / filename).write_bytes(data)
            archive = output / f"{name}.zip"
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
                for filename, data in content.items():
                    entry = ZipInfo(filename, date_time=(2026, 10, 7, 0, 0, 0))
                    entry.compress_type = ZIP_DEFLATED
                    z.writestr(entry, data)
            coordinator["packets"][packet_id] = {"directory": name, "specialization": spec,
                                                  "assignment": label,
                                                  "questions_sha256": digest(content["questions.json"]),
                                                  "archive_sha256": digest(archive.read_bytes())}
    (output / "coordinator-key.json").write_bytes(encoded(coordinator))
    (output / "README_RU.txt").write_text(
        "Пакет подготовлен, оценок людей ещё нет.\n"
        "Передавать оценщику только соответствующий backend-A.zip/backend-B.zip либо data-A.zip/data-B.zip.\n"
        "На каждую специализацию нужны два разных независимых оценщика.\n"
        "coordinator-key.json содержит уровни и эталоны; не включать его в слепую рассылку.\n"
        "Это проверка содержания, не подтверждение валидности грейдов.\n")
    return coordinator


def collect(packet_root: Path, reviews: list[dict]):
    key = json.loads((packet_root / "coordinator-key.json").read_text())
    expected_packets = key["packets"]
    seen_packets, seen_reviewers = set(), set()
    summaries = []
    complete_by_spec = Counter()
    synthetic = False
    for raw in reviews:
        review = Review.model_validate(raw)
        if review.packet_id not in expected_packets or review.packet_id in seen_packets:
            raise ValueError("Unknown or duplicate packet")
        seen_packets.add(review.packet_id)
        meta = expected_packets[review.packet_id]
        data = (packet_root / meta["directory"] / "questions.json").read_bytes()
        if digest(data) != meta["questions_sha256"]:
            raise ValueError("Questions changed after packet creation")
        packet = json.loads(data)
        tasks = {c["case_id"]: c["task"] for c in packet["cases"]}
        ids = [r.case_id for r in review.ratings]
        if len(ids) != len(set(ids)) or set(ids) != set(tasks):
            raise ValueError("Missing, duplicate or foreign case IDs")
        if review.reviewer_id:
            identity = (meta["specialization"], review.reviewer_id.strip().casefold())
            if identity in seen_reviewers:
                raise ValueError("Same reviewer cannot count twice for one specialization")
            seen_reviewers.add(identity)
        completed, issues = 0, []
        for rating in review.ratings:
            q = tasks[rating.case_id]
            if rating.status == "solved":
                if q["answer_type"] == "choice":
                    if rating.choice is None or rating.choice > len(q["choices"]) or rating.values is not None:
                        raise ValueError(f"{rating.case_id}: expected one numbered choice")
                else:
                    if rating.choice is not None:
                        raise ValueError(f"{rating.case_id}: expected the stated answer fields")
                    validate_values(q, rating.values)
            elif rating.choice is not None or rating.values is not None:
                raise ValueError(f"{rating.case_id}: answers require solved status")
            needs_comment = (rating.status not in (None, "solved") or
                             rating.clarity in ("ambiguous", "incorrect", "cannot_rate") or
                             rating.difficulty == "cannot_rate" or
                             (rating.status is not None and rating.work_relevance is None))
            if needs_comment and not rating.comments.strip():
                raise ValueError(f"{rating.case_id}: explain the limitation in comments")
            done = rating.status is not None and rating.clarity is not None and rating.difficulty is not None
            done = done and (rating.work_relevance is not None or bool(rating.comments.strip()))
            completed += bool(done)
            if rating.clarity in ("ambiguous", "incorrect"):
                issues.append(rating.case_id)
        touched = any(r.model_dump() != Rating(case_id=r.case_id).model_dump() for r in review.ratings)
        if review.origin == "unfilled" and (touched or review.reviewer_id):
            raise ValueError("Filled observations must declare their origin")
        synthetic |= review.origin == "synthetic_fixture"
        declared_blind = (review.origin == "human" and bool(review.reviewer_id and review.reviewer_id.strip())
                          and bool(review.relevant_background and review.relevant_background.strip())
                          and review.independent is True and review.used_ai is False
                          and review.saw_reference_answers is False)
        if declared_blind and completed == len(tasks):
            complete_by_spec[meta["specialization"]] += 1
        summaries.append({"specialization": meta["specialization"], "assignment": meta["assignment"],
                          "origin": review.origin, "completed": completed, "expected": len(tasks),
                          "declared_blind": declared_blind, "cases_needing_attention": issues})
    if synthetic and any(s["origin"] == "human" for s in summaries):
        raise ValueError("Do not mix synthetic fixtures with human observations")
    status = "TECHNICAL_SMOKE" if synthetic else (
        "DECLARED REVIEWS RECEIVED" if all(complete_by_spec[s] >= 2 for s in SPECIALIZATIONS)
        else "PARTIAL" if any(s["origin"] == "human" for s in summaries) else "NOT RUN")
    return {"content_review": status, "reviews": summaries,
            "complete_declared_blind_reviews": dict(complete_by_spec),
            "human_validation": "NOT RUN", "grade_calibration": "NOT RUN",
            "manual_provenance_and_content_review_required": True,
            "note": "File declarations are not verified human identity or proof of assessment validity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("collect")
    analyze.add_argument("--packet", type=Path, required=True)
    analyze.add_argument("--reviews", type=Path, nargs="*", default=[])
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        key = build(args.output)
        print(json.dumps({"packets": len(key['packets']), "cases": len(key['cases']), "human_validation": "NOT RUN"}))
    else:
        result = collect(args.packet, [json.loads(p.read_text()) for p in args.reviews])
        args.output.write_bytes(encoded(result))
        print(json.dumps({"content_review": result["content_review"], "human_validation": result["human_validation"]}))


if __name__ == "__main__":
    main()
