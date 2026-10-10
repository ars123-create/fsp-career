"""Archive this audit's exact source so later app changes cannot rewrite it."""
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/elicitation-audit-0.4.2"

README = """# Воспроизведение аудита способов ввода предпочтений

Это изолированный исследовательский комплект для версии приложения 0.4.2-dev.
Он не является комплектом сдачи платформы и не содержит рабочую БД/контакты.
1200 синтетических задач, 8 методов, 6 сценариев; независимых оценок людей нет.

В корне архива сохранены именно те пути и исходники, которые заморожены в
research/elicitation_controls/protocol.json. Позднейшее приложение может
развиваться отдельно. Python 3.14.7; точные зависимости в requirements.txt.

После распаковки, находясь в корне этого архива:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m research.elicitation_controls.verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=backend .venv/bin/python -m pytest research/elicitation_controls/test_controls.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=backend .venv/bin/python -m research.elicitation_controls.benchmark --split heldout --output /tmp/fsp-elicitation-reproduction.json
```

Последняя команда откажется перезаписывать существующий результат. Сравнивать
cases и summary; elapsed_seconds закономерно отличается. Иная версия Python,
NumPy/SciPy или изменение замороженного исходника прекращает heldout-запуск.
Файлы исходного результата не меняйте. SHA-256 содержимого указан в MANIFEST.

Объяснение допущений и методов: research/elicitation_controls/DESIGN.md.
Все результаты, включая отрицательные: research/elicitation_controls/RESULTS.md.
Числовой ввод передаёт больше информации, чем бинарные ответы; равное время
и когнитивное усилие не установлены. Ошибки ввода — искусственные сценарии.
"""


def dependency_lock():
    pending = ["numpy", "scipy", "fastapi", "pydantic", "email-validator", "pytest"]
    found = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in found:
            continue
        distribution = metadata.distribution(name)
        found[name] = distribution.version
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return "".join(f"{name}=={version}\n" for name, version in sorted(found.items()))


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    paths = set(protocol["hashes"])
    for name, digest in protocol["hashes"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    paths.update({"research/__init__.py", "research/elicitation_controls/__init__.py",
                  "backend/app/__init__.py", "research/elicitation_controls/protocol.json",
                  "research/elicitation_controls/verify.py", "research/elicitation_controls/package.py",
                  "research/elicitation_controls/RESULTS.md", "research/elicitation_controls/results/dev.json",
                  "research/elicitation_controls/results/heldout.json", "LICENSE", "THIRD_PARTY.md"})
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "licenses").iterdir() if p.is_file())
    files = {path: (ROOT / path).read_bytes() for path in sorted(paths)}
    files["README_REPRO_RU.md"] = README.encode()
    files["requirements.txt"] = dependency_lock().encode()
    manifest = {"scope": "Frozen isolated synthetic audit, not full application submission",
                "files": {path: {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
                          for path, raw in sorted(files.items())}}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "FSP_Elicitation_Audit_0.4.2.zip"
    with ZipFile(target, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path, raw in files.items():
            archive.writestr(path, raw)
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for path, item in manifest["files"].items():
            assert sha256(archive.read(path)).hexdigest() == item["sha256"]
    info = {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
            "sha256": sha256(target.read_bytes()).hexdigest(), "files": len(files), "crc_and_hashes": "PASS"}
    with (OUT / "ARCHIVE_VERIFICATION.json").open("x") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
