"""Build a captioned guide from actual UI captures, explicitly a montage, not a live recording.

Authoring dependencies: Pillow and local ffmpeg/ffprobe. No external service.
"""
from pathlib import Path
import json
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/video"
OUT.mkdir(parents=True, exist_ok=True)
FONT = ROOT / "backend/app/assets/DejaVuSans.ttf"
BOLD = ROOT / "backend/app/assets/DejaVuSans-Bold.ttf"
BG, PLUM, PINK = "#FAF9FC", "#310F53", "#F52C65"


def lines(draw, text, font, width):
    result = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if line and draw.textlength(trial, font=font) > width:
                result.append(line)
                line = word
            else:
                line = trial
        result.append(line)
    return result


def frame(number, title, file, heading, body, duration):
    canvas = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(canvas)
    d.text((42, 25), "ФСП / КАРЬЕРА · ВИДЕОГИД ПО ПРОТОТИПУ", font=ImageFont.truetype(BOLD, 23), fill=PINK)
    d.text((42, 76), title, font=ImageFont.truetype(BOLD, 47), fill=PLUM)
    source = Image.open(ROOT / "output/playwright" / file).convert("RGB")
    source.thumbnail((1320, 850), Image.Resampling.LANCZOS)
    canvas.paste(source, (42 + (1320 - source.width) // 2, 158 + (850 - source.height) // 2))
    y = 187
    for value, size, color, font_file in [(heading, 34, PLUM, BOLD), (body, 29, "#48404F", FONT)]:
        font = ImageFont.truetype(font_file, size)
        for line in lines(d, value, font, 468):
            d.text((1406, y), line, font=font, fill=color)
            y += size + 17
        y += 34
    assert y < 1010, (number, y)
    d.text((42, 1038), "Монтаж снимков рабочего интерфейса · все профили вымышлены · 07.10.2026", font=ImageFont.truetype(FONT, 21), fill="#655D6E")
    filename = f"frame-{number:02}.png"
    canvas.save(OUT / filename)
    return {"frame": filename, "source": file, "seconds": duration, "title": title}


items = [
    frame(1, "Навыки → подбор → знакомство", "search-final.png", "Два кабинета", "Кандидат подтверждает навыки.\n\nРаботодатель видит основания совпадения.\n\nКонтакты открываются с согласия.", 10),
    frame(2, "1. Проверить компетенции", "assessment-detail.png", "Проверка категории", "6 заданий, 25 минут.\n\nОтветы и история сохраняются.\n\nНеудачная попытка выше не понижает грейд.", 12),
    frame(3, "2. Получить результат и историю", "result.png", "Отдельная категория", "Грейд определяется тестом.\n\nЗарплата, спрос и ФСП ID его не меняют.\n\nБанк пока не откалиброван на людях.", 12),
    frame(4, "3. Найти подходящие профили", "search-final-detail.png", "Условия и факты", "Специализация, грейд, формат и зарплата.\n\nБаллы опираются на сохранённые проверки.\n\nРабота доступна и без ФСП ID.", 12),
    frame(5, "4. Сначала увидеть предложение", "invitation-before.png", "Контакты скрыты", "Кандидат видит компанию, задачи и вилку RUB.\n\nМожет принять или отклонить предложение.", 10),
    frame(6, "5. Разрешить знакомство", "invitation-after.png", "Доступ одной компании", "Принятое приглашение открывает контакт этой компании.\n\nДругая его не получает.\n\nДоступ можно отозвать.", 10),
    frame(7, "Проверить прототип самостоятельно", "search-final.png", "Локальный запуск", "python3 scripts/setup.py\n\ndocker compose up -d --build\n\n127.0.0.1:8019\n\nИнструкция и ограничения — в README и PDF.", 14),
]
concat = "ffconcat version 1.0\n" + "".join(f"file '{v['frame']}'\nduration {v['seconds']}\n" for v in items) + f"file '{items[-1]['frame']}'\n"
(OUT / "frames.ffconcat").write_text(concat)
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise SystemExit("Install official FFmpeg for optional video authoring.")
target = ROOT / "output/submission/FSP_Career_Walkthrough.mp4"
subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-n", "-safe", "1", "-i", str(OUT / "frames.ffconcat"), "-vf", "fps=24", "-fps_mode", "cfr", "-t", str(sum(v["seconds"] for v in items)), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
(ROOT / "output/validation/walkthrough.json").write_text(json.dumps({"format": "captioned montage of actual UI screenshots", "frames": items, "duration_seconds": sum(v["seconds"] for v in items), "audio": False}, ensure_ascii=False, indent=2))
print(target)
