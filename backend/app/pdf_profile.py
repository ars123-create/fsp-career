"""ReportLab Platypus flowables, no external resource loading or user HTML."""
from datetime import datetime, UTC
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from app.schemas import CandidateView

FONT_DIR = Path(__file__).parent / "assets"
pdfmetrics.registerFont(TTFont("FSP", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("FSPBold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("FSP", normal="FSP", bold="FSPBold")


def profile_pdf(profile: CandidateView) -> bytes:
    output = BytesIO()
    styles = {"body": ParagraphStyle("body", fontName="FSP", fontSize=10, leading=16, spaceAfter=9),
              "title": ParagraphStyle("title", fontName="FSPBold", fontSize=24, leading=30, spaceAfter=14, textColor=colors.HexColor("#310F53")),
              "heading": ParagraphStyle("heading", fontName="FSPBold", fontSize=13, leading=19, spaceBefore=18, spaceAfter=9)}
    story = []
    def p(text, style="body"):
        story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style]))
    p("ФСП / КАРЬЕРА")
    p(profile.name, "title")
    p(profile.headline)
    p(profile.category)
    if profile.is_demo:
        p("ДЕМОНСТРАЦИОННЫЙ ПРОФИЛЬ. Данные и результаты синтетические.")
    story.append(HRFlowable(width="100%", color=colors.HexColor("#FF0053"), thickness=2))
    p("О специалисте", "heading")
    p(profile.about or "Описание пока не заполнено.")
    work = {"remote": "Удалённо", "office": "В офисе", "hybrid": "Гибрид"}
    p(f"Город: {profile.city or 'не указан'}. Опыт: {profile.experience_years} лет. {work.get(profile.work_format, profile.work_format)}.")
    p(f"Зарплатные ожидания: {profile.salary_min:,} - {profile.salary_max:,} RUB" if profile.salary_max else "Зарплатные ожидания не указаны.")
    p("Заявленный стек: " + (", ".join(profile.stack) or "не указан"))
    p("Подтверждение навыков", "heading")
    for skill, e in profile.skills.items():
        stamp = datetime.fromtimestamp(e["checked_at"], UTC).strftime("%d.%m.%Y")
        p(f"{skill}: {e['score']:.0f}% ответов, семейств: {e['families']}, проверка {stamp}.")
    if not profile.skills:
        p("Пока нет завершённой успешной проверки.")
    p("Это результат ограниченного учебного банка заданий, а не вероятность успешного найма.")
    p("Достижения ФСП", "heading")
    for a in profile.achievements:
        p(f"{a['title']} ({a.get('year', '')}). Источник: {a['source']}.")
    if not profile.achievements:
        p("Подтверждённых достижений нет. Это не ограничивает поиск работы.")
    p("Контакты", "heading")
    if profile.contacts:
        for v in profile.contacts.model_dump().values():
            if v:
                p(v)
    else:
        p("Закрыты. Доступ появится после принятия вашего приглашения или самостоятельного отклика кандидата.")
    story.append(Spacer(1, 18))
    p("Профиль сформирован " + datetime.now(UTC).strftime("%d.%m.%Y %H:%M UTC") + ". Актуальность разрешения проверяется при каждом скачивании.")
    def footer(canvas, doc):
        canvas.setFont("FSP", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(45, 27, "ФСП / Карьера · профиль навыков")
        canvas.drawRightString(A4[0] - 45, 27, str(doc.page))
    SimpleDocTemplate(output, pagesize=A4, leftMargin=45, rightMargin=45, topMargin=42,
                      bottomMargin=48, title="ФСП / Карьера — профиль", author="ФСП / Карьера").build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
