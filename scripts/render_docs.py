"""Build the standalone technical PDF from the reviewed Russian source."""
from pathlib import Path
from html import escape
import re
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, KeepTogether
ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('FSP',str(ROOT/'backend/app/assets/DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('FSP-Bold',str(ROOT/'backend/app/assets/DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('FSP',normal='FSP',bold='FSP-Bold',italic='FSP',boldItalic='FSP-Bold')
plum=colors.HexColor('#310F53');pink=colors.HexColor('#FF0053')
styles={
 'body':ParagraphStyle('Body',fontName='FSP',fontSize=9.7,leading=14.5,spaceAfter=9,textColor=colors.HexColor('#2A2530'),allowWidows=0,allowOrphans=0),
 'h1':ParagraphStyle('H1',fontName='FSP-Bold',fontSize=25,leading=32,spaceAfter=20,textColor=plum),
 'h2':ParagraphStyle('H2',fontName='FSP-Bold',fontSize=15,leading=20,spaceBefore=18,spaceAfter=10,textColor=plum,keepWithNext=True),
 'label':ParagraphStyle('Label',fontName='FSP-Bold',fontSize=10,leading=16,textColor=pink,spaceAfter=16),
}
def rich(s):
 s=escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s);s=re.sub(r'`([^`]+)`',r'<font color="#310F53">\1</font>',s);return s

def footer(c,d):
 c.setStrokeColor(colors.HexColor('#E6DEEC'));c.line(45,39,A4[0]-45,39);c.setFont('FSP',8);c.setFillColor(colors.HexColor('#736A7B'));c.drawString(45,25,'ФСП / Карьера · 0.1.0 · 7 октября 2026');c.drawRightString(A4[0]-45,25,str(d.page))
out=ROOT/'output/submission/FSP_Career_Documentation.pdf';out.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=44,bottomMargin=55,title='ФСП / Карьера — техническая документация',author='FSP Career project contributors')
story=[Paragraph('СПЕЦТРЕК ФСП · ХАКАТОН 2026',styles['label']),Paragraph('ФСП / Карьера',styles['h1']),Paragraph('Техническая документация и результаты проверок',styles['h2']),Paragraph('Работающий прототип обратного найма по подтверждённым навыкам. Данные стенда вымышлены. Версия 0.1.0.',styles['body']),Spacer(1,14)]
im=Image(str(ROOT/'output/playwright/search-final.png'));im.drawWidth=505;im.drawHeight=505*960/1440;story.append(im);story += [Spacer(1,20),Paragraph('Кандидат подтверждает навыки. Работодатель видит основания совпадения и предлагает условия. Контакты открываются после согласия конкретной компании.',styles['body']),Paragraph('Пилот на людях и официальный API ФСП пока не подключены. Документ отделяет реализованные механизмы от непроверенных свойств.',styles['body']),PageBreak()]
source=(ROOT/'docs/TECHNICAL_RU.md').read_text().splitlines();paragraph=[]
def flush():
 if paragraph:story.append(Paragraph(rich(' '.join(paragraph)),styles['body']));paragraph.clear()
for line in source[3:]:
 if line.startswith('## '):flush();story.append(Paragraph(rich(line[3:]),styles['h2']))
 elif not line.strip():flush()
 else:paragraph.append(line)
flush();story.append(PageBreak());story.append(Paragraph('Запуск и демонстрация',styles['h1']))
for text in [
 '1. Из корня проекта: python3 scripts/setup.py; затем docker compose up -d --build.',
 '2. Приложение: http://127.0.0.1:8019. На главной — входы кандидата и работодателя. Проверка API: /docs. Локальная почта: http://127.0.0.1:8026.',
 '3. Кандидат: профиль → проверка → история. Работодатель: запрос Python backend, SQL, Middle, удалённо, 180–250 тыс → фильтры → профиль → предложение.',
 '4. Путь согласия: кандидат принимает исходное демо-приглашение «Орбита Технологии». Эта компания получает контакт; другой работодатель — нет. Для новой регистрации письмо находится в Mailpit.',
 '5. Проверки: PYTHONPATH=backend .venv/bin/pytest backend/tests -q; npm --prefix frontend run build. Нагрузочные скрипты и frozen результаты приложены к исходникам.',
 '6. Стенд локальный. Публичный хост, внешний SMTP и загрузка на платформу не настроены автоматически. Сведения о команде оставлены пустыми по указанию участника.',
 'Точный состав и версии: README.md, docs/IMPLEMENTATION_STATUS.md, docs/IMPLEMENTATION_REUSE.md, THIRD_PARTY.md и lock-файлы. Полный HTTP-контракт: docs/openapi.json.',
 'Исходники критериев: materials/ФСП спец. трек ТЗ.pdf и materials/ЛЦТ_2026_Шаблон презентации ФСП.pptx. Изложение требований и источники по страницам — docs/REQUIREMENTS.md.',
]:story.append(Paragraph(rich(text),styles['body']))
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
