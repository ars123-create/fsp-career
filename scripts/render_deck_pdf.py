"""Create a fixed-layout PDF companion from reviewed final-PPTX renders."""
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas

root = Path(__file__).resolve().parents[1]
images = sorted((root / "tmp/deck/final-render").glob("slide-*.png"))
assert len(images) == 11, "Render the finalized PPTX before creating its PDF companion."
target = root / "output/submission/FSP_Career_Pitch.pdf"
c = Canvas(str(target), pagesize=(960, 540))
c.setTitle("ФСП / Карьера — презентация прототипа")
c.setAuthor("Команда не указана")
for image in images:
    c.drawImage(str(image), 0, 0, 960, 540)
    c.showPage()
c.save()
print(target)
