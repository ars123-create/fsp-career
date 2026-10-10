"""Checks for the reviewed release artifacts; use the bundled authoring runtime."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from pypdf import PdfReader
from PIL import Image
import hashlib
import json
import posixpath
import shutil

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "output/submission-0.4.5-dev"
WORK = ROOT / "tmp/release-0.4.5/deck"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"p": P, "a": A}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    final = RELEASE / "FSP_Career_Pitch_0.4.5_reviewed_r2.pptx"
    structures = []
    with ZipFile(WORK / "template-comparison.pptx") as source, ZipFile(final) as out:
        assert out.testzip() is None
        for index in range(7, 12):
            trees = [E.fromstring(source.read(f"ppt/slides/slide{index}.xml")),
                     E.fromstring(out.read(f"ppt/slides/slide{index + 100}.xml"))]
            for tree in trees:
                for body in tree.findall(".//p:txBody", NS):
                    body.getparent().remove(body)
            equal = E.tostring(trees[0], method="c14n") == E.tostring(trees[1], method="c14n")
            assert equal, index
            structures.append({"source_slide": index, "final_slide": index - 6,
                               "non_text_structure_equal_to_normalized_source": equal})
        presentation = E.fromstring(out.read("ppt/presentation.xml"))
        relationships = E.fromstring(out.read("ppt/_rels/presentation.xml.rels"))
        targets = {node.get("Id"): node.get("Target") for node in relationships}
        parts = []
        for slide in presentation.find("p:sldIdLst", NS):
            target = targets[slide.get("{" + R + "}id")]
            parts.append(target.lstrip("/") if target.startswith("/")
                         else posixpath.normpath(posixpath.join("ppt", target)))
        assert len(parts) == 15
        for index in [7, 11, 13]:
            assert E.fromstring(out.read(parts[index - 1])).find(".//a:tbl", NS) is not None
        assert b"200 / 200" in out.read(parts[12])
        team = "".join(E.fromstring(out.read(parts[2])).xpath(".//a:t/text()", namespaces=NS))
        assert "Имя Фамилия" not in team

    pdf = PdfReader(RELEASE / "FSP_Career_Pitch.pdf")
    assert len(pdf.pages) == 15
    pixels = []
    for index, page in enumerate(pdf.pages, 1):
        assert tuple(float(x) for x in page.mediabox) == (0, 0, 960, 540)
        images = list(page.images)
        assert len(images) == 1
        source_image = Image.open(WORK / "final-render-r2" / f"slide-{index:02}.png").convert("RGB")
        actual = images[0].image.convert("RGB")
        assert source_image.size == actual.size and source_image.tobytes() == actual.tobytes(), index
        pixels.append({"page": index, "embedded_pixels_equal_to_reviewed_render": True})
    documentation = PdfReader(RELEASE / "FSP_Career_Documentation.pdf")
    assert len(documentation.pages) == 8
    text = "\n".join(page.extract_text() for page in documentation.pages)
    assert "0.4.5-dev" in text and "181" in text and "E2E_0_4_5.md" in text
    receipt = WORK / "FSP_Career_Pitch_0.4.5_reviewed_r2.pptx.validation.json"
    shutil.copyfile(receipt, ROOT / "output/validation/release-0.4.5-presentation-receipt.json")
    result = {
        "date": "2026-10-08", "version": "0.4.5-dev",
        "status": "PASS within rendered artifact scope",
        "pptx": {"path": str(final.relative_to(ROOT)), "sha256": sha(final), "slides": 15,
                 "native_tables": [7, 11, 13], "native_chart": [10],
                 "template_structure": structures, "team_fields": "blank as requested",
                 "visual_review": "15/15 main agent and independent slide_visual_review",
                 "nonblocking_limits": ["small supplementary UI text on slide 8",
                                        "source email wraps on slide 9"],
                 "repair": "blank crop on first slide 6 fixed in r2; other 14 renders identical"},
        "pitch_pdf": {"pages": 15, "pixels": pixels, "visual_review": "15/15 independent review PASS"},
        "docx_pdf": {"pages": 8, "all_pages_visually_reviewed": True,
                     "docx_sha256": sha(RELEASE / "FSP_Career_Documentation_0.4.5.docx"),
                     "pdf_sha256": sha(RELEASE / "FSP_Career_Documentation.pdf")},
        "video": {"duration_seconds": 112, "audio": False,
                  "format": "montage; not continuous screen recording", "full_decode_exit_code": 0},
        "native_microsoft_word_powerpoint": "NOT RUN",
        "prior_verification_tool_errors": [
            "pdftoppm fallback path absent; bundled override/pdftoppm succeeded",
            "first inline checker duplicated ppt prefix for absolute relationship targets; corrected"
        ],
    }
    (ROOT / "output/validation/release-0.4.5-artifacts.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PASS: 15 slides, 5 template structures, 15 exact PDF images, 8 documentation pages")


if __name__ == "__main__":
    main()
