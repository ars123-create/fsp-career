"""Repair duplicate photo-placeholder IDs in a derived PPTX, preserving all geometry.

The organiser template contains five photo placeholders with idx=4294967295 on
slide 9. Keep the original untouched; use the same deterministic repair in the
comparison copy and exported candidate. Only duplicate p:ph idx attributes change.
"""
from collections import Counter
from pathlib import Path
import json
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def normalize(source: Path, destination: Path):
    changes = []
    with ZipFile(source) as zin, ZipFile(destination, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                root = ET.fromstring(data)
                keys = Counter((p.get("type", "obj"), p.get("idx", "")) for p in root.findall(".//p:ph", NS))
                duplicates = {k for k, v in keys.items() if v > 1}
                if duplicates:
                    serial = 9000
                    def repair(match):
                        nonlocal serial
                        tag = match.group(0)
                        kind = re.search(rb'\btype="([^"]+)"', tag)
                        idx = re.search(rb'\bidx="([^"]+)"', tag)
                        key = ((kind.group(1).decode() if kind else "obj"), idx.group(1).decode() if idx else "")
                        if key not in duplicates:
                            return tag
                        while (key[0], str(serial)) in keys:
                            serial += 1
                        changes.append({"part": item.filename, "type": key[0], "before": key[1], "after": str(serial)})
                        new = re.sub(rb'\bidx="[^"]+"', f'idx="{serial}"'.encode(), tag)
                        serial += 1
                        return new
                    data = re.sub(rb'<p:ph\b[^>]*/>', repair, data)
            zout.writestr(item, data)
    return changes


if __name__ == "__main__":
    print(json.dumps(normalize(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False))
