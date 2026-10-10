"""Keep the organiser's original slide/layout/master parts in an authored deck.

Artifact Tool imports templates into its editable model. Its export may flatten
inherited placeholders; this packaging pass restores source parts for mandatory
slides and transfers only edited text bodies, by the preserved shape IDs.
The authoring and editable charts remain Artifact Tool output. Originals are
read-only. Requires lxml (document-authoring runtime, not application runtime).
"""
from copy import deepcopy
from pathlib import PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED
import posixpath
import re
import sys
from lxml import etree as E

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"p": P, "a": A}


def xml(data):
    return E.fromstring(data)


def encoded(root):
    return E.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def relpart(part):
    p = PurePosixPath(part)
    return str(p.parent / "_rels" / (p.name + ".rels"))


def target_part(part):
    assert part.startswith("ppt/"), part
    p = PurePosixPath(part)
    m = re.fullmatch(r"(slide|slideLayout|slideMaster|notesSlide|notesMaster|theme)([0-9]+)\.xml", p.name)
    name = f"{m[1]}{int(m[2])+100}.xml" if m else "template_" + p.name
    return str(p.parent / name)


def merge(source, candidate, output):
    with ZipFile(source) as template, ZipFile(candidate) as authored:
        parts = {n: authored.read(n) for n in authored.namelist()}
        copied = set()
        def copy_part(part):
            if part in copied:
                return
            copied.add(part)
            parts[target_part(part)] = template.read(part)
            rp = relpart(part)
            if rp not in template.namelist():
                return
            rels = xml(template.read(rp))
            for rel in rels:
                if rel.get("TargetMode") == "External":
                    continue
                target = rel.get("Target")
                resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
                copy_part(resolved)
                rel.set("Target", posixpath.relpath(target_part(resolved), posixpath.dirname(target_part(part))))
            parts[relpart(target_part(part))] = encoded(rels)
        pres = xml(parts["ppt/presentation.xml"])
        rels = xml(parts["ppt/_rels/presentation.xml.rels"])
        relmap = {r.get("Id"): r for r in rels}
        for target_index, source_index in enumerate(range(7, 12), 1):
            source_part = f"ppt/slides/slide{source_index}.xml"
            copy_part(source_part)
            original = xml(parts[target_part(source_part)])
            changed = xml(parts[f"ppt/slides/slide{target_index}.xml"])
            by_id = {s.find("./p:nvSpPr/p:cNvPr", NS).get("id"): s for s in changed.findall(".//p:sp", NS)}
            for shape in original.findall(".//p:sp", NS):
                identity = shape.find("./p:nvSpPr/p:cNvPr", NS).get("id")
                edited = by_id.get(identity)
                if edited is None:
                    continue
                old_text = "".join(shape.xpath(".//a:t/text()", namespaces=NS))
                new_text = "".join(edited.xpath(".//a:t/text()", namespaces=NS))
                if old_text != new_text:
                    old_body, new_body = shape.find("p:txBody", NS), edited.find("p:txBody", NS)
                    if new_body is not None:
                        if old_body is not None:
                            shape.replace(old_body, deepcopy(new_body))
                        else:
                            shape.append(deepcopy(new_body))
            parts[target_part(source_part)] = encoded(original)
            sld = pres.findall("./p:sldIdLst/p:sldId", NS)[target_index - 1]
            relmap[sld.get("{" + R + "}id")].set("Target", target_part(source_part)[4:])
        # A master referenced by a restored layout must also be registered in
        # presentation.xml; otherwise readers can fall back to unrelated defaults.
        masters = pres.find("p:sldMasterIdLst", NS)
        used_ids = {int(x.get("id")) for x in masters}
        next_id = max(used_ids) + 1
        for index, part in enumerate(sorted(copied)):
            if re.fullmatch(r"ppt/slideMasters/slideMaster[0-9]+\.xml", part):
                rel_id = f"rIdSourceMaster{index}"
                E.SubElement(rels, "{"+REL+"}Relationship", Id=rel_id, Type=R+"/slideMaster", Target=target_part(part)[4:])
                item = E.SubElement(masters, "{"+P+"}sldMasterId", id=str(next_id))
                item.set("{"+R+"}id", rel_id)
                next_id += 1
        parts["ppt/presentation.xml"] = encoded(pres)
        parts["ppt/_rels/presentation.xml.rels"] = encoded(rels)
        types = xml(parts["[Content_Types].xml"])
        existing_defaults = {x.get("Extension") for x in types if x.tag.endswith("Default")}
        for node in xml(template.read("[Content_Types].xml")):
            if node.tag.endswith("Override") and node.get("PartName").lstrip("/") in copied:
                node.set("PartName", "/" + target_part(node.get("PartName").lstrip("/")))
                types.append(node)
            elif node.tag.endswith("Default") and node.get("Extension") not in existing_defaults:
                types.append(node)
                existing_defaults.add(node.get("Extension"))
        parts["[Content_Types].xml"] = encoded(types)
        with ZipFile(output, "w", ZIP_DEFLATED) as z:
            for name, data in parts.items():
                z.writestr(name, data)
    print(f"Preserved five source slides and {len(copied)} referenced parts; transferred only edited text bodies.")


if __name__ == "__main__":
    merge(*sys.argv[1:4])
