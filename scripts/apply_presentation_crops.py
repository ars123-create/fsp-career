"""Apply declared native PPTX crops when the authoring API drops crop fields.

The original image bytes are preserved. Coordinates come from CUA capture
metadata; this is slide layout, not alteration of screenshot content.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import sys
from lxml import etree as E

A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
NS = {'a': A, 'p': P}
source, target, capture_json = map(Path, sys.argv[1:4])
placements = json.loads(Path(sys.argv[4]).read_text())
captures = json.loads(capture_json.read_text())
with ZipFile(source) as zin, ZipFile(target, 'x', ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        data = zin.read(info.filename)
        for spec in placements:
            if info.filename != f"ppt/slides/slide{spec['slide']}.xml":
                continue
            root = E.fromstring(data)
            pictures = root.findall('.//p:spTree/p:pic', NS)
            assert len(pictures) == 1, (info.filename, len(pictures))
            pic = pictures[0]
            crop = captures[spec['capture']]['crop']
            fill = pic.find('p:blipFill', NS)
            previous = fill.find('a:srcRect', NS)
            if previous is not None:
                fill.remove(previous)
            rect = E.Element(f'{{{A}}}srcRect')
            for k, name in [('l','left'), ('t','top'), ('r','right'), ('b','bottom')]:
                rect.set(k, str(round(100000*crop[name])))
            fill.insert(1, rect)
            xfrm = pic.find('p:spPr/a:xfrm', NS)
            x,y,w,h=spec['frame']
            for node, values in [('off', {'x':x,'y':y}), ('ext', {'cx':w,'cy':h})]:
                element=xfrm.find(f'a:{node}', NS)
                for key,value in values.items(): element.set(key,str(round(9525*value)))
            data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        zout.writestr(info,data)
print(json.dumps({'native_crops':len(placements),'source_images':'unchanged','target':str(target)}))
