"""Repair text-only inheritance in required slides; keep all shape geometry/artwork."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
from lxml import etree as E

A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
NS = {'a': A, 'p': P}
source, target = map(Path, sys.argv[1:])
with ZipFile(source) as zin, ZipFile(target, 'x', ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        data = zin.read(info.filename)
        if info.filename in {'ppt/slides/slide108.xml', 'ppt/slides/slide109.xml', 'ppt/slides/slide111.xml'}:
            root = E.fromstring(data)
            for shape in root.findall('.//p:sp', NS):
                body = shape.find('p:txBody', NS)
                if body is None:
                    continue
                value = ''.join(body.xpath('.//a:t/text()', namespaces=NS))
                is_name = value.startswith(('Лобутев', 'Просветов'))
                marketing = value.startswith('Первый сегмент:')
                team = info.filename.endswith('slide109.xml')
                summary = value.startswith('Капитан:')
                if not (team or summary or marketing):
                    continue
                for paragraph in body.findall('a:p', NS):
                    props = paragraph.find('a:pPr', NS)
                    if props is None:
                        props = E.Element(f'{{{A}}}pPr'); paragraph.insert(0, props)
                    for child in list(props):
                        if E.QName(child).localname.startswith('bu'):
                            props.remove(child)
                    E.SubElement(props, f'{{{A}}}buNone')
                    if is_name:
                        for name in ('lnSpc', 'spcBef', 'spcAft'):
                            for old in props.findall(f'a:{name}', NS): props.remove(old)
                            setting = E.SubElement(props, f'{{{A}}}{name}')
                            E.SubElement(setting, f'{{{A}}}'+('spcPct' if name == 'lnSpc' else 'spcPts'), val='100000' if name == 'lnSpc' else '0')
                    if marketing:
                        # Imported paragraphs may omit rPr completely and inherit
                        # white text from the original purple slide theme.
                        for run in paragraph.findall('a:r', NS):
                            if run.find('a:rPr', NS) is None:
                                run.insert(0, E.Element(f'{{{A}}}rPr', sz='1400'))
                    for run_props in paragraph.findall('.//a:rPr', NS) + paragraph.findall('a:endParaRPr', NS):
                        if is_name: run_props.set('sz', '1425')
                        if marketing:
                            for old in run_props.findall('a:solidFill', NS):run_props.remove(old)
                            color=E.SubElement(run_props,f'{{{A}}}solidFill');E.SubElement(color,f'{{{A}}}srgbClr',val='241C2E')
                if marketing:
                    for props in body.findall('.//a:defRPr', NS):
                        for old in props.findall('a:solidFill', NS):props.remove(old)
                        fill=E.SubElement(props,f'{{{A}}}solidFill');E.SubElement(fill,f'{{{A}}}srgbClr',val='241C2E')
            data = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        zout.writestr(info, data)
print('Required slide text spacing, empty bullets and inherited contrast repaired; geometry unchanged.')
