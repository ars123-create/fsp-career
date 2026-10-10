"""Build PDF and a silent slide/screenshot montage from the finalized deck renders."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import shutil
import subprocess
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'output/submission-0.4.4-dev'
WORK = ROOT / 'tmp/release-0.4.4/deck/final-render'
DECK = RELEASE / 'FSP_Career_Pitch_0.4.4_reviewed.pptx'
PDF = RELEASE / 'FSP_Career_Pitch.pdf'
VIDEO = RELEASE / 'FSP_Career_Walkthrough.mp4'
images = sorted(WORK.glob('slide-*.png'))
assert len(images) == 15
assert DECK.is_file()
assert not PDF.exists() and not VIDEO.exists(), 'Preserve reviewed outputs; use a new revision path.'
with ZipFile(DECK) as z:
    assert z.testzip() is None
c = Canvas(str(PDF), pagesize=(960, 540))
c.setTitle('ФСП / Карьера — прототип 0.4.4-dev')
c.setAuthor('Команда не указана')
for image in images:
    c.drawImage(str(image), 0, 0, 960, 540)
    c.showPage()
c.save()
items = [(6,8),(7,12),(8,12),(9,10),(10,12),(11,12),(12,12),(13,12),(14,10),(15,12)]
concat = WORK / 'walkthrough.ffconcat'
concat.write_text('ffconcat version 1.0\n' + ''.join(
    f"file 'slide-{n:02}.png'\nduration {duration}\n" for n,duration in items
) + f"file 'slide-{items[-1][0]:02}.png'\n")
ffmpeg = shutil.which('ffmpeg')
assert ffmpeg, 'Optional video authoring requires FFmpeg.'
duration = sum(d for _,d in items)
subprocess.run([ffmpeg, '-hide_banner','-loglevel','warning','-n','-safe','1',
    '-i',str(concat),'-vf','scale=1920:1080,fps=24','-fps_mode','cfr','-t',str(duration),
    '-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
    '-movflags','+faststart','-metadata','title=FSP Career 0.4.4-dev',
    '-metadata','comment=Silent montage of slides and actual fictional-demo screenshots; not a continuous screen recording.',str(VIDEO)],check=True)
probe=json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error',
    '-show_streams','-show_format','-of','json',str(VIDEO)]))
streams=probe['streams']
assert len(streams)==1 and streams[0]['codec_name']=='h264'
assert (streams[0]['width'],streams[0]['height'])==(1920,1080)
assert abs(float(probe['format']['duration'])-duration)<0.1
result={'version':'0.4.4-dev','format':'silent montage of final slides and actual UI captures',
    'continuous_screen_recording':False,'audio':False,'duration_seconds':duration,
    'frames':[{'slide':n,'seconds':d,'sha256':hashlib.sha256((WORK/f'slide-{n:02}.png').read_bytes()).hexdigest()} for n,d in items],
    'pptx_sha256':hashlib.sha256(DECK.read_bytes()).hexdigest(),
    'video_sha256':hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
    'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'technical_probe':'PASS'}
(ROOT/'output/validation/release-0.4.4-media.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
