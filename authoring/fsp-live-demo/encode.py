"""Trim the captured browser streams to the logged scenes and join to silent MP4."""
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

src=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
report=json.loads((src/'record.json').read_text())
assert report['status']=='PASS', report.get('failure')
clips=src/'clips'
clips.mkdir(exist_ok=True)
out.parent.mkdir(parents=True,exist_ok=True)

def encode(s):
    target=clips/f'{s["chapter"]:02}.mp4'
    cmd=['/opt/homebrew/bin/ffmpeg','-y','-v','error','-ss',str(s['start']),'-i',str(src/(s['role']+'.webm')),'-t',str(s['end']-s['start']),'-an','-vf','fps=30','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    subprocess.run(cmd,check=True)
    return target
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    paths=list(pool.map(encode,report['segments']))
concat=clips/'concat.txt'
concat.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in paths))
subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-map','0:v:0','-c','copy','-an','-movflags','+faststart',str(out)],check=True)
print(json.dumps({'video':str(out),'chapters':len(paths),'bytes':out.stat().st_size},ensure_ascii=False))
