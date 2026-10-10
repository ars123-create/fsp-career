from pathlib import Path
import subprocess,json,re,math
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/playwright/review-0.5.1'
DEST=ROOT/'output/submission-0.5.1';DEST.mkdir(exist_ok=True)
chapters=[
 ('Вход в личный кабинет','Обычная форма входа: почта и пароль. Два отдельных кабинета.',22),
 ('Профиль и подтверждённые навыки','Опыт и предпочтения — в профиле. Грейд — по отдельной проверке; на экране демо-результат.',26),
 ('Подбор под потребности команды','Описание задачи → фильтры → уточнение приоритетов → сохранённая подборка.',27),
 ('Адресное приглашение','Роль, задачи и обязательная вилка зарплаты. Контакты пока скрыты.',25),
 ('Решение кандидата','Условия и компания видны заранее. Принятие открывает контакты этой компании.',13),
 ('Контакты и PDF-профиль','candidate@demo.fsp.test · employer@demo.fsp.test     Пароль обоих: 12345678',15),
]
font='/System/Library/Fonts/Supplemental/Arial.ttf';bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def duration(path):
 return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(path)],text=True))
log=[]
for i,(title,subtitle,target) in enumerate(chapters,1):
 src=OUT/f'scene-{i:02}.webm';total=duration(src)
 p=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-i',str(src),'-vf','freezedetect=n=-50dB:d=2','-an','-f','null','-'],capture_output=True,text=True,check=True)
 (OUT/f'freezes-{i:02}.txt').write_text(p.stderr)
 intervals=[];start=None
 for line in p.stderr.splitlines():
  m=re.search(r'freeze_start: ([\d.]+)',line)
  if m:start=float(m[1])
  m=re.search(r'freeze_end: ([\d.]+)',line)
  if m and start is not None:
   end=float(m[1]);
   if end-start>4: intervals.append((start+1.6,end-1.6))
   start=None
 if start is not None and total-start>4:intervals.append((start+1.6,total-1.6))
 keeps=[];cursor=0
 for a,b in intervals:
  keeps.append((cursor,a));cursor=b
 keeps.append((cursor,total))
 keeps=[(a,b) for a,b in keeps if b-a>.03]
 content=sum(b-a for a,b in keeps)
 startpad=3 if i in (5,6) else .6
 endpad=max(3,target-content-startpad)
 speed=min(1,(target-startpad-3)/content)
 actual=max(target,startpad+content*speed+3)
 filt=[]
 for j,(a,b) in enumerate(keeps):filt.append(f'[0:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{j}]')
 inputs=''.join(f'[v{j}]' for j in range(len(keeps)))
 filt.append(f'{inputs}concat=n={len(keeps)}:v=1:a=0,setpts={speed:.8f}*PTS,tpad=start_mode=clone:start_duration={startpad}:stop_mode=clone:stop_duration={endpad+5},trim=duration={actual},fps=25,scale=1440:810:flags=lanczos,pad=1440:906:0:0:color=0x29133d[base]')
 filt.append('[base][1:v]overlay=0:810:shortest=0,format=yuv420p[out]')
 img=Image.new('RGB',(1440,96),'#29133d');d=ImageDraw.Draw(img);d.rectangle((0,0,1440,3),fill='#ff0065')
 d.text((28,16),f'{i:02}  /  {title}',font=ImageFont.truetype(bold,24),fill='white')
 d.text((28,55),subtitle,font=ImageFont.truetype(font,18),fill='#e3d9eb')
 caption=OUT/f'title-{i:02}.png';img.save(caption)
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(src),'-loop','1','-i',str(caption),'-filter_complex',';'.join(filt),'-map','[out]','-an','-t',str(actual),'-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'chapter-{i:02}.mp4')]
 subprocess.run(cmd,check=True)
 log.append({'chapter':i,'title':title,'raw_seconds':total,'removed_idle_intervals':intervals,'retained_seconds':content,'output_seconds':duration(OUT/f'chapter-{i:02}.mp4'),'audio':False})
 print('Rendered chapter',i,log[-1]['output_seconds'],flush=True)
(OUT/'chapters.ffconcat').write_text('ffconcat version 1.0\n'+''.join(f"file chapter-{i:02}.mp4\n" for i in range(1,7)))
target=DEST/'FSP_Career_Overview_0.5.1.mp4'
subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-safe','1','-i',str(OUT/'chapters.ffconcat'),'-c','copy','-an','-movflags','+faststart',str(target)],check=True)
(OUT/'video-validation.json').write_text(json.dumps({'version':'0.5.1','source':'Actual recorded browser interactions; idle periods trimmed; captions; no audio','chapters':log,'duration':duration(target)},ensure_ascii=False,indent=2)+'\n')
print('Video ready',duration(target),flush=True)
