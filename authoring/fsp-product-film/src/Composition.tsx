import React from 'react';
import {AbsoluteFill, Composition, Easing, Img, Sequence, staticFile, useCurrentFrame, interpolate} from 'remotion';
import {loadFont} from '@remotion/fonts';
import assets from './assets.json';

for (const weight of ['400','700','800']) {
  void loadFont({family:'ManropeRU',url:staticFile(`fonts/manrope-cyrillic-${weight}-normal.woff2`),weight});
  void loadFont({family:'Manrope',url:staticFile(`fonts/manrope-latin-${weight}-normal.woff2`),weight});
}
const C={ink:'#29143c',pink:'#ff0867',paper:'#f6f3f9',muted:'#8c769d',line:'#dfd5e8'};
const ease=Easing.bezier(.22,1,.36,1);
const clamp={extrapolateLeft:'clamp',extrapolateRight:'clamp'} as const;
const titles=['','01 / ПРОФИЛЬ','02 / ПРОВЕРКА','03 / ПОИСК','04 / ПРИОРИТЕТЫ','05 / ПОДБОРКА','06 / ПРЕДЛОЖЕНИЕ','07 / СОГЛАСИЕ',''];
export const durations=[210,270,270,270,300,210,270,300,210];
export const starts=durations.reduce<number[]>((a,_,i)=>[...a,i===0?0:a[i-1]+durations[i-1]-12],[]);
export const TOTAL=starts[starts.length-1]+durations[durations.length-1];

function Mark({size=42,light=false}:{size?:number;light?:boolean}) {
 return <div style={{display:'flex',alignItems:'center',gap:18}}>
  <svg width={size} height={size} viewBox="0 0 48 48"><path d="M2 19L15 14V43L2 48Z" fill={C.pink}/><path d="M17 13L30 8V35L17 40Z" fill="#a77cc1"/><path d="M32 7L45 2V28L32 33Z" fill={light?'#ede0f5':C.ink}/></svg>
  <span style={{fontSize:size*.78,fontWeight:800,letterSpacing:-1.4,color:light?'#fff':C.ink}}>ФСП <span style={{fontWeight:400,color:C.muted}}>/</span> Карьера</span>
 </div>
}
function Shell({index,dark=false,children}:{index:number;dark?:boolean;children:React.ReactNode}) {
 const f=useCurrentFrame();const d=durations[index];
 return <AbsoluteFill style={{background:dark?C.ink:C.paper,color:dark?'white':C.ink,fontFamily:'ManropeRU, Manrope, sans-serif',opacity:interpolate(f,[0,12,d-12,d],[0,1,1,0],clamp)}}>
  <div style={{position:'absolute',left:96,right:96,top:52,display:'flex',alignItems:'center',justifyContent:'space-between'}}>
   <Mark light={dark}/><span style={{fontSize:27,letterSpacing:3,fontWeight:700,color:dark?'#cbb5db':C.muted}}>{titles[index]}</span>
  </div>
  {children}
  <div style={{position:'absolute',left:96,right:96,bottom:34,display:'flex',alignItems:'center',justifyContent:'space-between',fontSize:23,color:dark?'#b69cc8':'#907d9e'}}>
   <span>Хакатонный прототип · демонстрационные данные</span><span>{String(index+1).padStart(2,'0')} / 09</span>
  </div>
  <div style={{position:'absolute',bottom:0,left:0,height:4,background:C.pink,width:interpolate(f,[0,d],[0,1920],clamp)}}/>
 </AbsoluteFill>
}
function Head({children,sub}:{children:React.ReactNode;sub?:string}) {
 const f=useCurrentFrame();
 return <div style={{position:'absolute',top:144,left:96,right:96,opacity:interpolate(f,[7,29],[0,1],clamp),translate:`0 ${interpolate(f,[7,36],[25,0],{...clamp,easing:ease})}px`}}>
  <h1 style={{fontSize:88,lineHeight:1.08,letterSpacing:-4,fontWeight:800,margin:0}}>{children}</h1>
  {sub&&<p style={{fontSize:34,color:C.muted,margin:'22px 0 0',lineHeight:1.35}}>{sub}</p>}
 </div>
}
type Asset=keyof typeof assets;
function Crop({src,box,crop,zoom=1.035,delay=18,duration=200,shadow=true}:{src:Asset;box:[number,number,number,number];crop?:[number,number,number,number];zoom?:number;delay?:number;duration?:number;shadow?:boolean}) {
 const f=useCurrentFrame();const [x,y,w,h]=box;const a=assets[src];const [cx,cy,cw,ch]=crop??[0,0,a.w/2,a.h/2];
 const scale=Math.max(w/(cw*2),h/(ch*2));
 return <div style={{position:'absolute',left:x,top:y,width:w,height:h,borderRadius:24,background:'#fff',overflow:'hidden',border:'1px solid #e4dbe9',boxShadow:shadow?'0 28px 65px #27133718':undefined,opacity:interpolate(f,[delay,delay+22],[0,1],clamp),translate:`0 ${interpolate(f,[delay,delay+35],[35,0],{...clamp,easing:ease})}px`}}>
  <div style={{position:'absolute',inset:0,scale:interpolate(f,[delay+24,duration],[1,zoom],{...clamp,easing:ease})}}>
   <Img src={staticFile(`screens/${src}`)} style={{position:'absolute',width:a.w*scale,maxWidth:'none',height:a.h*scale,left:w/2-(cx+cw/2)*2*scale,top:h/2-(cy+ch/2)*2*scale}}/>
  </div>
 </div>
}
function Phrase({top,children,delay=20,color=C.ink,size=55}:{top:number;children:React.ReactNode;delay?:number;color?:string;size?:number}) {
 const f=useCurrentFrame();return <div style={{position:'absolute',left:116,top,width:590,fontSize:size,lineHeight:1.25,letterSpacing:-1.5,fontWeight:700,color,opacity:interpolate(f,[delay,delay+20],[0,1],clamp),translate:`0 ${interpolate(f,[delay,delay+30],[18,0],{...clamp,easing:ease})}px`}}>{children}</div>
}
function Intro() {
 const f=useCurrentFrame();return <Shell index={0} dark>
  <div style={{position:'absolute',right:-260,top:160,width:920,height:920,border:'1px solid #6d467e',borderRadius:460,scale:interpolate(f,[0,210],[.85,1.04],clamp)}}/>
  <div style={{position:'absolute',left:96,top:230,width:1470}}>
   <div style={{fontSize:31,letterSpacing:5,color:'#c6acd7',marginBottom:32,opacity:interpolate(f,[12,35],[0,1],clamp)}}>ОБРАТНЫЙ НАЙМ В ИТ</div>
   {['Навыки говорят','за вас.'].map((line,i)=><div key={line} style={{fontSize:160,fontWeight:800,letterSpacing:-8,lineHeight:1.08,color:i? '#ff5b98':'white',opacity:interpolate(f,[20+i*15,45+i*15],[0,1],clamp),translate:`0 ${interpolate(f,[20+i*15,55+i*15],[45,0],{...clamp,easing:ease})}px`}}>{line}</div>)}
   <div style={{height:5,background:C.pink,marginTop:38,width:interpolate(f,[64,100],[0,920],{...clamp,easing:ease})}}/>
   <p style={{fontSize:46,color:'#d2bddf',marginTop:42,opacity:interpolate(f,[80,110],[0,1],clamp)}}>От подтверждения навыков — к предложению команды.</p>
  </div>
 </Shell>
}
function Profile() {return <Shell index={1}>
 <Head>Профиль с подтверждениями.</Head>
 <Crop src="profile.png" box={[716,304,1108,656]} crop={[252,206,1156,686]} zoom={1.045}/>
 <Phrase top={345}>Опыт.<br/>Технологии.<br/>Условия работы.</Phrase>
 <div style={{position:'absolute',left:116,top:643,width:490,height:3,background:C.pink}}/>
 <Phrase top={687} size={37} delay={70} color={C.muted}>Профиль можно заполнить<br/>без ФСП ID.</Phrase>
 </Shell>}
function Assessment() {
 const f=useCurrentFrame();return <Shell index={2}>
 <Head>Покажите, что умеете.</Head>
 <Phrase top={355}>Практические<br/>задания.</Phrase>
 <Phrase top={569} size={43} delay={58} color={C.muted}>Результаты и разбор<br/>сохраняются в истории.</Phrase>
 <div style={{opacity:interpolate(f,[157,180],[1,0],clamp)}}><Crop src="task.png" box={[735,300,1089,658]} crop={[258,275,818,525]} zoom={1.025}/></div>
 <div style={{opacity:interpolate(f,[157,180],[0,1],clamp)}}><Crop src="result.png" box={[735,300,1089,658]} crop={[257,170,820,630]} zoom={1.03} delay={0}/></div>
 <div style={{position:'absolute',left:756,top:912,fontSize:25,color:C.muted,background:'#fff',padding:'8px 18px',borderRadius:8,opacity:interpolate(f,[170,185],[0,1],clamp)}}>Пример результата из демонстрационной истории</div>
 </Shell>
}
function Search(){return <Shell index={3}>
 <Head sub="Специализация · грейд · формат · зарплата">Опишите, кого ищете.</Head>
 <Crop src="search.png" box={[96,351,1728,599]} crop={[251,152,1158,402]} zoom={1.025}/>
 </Shell>}
function Priorities(){const f=useCurrentFrame();return <Shell index={4}>
 <Head>Что важнее вашей команде?</Head>
 <Crop src="question.png" box={[96,315,1280,577]} zoom={1} duration={300}/>
 <div style={{position:'absolute',left:716,top:453,width:306,height:264,borderRadius:15,border:`4px solid ${C.pink}`,background:'#ff086705',opacity:interpolate(f,[118,140],[0,1],clamp)}}/>
 <div style={{position:'absolute',left:1440,top:386,width:365,display:'flex',flexDirection:'column',gap:32}}>
  <div style={{fontSize:79,lineHeight:1.05,fontWeight:800,letterSpacing:-4,color:C.pink}}>А или Б?</div>
  <div style={{fontSize:38,lineHeight:1.3,color:C.muted}}>Сравните<br/>сочетания<br/>навыков.</div>
  <div style={{fontSize:36,lineHeight:1.3,fontWeight:700,opacity:interpolate(f,[132,154],[0,1],clamp)}}>Ответ уточнит<br/>подборку.</div>
 </div>
 <div style={{position:'absolute',left:116,top:938,fontSize:30,color:C.muted}}>До трёх уточнений. Грейд кандидата остаётся прежним.</div>
 </Shell>}
function Match(){return <Shell index={5}>
 <Head>Подборка с объяснением.</Head>
 <Crop src="matched.png" box={[96,308,1100,646]} crop={[254,30,828,485]} zoom={1.035}/>
 <Crop src="matched.png" box={[1250,308,574,646]} crop={[1100,614,302,341]} zoom={1.03} delay={45}/>
 </Shell>}
function Offer(){const f=useCurrentFrame();return <Shell index={6}>
 <Head sub="Роль, задачи и зарплатная вилка — до знакомства.">Предложение с условиями.</Head>
 <Crop src="candidate-offer.png" box={[96,389,1728,572]} zoom={1}/>
 <div style={{position:'absolute',left:129,top:505,width:613,height:74,borderRadius:12,border:`3px solid ${C.pink}`,opacity:interpolate(f,[90,115],[0,1],clamp)}}/>
 </Shell>}
function Consent(){const f=useCurrentFrame();return <Shell index={7} dark>
 <Head>Контакты — по согласию.</Head>
 <div style={{position:'absolute',left:98,top:286,fontSize:34,fontWeight:700,color:'#d0b9e0'}}>{f<145?'Кандидат видит предложение и принимает решение.':'Работодатель получает контакты после принятия.'}</div>
 <div style={{opacity:interpolate(f,[132,153],[1,0],clamp)}}><Crop src="candidate-offer.png" box={[96,394,1728,572]} zoom={1} shadow={false}/></div>
 <div style={{opacity:interpolate(f,[132,153],[0,1],clamp)}}><Crop src="contact-card.png" box={[96,368,1728,607]} zoom={1.005} delay={0} shadow={false}/></div>
 <div style={{position:'absolute',left:1430,top:525,width:350,height:327,borderRadius:16,border:'4px solid #32a685',opacity:interpolate(f,[171,196],[0,1],clamp)}}/>
 </Shell>}
function Outro(){const f=useCurrentFrame();return <Shell index={8} dark>
 <div style={{position:'absolute',inset:'190px 96px 140px',display:'flex',flexDirection:'column',justifyContent:'center'}}>
  {['Подтвердить.','Найти.','Договориться.'].map((t,i)=><div key={t} style={{fontSize:138,lineHeight:1.14,fontWeight:800,letterSpacing:-7,color:i===2?'#ff5b98':'white',opacity:interpolate(f,[12+i*26,36+i*26],[0,1],clamp),translate:`${interpolate(f,[12+i*26,46+i*26],[40,0],{...clamp,easing:ease})}px 0`}}>{t}</div>)}
  <p style={{fontSize:36,marginTop:45,color:'#d0b9e0',opacity:interpolate(f,[112,135],[0,1],clamp)}}>ФСП / Карьера · проверяемые навыки и взаимный выбор</p>
 </div>
 </Shell>}
const SCENES=[Intro,Profile,Assessment,Search,Priorities,Match,Offer,Consent,Outro];
export const ProductFilm=()=> <AbsoluteFill style={{background:C.ink}}>{SCENES.map((Scene,i)=><Sequence key={i} name={['Титул','Профиль','Проверка','Потребность','Сравнение','Подборка','Предложение','Согласие','Финал'][i]} from={starts[i]} durationInFrames={durations[i]}><Scene/></Sequence>)}</AbsoluteFill>;
export const MyComposition=()=> <Composition id="FspProductFilm" component={ProductFilm} durationInFrames={TOTAL} fps={30} width={1920} height={1080}/>;
