const {chromium}=require('playwright-core');
const fs=require('node:fs');const path=require('node:path');const {spawnSync}=require('node:child_process');
const overlay=require('./overlay.cjs');
const out=path.resolve(process.argv[2]);const base=process.argv[3]||'http://127.0.0.1:8034';
const dry=process.argv.includes('--rehearsal');
const chrome=process.env.DEMO_CHROME||'/Users/arsenij/Library/Caches/ms-playwright/chromium-1247/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';
const report={mode:dry?'rehearsal':'record',base,segments:[],errors:[],answers:[],checks:{}};
fs.mkdirSync(out,{recursive:true});
let browser;const sessions=[];let chapter=0;
const delay=ms=>new Promise(r=>setTimeout(r,dry?Math.min(100,ms):ms));
async function caption(s,title,sub='',label=''){
 await s.page.waitForFunction(()=>!!window.fspPresentation);
 await s.page.evaluate(v=>window.fspPresentation.caption(v.label,v.title,v.sub),{label:label||`${String(chapter).padStart(2,'0')} / ${s.role==='candidate'?'КАНДИДАТ':'РАБОТОДАТЕЛЬ'}`,title,sub});
}
async function scroll(s,loc){
 await loc.waitFor({state:'visible'});
 const b=await loc.boundingBox();
 if(b.y<120||b.y+b.height>740){await loc.evaluate(el=>el.scrollIntoView({behavior:'smooth',block:'center'}));await delay(900)}
}
async function click(s,loc){
 await scroll(s,loc);const b=await loc.boundingBox();
 await s.page.evaluate(p=>window.fspPresentation.point(p.x,p.y),{x:b.x+b.width/2,y:b.y+b.height/2});
 await delay(240);await loc.click();await delay(520);
}
async function fill(s,loc,value,{slow=true}={}){
 await click(s,loc);await loc.fill('');
 if(!dry&&slow)await loc.pressSequentially(value,{delay:23});else await loc.fill(value);
}
async function focus(s,loc){await scroll(s,loc);const b=await loc.boundingBox();await s.page.evaluate(b=>window.fspPresentation.focus(b),b);await delay(1800)}
async function shot(s,name){await s.page.screenshot({path:path.join(out,name+'.png')});}
async function scene(s,title,sub,fn,hold=2200){
 chapter++;const start=(Date.now()-s.zero)/1000;
 await caption(s,title,sub);await delay(850);await fn();await delay(hold);
 const end=(Date.now()-s.zero)/1000;
 report.segments.push({chapter,role:s.role,title,start,end});
 report.checks[`overflow_${chapter}`]=await s.page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);
 await shot(s,String(chapter).padStart(2,'0'));console.log(`${chapter}: ${title}`);
}
async function session(role){
 const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1,locale:'ru-RU',acceptDownloads:true,...(!dry?{recordVideo:{dir:out,size:{width:1440,height:900}}}:{})});
 await context.addInitScript(overlay);const zero=Date.now();const page=await context.newPage();page.setDefaultTimeout(12000);
 page.on('pageerror',e=>report.errors.push({role,message:e.message}));
 const s={role,context,page,zero};sessions.push(s);return s;
}
async function login(s){
 await s.page.goto(base+'/login');
 await s.page.getByRole('textbox',{name:'Email',exact:true}).fill(s.role+'@demo.fsp.test');
 await s.page.getByRole('textbox',{name:'Пароль',exact:true}).fill('12345678');
 await s.page.getByRole('button',{name:'Войти',exact:true}).click();
 await s.page.waitForURL(s.role==='candidate'?'**/dashboard':'**/search');
}
(async()=>{
 browser=await chromium.launch({headless:true,executablePath:chrome});
 const c=await session('candidate');const e=await session('employer');
 // Prepare a fresh demonstrative invitation through normal UI. No DB patches.
 await login(e);await e.page.goto(base+'/invitations');
 const withdrawal=e.page.getByRole('button',{name:'Отозвать приглашение',exact:true});
 await withdrawal.waitFor();await withdrawal.click();
 await e.page.getByText('Отозвано',{exact:true}).waitFor();
 await c.page.goto(base+'/');await c.page.evaluate(()=>document.fonts.ready);
 await scene(c,'ФСП / Карьера','Проверяемые навыки → подходящая команда → знакомство по согласию',async()=>{},3200);
 await scene(c,'Входим в кабинет кандидата','Обычный вход по почте. Данные в обзоре демонстрационные.',async()=>{
  await click(c,c.page.getByRole('link',{name:'Войти',exact:true}));
  await fill(c,c.page.getByRole('textbox',{name:'Email',exact:true}),'candidate@demo.fsp.test');
  await fill(c,c.page.getByRole('textbox',{name:'Пароль',exact:true}),'12345678',{slow:false});
  await click(c,c.page.getByRole('button',{name:'Войти',exact:true}));await c.page.waitForURL('**/dashboard');await delay(1600);
 });
 await scene(c,'Профиль объединяет опыт и подтверждения','Категория, стек, условия работы и история достижений.',async()=>{
  await click(c,c.page.getByRole('link',{name:'Мой профиль',exact:true}));await c.page.getByRole('heading',{name:'Опыт и предпочтения'}).waitFor();
  await delay(1800);await focus(c,c.page.getByRole('heading',{name:'История ФСП',exact:true}));
  await c.page.evaluate(()=>window.scrollBy({top:390,behavior:'smooth'}));await delay(2000);
 });
 await c.page.getByRole('link',{name:'Проверка навыков',exact:true}).click();
 let attempt;
 await scene(c,'Проверяем навыки на практических задачах','Короткая проверка: три задания. Подтверждённый грейд не меняется.',async()=>{
  const response=c.page.waitForResponse(r=>r.url().endsWith('/api/assessments')&&r.request().method()==='POST');
  await click(c,c.page.getByRole('button',{name:'Короткая проверка',exact:true}));attempt=await (await response).json();
  if(attempt.questions.length!==3)throw new Error('Expected three practical tasks');
  await focus(c,c.page.locator('.question-panel h2').first());
 });
 for(let i=0;i<attempt.questions.length;i++){
  const q=attempt.questions[i];
  const solved=spawnSync('python3',[path.join(__dirname,'solve-visible.py')],{input:JSON.stringify(q),encoding:'utf8'});
  if(solved.status!==0)throw new Error(solved.stderr);const answers=JSON.parse(solved.stdout);
  report.answers.push({family:q.family,values:answers});
  await scene(c,`${q.skill}: ${q.title}`,'Вводим ответ и сохраняем его на сервере.',async()=>{
   await scroll(c,c.page.locator('.question-panel h2').first());await delay(1800);
   const fields=c.page.locator('.practical-answer input');
   for(let j=0;j<q.fields.length;j++)await fill(c,fields.nth(j),answers[q.fields[j].name]);
   await click(c,c.page.getByRole('button',{name:'Сохранить ответ',exact:true}));
   await c.page.getByText('Ответ сохранён. Его можно изменить до завершения.',{exact:true}).waitFor();
   await delay(650);await caption(c,i===2?'Отправляем ответы на проверку':'Ответ сохранён',i===2?'Сервер рассчитывает результат и сохраняет попытку.':'Переходим к следующему заданию.');
   await click(c,c.page.getByRole('button',{name:i===2?'Завершить проверку':'Далее',exact:true}));
  },1200);
 }
 await scene(c,'Результат и разбор сохраняются в истории','Это только что завершённая попытка, а не подставленный результат.',async()=>{
  await c.page.locator('.result-header').waitFor();await focus(c,c.page.locator('.result-header'));
  report.checks.actual_result=await c.page.locator('.result-header').innerText();
  await scroll(c,c.page.locator('.answer-review details').first());
  await click(c,c.page.locator('.answer-review summary').first());await delay(2200);
 });
 await e.page.goto(base+'/search');
 await scene(e,'Работодатель описывает потребность','Python backend, SQL, Middle, удалённо, 180–250 тысяч рублей.',async()=>{
  await fill(e,e.page.getByRole('textbox',{name:'Потребность команды'}),'Python backend, SQL, Middle, удалённо, 180–250 тыс.');
  await click(e,e.page.getByRole('button',{name:'Найти специалистов',exact:true}));
  await e.page.getByRole('button',{name:'Выбрать Б',exact:true}).waitFor();
 });
 await scene(e,'Уточняем, что важнее команде','Сравнение двух сочетаний навыков помогает уточнить подборку.',async()=>{
  await focus(e,e.page.getByRole('table',{name:'Два сочетания результатов проверок'}));await delay(1900);
  await click(e,e.page.getByRole('button',{name:'Выбрать Б',exact:true}));
  await e.page.getByText(/Учтено ответов: 1/).waitFor();
  const show=e.page.getByRole('button',{name:'Посмотреть подборку',exact:true});if(await show.count())await click(e,show);
 });
 await scene(e,'Видим кандидатов и основания подбора','Проверенные навыки, формат работы и пересечение зарплатных ожиданий.',async()=>{
  await scroll(e,e.page.locator('.candidate-row').first());await delay(1800);
  await focus(e,e.page.locator('.detail-section').filter({has:e.page.getByRole('heading',{name:'Почему в подборке',exact:true})}));
 });
 await scene(e,'Для другой роли ищем Junior в гибридном формате','Уточняем условия и открываем подходящий профиль.',async()=>{
  await fill(e,e.page.getByRole('textbox',{name:'Потребность команды'}),'Python backend, SQL, Junior, гибрид, 100–150 тыс.');
  await e.page.getByLabel('Грейд',{exact:true}).selectOption('Junior');
  await e.page.getByLabel('Формат работы',{exact:true}).selectOption('hybrid');
  await click(e,e.page.getByRole('button',{name:'Найти специалистов',exact:true}));
  const row=e.page.locator('.candidate-row').filter({hasText:'Москва'});await row.waitFor();
  await click(e,row.locator('.candidate-main'));await delay(1200);
 });
 await scene(e,'Отправляем конкретное предложение','Роль, задачи и зарплатная вилка указаны до знакомства.',async()=>{
  const row=e.page.locator('.candidate-row').filter({hasText:'Москва'});
  await click(e,row.getByRole('button',{name:'Пригласить',exact:true}));
  await fill(e,e.page.getByLabel('Роль в команде',{exact:true}),'Python-разработчик · команда платформы');
  await fill(e,e.page.getByLabel('Команда, задачи и условия',{exact:true}),'Разработка API на FastAPI и работа с SQL. Наставник, код-ревью, гибридный формат. Демонстрационное предложение.');
  await fill(e,e.page.getByLabel('Зарплата от, ₽ / месяц',{exact:true}),'110000',{slow:false});
  await fill(e,e.page.getByLabel('До, ₽ / месяц',{exact:true}),'150000',{slow:false});
  await delay(1800);await click(e,e.page.getByRole('button',{name:'Отправить приглашение',exact:true}));
  await e.page.getByRole('dialog').waitFor({state:'hidden'});
  await click(e,e.page.getByRole('link',{name:/^Приглашения/}));
  await e.page.getByRole('heading',{name:'Python-разработчик · команда платформы',exact:true}).waitFor();
 });
 await scene(e,'До ответа контакты скрыты','Работодатель видит обезличенный профиль и статус предложения.',async()=>{
  const offer=e.page.locator('.offer').filter({hasText:'Python-разработчик · команда платформы'});
  report.checks.contacts_before=await offer.getByText('Контакты защищены',{exact:true}).isVisible();await focus(e,offer.locator('.offer-contact'));
 });
 await c.page.goto(base+'/invitations');
 await scene(c,'Кандидат выбирает, с кем знакомиться','Проверяет компанию, задачи и условия — затем принимает предложение.',async()=>{
  const offer=c.page.locator('.offer').filter({hasText:'Python-разработчик · команда платформы'});
  await offer.waitFor();await click(c,offer.locator('summary'));await delay(2500);
  await click(c,offer.getByRole('button',{name:'Принять и открыть контакты',exact:true}));
  await offer.getByText('Знакомство разрешено',{exact:true}).waitFor();
 });
 await e.page.reload();
 await scene(e,'После согласия доступны контакты и PDF','Разрешение получила компания, чьё предложение принято.',async()=>{
  const offer=e.page.locator('.offer').filter({hasText:'Python-разработчик · команда платформы'});
  await offer.getByText('Контакты доступны',{exact:true}).waitFor();await focus(e,offer.locator('.offer-contact'));
  report.checks.contacts_after=await offer.getByText('candidate@demo.fsp.test',{exact:true}).isVisible();
  const event=e.page.waitForEvent('download');await click(e,offer.getByRole('button',{name:'Профиль PDF',exact:true}));
  const download=await event;await download.saveAs(path.join(out,'fsp-profile.pdf'));
  report.checks.pdf_download=fs.readFileSync(path.join(out,'fsp-profile.pdf')).subarray(0,5).toString()==='%PDF-';
 },3200);
 await scene(e,'ФСП / Карьера','Подтвердить навыки. Найти команду. Договориться.',async()=>{},3800);
 if(report.errors.length||Object.values(report.checks).some(v=>v===false))throw new Error('A recording check failed');
 report.status='PASS';
})().catch(async error=>{
 report.status='FAIL';report.failure=error.stack;console.error(error);
 for(const s of sessions)await shot(s,'failure-'+s.role).catch(()=>{});
 process.exitCode=1;
}).finally(async()=>{
 for(const s of sessions){const video=s.page.video();await s.context.close();if(video){await video.saveAs(path.join(out,s.role+'.webm'));}}
 if(browser)await browser.close();
 fs.writeFileSync(path.join(out,'record.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({status:report.status,scenes:report.segments.length,checks:report.checks,errors:report.errors}));
});
