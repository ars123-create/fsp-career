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
 const email=process.env.FLOW_EMAIL||'fullflow@example.org';
 await c.page.goto(base+'/register');
 await scene(c,'Новый кандидат: от регистрации до предложения','Полный путь на чистом аккаунте. Все данные вымышлены.',async()=>{
  await fill(c,c.page.getByLabel('Имя и фамилия'),'Мария Соколова · демо');
  await fill(c,c.page.getByLabel('Email',{exact:true}),email);
  await fill(c,c.page.getByLabel(/^Пароль/),'DemoFlow2026!',{slow:false});
  await c.page.getByRole('checkbox').check();
  await click(c,c.page.getByRole('button',{name:'Создать аккаунт',exact:true}));
  await c.page.getByRole('button',{name:'Открыть демо-письмо'}).waitFor();
 });
 await scene(c,'Подтверждаем адрес из письма','Для локального стенда используется демонстрационная почта.',async()=>{
  await click(c,c.page.getByRole('button',{name:'Открыть демо-письмо'}));
  await click(c,c.page.getByRole('link',{name:'Перейти по ссылке из письма'}));
  await click(c,c.page.getByRole('button',{name:'Подтвердить адрес',exact:true}));
  await delay(700);
 });
 await c.page.goto(base+'/login');await c.page.getByLabel('Email',{exact:true}).fill(email);await c.page.getByLabel(/^Пароль/).fill('DemoFlow2026!');await c.page.getByRole('button',{name:'Войти',exact:true}).click();await c.page.waitForURL('**/dashboard');
 await c.page.getByRole('link',{name:'Мой профиль',exact:true}).click();
 await scene(c,'Заполняем опыт и пожелания','Самоописание помогает знакомству; категорию ещё предстоит подтвердить.',async()=>{
  await fill(c,c.page.getByLabel('Заголовок профиля'),'Python-разработчик · новая категория');
  await fill(c,c.page.getByLabel('О себе и своём опыте'),'Создаю учебные API и SQL-отчёты. Хочу развиваться в команде с наставником.');
  await fill(c,c.page.getByLabel('Город',{exact:true}),'Казань');
  await fill(c,c.page.getByLabel('Отрасль',{exact:true}),'Информационные технологии');
  await c.page.getByLabel(/^Формат работы/).selectOption('remote');
  await fill(c,c.page.getByLabel('Технологии через запятую'),'Python, SQL, API');
  await fill(c,c.page.getByLabel('Зарплата от, ₽ / месяц',{exact:true}),'100000',{slow:false});
  await fill(c,c.page.getByLabel('До, ₽ / месяц',{exact:true}),'150000',{slow:false});
 });
 await scene(c,'Публикация — по отдельному согласию','Без ФСП ID доступен весь путь. Контакты пока закрыты.',async()=>{
  await c.page.getByRole('checkbox',{name:'Разрешаю обработку данных для работы платформы.'}).check();
  await c.page.getByRole('checkbox',{name:'Разрешаю показывать обезличенный профиль работодателям.'}).check();
  await c.page.getByRole('checkbox',{name:'Опубликовать профиль в поиске.'}).check();
  await click(c,c.page.getByRole('button',{name:'Сохранить профиль',exact:true}));await delay(800);
 });
 await c.page.getByRole('link',{name:'Проверка навыков',exact:true}).click();
 let attempt;
 await scene(c,'Выбираем специализацию и заявленный уровень','Основная проверка: 12 заданий, из них 6 практических.',async()=>{
  await c.page.getByLabel(/^Специализация/).selectOption('backend');await c.page.getByLabel(/^Грейд/).selectOption('Junior');
  const response=c.page.waitForResponse(r=>r.url().endsWith('/api/assessments')&&r.request().method()==='POST');
  await click(c,c.page.getByRole('button',{name:'Начать проверку',exact:true}));attempt=await(await response).json();
  if(attempt.questions.length!==12)throw Error('Expected twelve questions');
 });
 fs.writeFileSync(path.join(out,'public-questions.json'),JSON.stringify(attempt.questions,null,2));
 if(process.argv.includes('--inspect')){report.status='INSPECT';return;}
 // Main answer sequence is inserted below after public questions are inspected.

 for(let i=0;i<attempt.questions.length;i++){
  const q=attempt.questions[i];
  const solve=spawnSync('python3',[path.join(__dirname,'solve-main-visible.py')],{input:JSON.stringify(q),encoding:'utf8'});
  if(solve.status!==0)throw Error(solve.stderr);const a=JSON.parse(solve.stdout);report.answers.push({family:q.family,values:a});
  await scene(c,`${i+1} / 12 · ${q.skill}: ${q.title}`,q.answer_type==='fields'?'Практический случай: вычисляем результат и сохраняем оба значения.':'Параметры и порядок вариантов различаются между попытками.',async()=>{
   await scroll(c,c.page.locator('.question-panel h2').first());await delay(1100);
   if(q.answer_type==='fields'){
    for(let j=0;j<q.fields.length;j++)await fill(c,c.page.locator('.practical-answer input').nth(j),a[q.fields[j].name],{slow:false});
    await click(c,c.page.getByRole('button',{name:'Сохранить ответ',exact:true}));
   }else await click(c,c.page.locator('.answer-options button').nth(a.choice));
   await c.page.getByText('Ответ сохранён. Его можно изменить до завершения.',{exact:true}).waitFor();
   await click(c,c.page.getByRole('button',{name:i===11?'Завершить проверку':'Далее',exact:true}));
  },700);
 }
 await scene(c,'Основной тест завершён — категория присвоена','Решение принято сервером по ответам на все 12 заданий.',async()=>{
  await c.page.locator('.result-header').waitFor();report.checks.result=await c.page.locator('.result-header').innerText();
  if(!report.checks.result.includes('100%'))throw Error('Unexpected score: '+report.checks.result);
  await focus(c,c.page.locator('.result-header'));
 });
 await c.page.getByRole('link',{name:'Мой профиль',exact:true}).click();
 await scene(c,'В профиле появилась подтверждённая категория','Python backend · Junior. ФСП ID не требуется.',async()=>{
  await c.page.locator('.profile-side').getByRole('heading',{name:'Junior',exact:true}).waitFor();
  report.checks.grade_awarded=true;await focus(c,c.page.locator('.profile-side section').first());
 });
 await login(e);
 await scene(e,'Работодатель ищет подтверждённого Junior','Python, SQL, удалённо, 100–150 тысяч рублей.',async()=>{
  await fill(e,e.page.getByRole('textbox',{name:'Потребность команды'}),'Python backend, SQL, Junior, удалённо, 100–150 тыс.');
  await e.page.getByLabel(/^Грейд/).selectOption('Junior');await e.page.getByLabel(/^Формат работы/).selectOption('remote');
  await click(e,e.page.getByRole('button',{name:'Найти специалистов',exact:true}));
  const row=e.page.locator('.candidate-row').filter({hasText:'Python-разработчик · новая категория'});await row.waitFor();
  await click(e,row.locator('.candidate-main'));report.checks.new_candidate_found=true;
 });
 await scene(e,'Отправляем конкретное предложение','Роль, задачи и зарплатная вилка указаны до знакомства.',async()=>{
  const row=e.page.locator('.candidate-row').filter({hasText:'Python-разработчик · новая категория'});
  await click(e,row.getByRole('button',{name:'Пригласить',exact:true}));
  await fill(e,e.page.getByLabel('Роль в команде',{exact:true}),'Python-разработчик · команда платформы');
  await fill(e,e.page.getByLabel('Команда, задачи и условия',{exact:true}),'Разработка API на FastAPI и работа с SQL. Наставник, код-ревью, удалённый формат. Демонстрационное предложение.');
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
  report.checks.contacts_after=await offer.getByText(email,{exact:true}).isVisible();
  const event=e.page.waitForEvent('download');await click(e,offer.getByRole('button',{name:'Профиль PDF',exact:true}));
  const download=await event;await download.saveAs(path.join(out,'fsp-profile.pdf'));
  report.checks.pdf_download=fs.readFileSync(path.join(out,'fsp-profile.pdf')).subarray(0,5).toString()==='%PDF-';
 },3200);

 await c.page.goto(base+'/jobs');
 await scene(c,'Кандидат может откликнуться самостоятельно','Дополнительный путь: открытая вакансия с зарплатой и условиями.',async()=>{
  await click(c,c.page.getByRole('button',{name:'Откликнуться',exact:true}).first());
  await fill(c,c.page.getByLabel('Сопроводительное сообщение'),'Интересна разработка API и работа с SQL. Готова обсудить задачи и условия.');
  await c.page.getByRole('dialog').getByRole('checkbox').check();
  await click(c,c.page.getByRole('button',{name:'Отправить отклик',exact:true}));
  await c.page.getByRole('dialog').waitFor({state:'hidden'});
 });
 await c.page.goto(base+'/applications');
 await scene(c,'Отклик сохранён и доступен в кабинете','Контакт получает компания, на чью вакансию откликнулся кандидат.',async()=>{
  await c.page.getByRole('heading',{name:'Мои отклики',exact:true}).waitFor();
  const text=await c.page.locator('main').innerText();
  report.checks.application_submitted=text.includes('Интересна разработка API');
  if(!report.checks.application_submitted)throw Error('Application not visible');
 },3000);
 await scene(c,'Полный путь работает','Регистрация → тест → категория → подбор → предложение → согласие. Есть и самостоятельный отклик.',async()=>{},3300);
 if(report.errors.length||Object.values(report.checks).some(v=>v===false))throw Error('Recording checks failed');

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
