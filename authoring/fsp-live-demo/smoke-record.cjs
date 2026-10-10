const {chromium}=require('playwright-core');
const fs=require('fs');const path=require('path');
(async()=>{
 const out=path.resolve('../../output/playwright/live-demo-v3/smoke');fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'/Users/arsenij/Library/Caches/ms-playwright/chromium-1247/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'});
 const context=await browser.newContext({viewport:{width:1440,height:900},recordVideo:{dir:out,size:{width:1440,height:900}}});
 const page=await context.newPage();await page.setContent('<body style="background:#29143c;color:white;font:48px sans-serif;padding:100px">ФСП / Карьера<button>Проверка записи</button></body>');
 await page.waitForTimeout(1200);await page.getByRole('button').click();await page.screenshot({path:path.join(out,'smoke.png')});
 const video=page.video();await context.close();await video.saveAs(path.join(out,'smoke.webm'));await browser.close();console.log('Minimal recordVideo example PASS');
})().catch(e=>{console.error(e);process.exit(1)});
