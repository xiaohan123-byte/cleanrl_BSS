// Headless artifact rendering / layout verification with baoyu-design's dependency.
import { pathToFileURL } from 'node:url';
import fs from 'node:fs/promises';
import path from 'node:path';
const root=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const deps=process.argv[2];
const {chromium}=await import(pathToFileURL(path.join(deps,'node_modules/playwright/index.mjs')));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:4311/intro.html');
 await page.evaluate(()=>document.fonts.ready);
 await page.evaluate(()=>{document.querySelector('deck-stage').setAttribute('noscale','');document.querySelector('.note-toggle').style.display='none';});
 await fs.mkdir(path.join(root,'preview'),{recursive:true});
 const checks=[];
 for(let i=0;i<6;i++){
  await page.evaluate(i=>document.querySelector('deck-stage').goTo(i),i);
  const slide=page.locator('deck-stage > [data-deck-active]');
  await slide.screenshot({path:path.join(root,'preview',`slide-${i+1}.png`),animations:'disabled'});
  checks.push(await slide.evaluate(el=>{
   const root=el.getBoundingClientRect();
   return {title:el.dataset.label,width:root.width,height:root.height,notes:el.dataset.speakerNotes,
    overflow:[...el.querySelectorAll('*')].filter(x=>x.children.length===0&&x.textContent.trim()).flatMap(x=>{
     const r=x.getBoundingClientRect();return (r.left<root.left-1||r.top<root.top-1||r.right>root.right+1||r.bottom>root.bottom+1||x.scrollWidth>x.clientWidth+2)?[{text:x.textContent,rect:{x:r.x,y:r.y,w:r.width,h:r.height},scroll:x.scrollWidth,client:x.clientWidth}]:[];})};
  }));
 }
 await page.evaluate(()=>document.querySelector('deck-stage').removeAttribute('noscale'));
 await page.keyboard.press('Home');
 await page.keyboard.press('ArrowRight');
 const navigation=await page.locator('deck-stage > [data-deck-active]').getAttribute('data-label');
 await page.keyboard.press('n');
 const notesVisible=await page.locator('#notesPanel').evaluate(el=>getComputedStyle(el).display!=='none');
 await page.goto('http://127.0.0.1:4311/'+encodeURIComponent('引言汇报.html'));
 const standaloneSlides=await page.locator('deck-stage > section').count();
 const result={errors,checks,navigation,notesVisible,standaloneSlides};
 await fs.writeFile(path.join(root,'preview','html_check.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
}finally{await browser.close();}
