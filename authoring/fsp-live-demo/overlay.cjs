// Original presentation layer; actual application content and API are unchanged.
module.exports = function installPresentation() {
  const mount = () => {
    if (document.querySelector('#fsp-demo-caption')) return;
    const style = document.createElement('style');
    style.textContent = `
      #fsp-demo-caption{position:fixed;left:50%;bottom:22px;transform:translate(-50%,12px);width:max-content;max-width:1020px;box-sizing:border-box;padding:14px 26px 16px;border-radius:14px;border:1px solid #ffffff30;border-left:4px solid #ff0867;background:#29143cf5;color:white;box-shadow:0 14px 45px #17062040;z-index:2147483646;pointer-events:none;opacity:0;transition:opacity .35s,transform .35s;font-family:Manrope,system-ui,sans-serif;text-align:center}
      #fsp-demo-caption.visible{opacity:1;transform:translate(-50%,0)}
      #fsp-demo-caption .chapter{font-size:11px;letter-spacing:2px;font-weight:800;color:#ffa8cc;margin-bottom:5px}
      #fsp-demo-caption .title{font-size:23px;font-weight:750;line-height:1.25}
      #fsp-demo-caption .sub{font-size:15px;color:#e3d4ec;line-height:1.4;margin-top:5px}
      #fsp-demo-point{position:fixed;width:15px;height:15px;border:2px solid white;background:#ff0867;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;z-index:2147483647;opacity:0;box-shadow:0 0 0 2px #ff086744}
      #fsp-demo-ring{position:fixed;width:38px;height:38px;border:2px solid #ff0867;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;z-index:2147483647;opacity:0}
      #fsp-demo-focus{position:fixed;border:3px solid #ff0867;border-radius:10px;pointer-events:none;z-index:2147483645;opacity:0;box-shadow:0 0 0 5px #ff08670c}
    `;
    document.head.append(style);
    const caption=document.createElement('div');caption.id='fsp-demo-caption';
    for(const cls of ['chapter','title','sub']){const el=document.createElement('div');el.className=cls;caption.append(el)}
    const point=document.createElement('div');point.id='fsp-demo-point';
    const ring=document.createElement('div');ring.id='fsp-demo-ring';
    const focus=document.createElement('div');focus.id='fsp-demo-focus';
    document.body.append(caption,point,ring,focus);
    window.fspPresentation={
      caption(chapter,title,sub=''){
        caption.querySelector('.chapter').textContent=chapter;
        caption.querySelector('.title').textContent=title;
        caption.querySelector('.sub').textContent=sub;
        caption.querySelector('.sub').style.display=sub?'block':'none';
        caption.classList.toggle('visible',!!title);
      },
      point(x,y){
        for(const el of [point,ring]){el.style.left=x+'px';el.style.top=y+'px';el.getAnimations().forEach(a=>a.cancel())}
        point.animate([{opacity:0},{opacity:1,offset:.15},{opacity:1,offset:.65},{opacity:0}],{duration:1000});
        ring.animate([{opacity:0,scale:.5},{opacity:.9,scale:.7,offset:.2},{opacity:0,scale:1.8}],{duration:1000});
      },
      focus(b){
        Object.assign(focus.style,{left:(b.x-6)+'px',top:(b.y-6)+'px',width:(b.width+12)+'px',height:(b.height+12)+'px'});
        focus.animate([{opacity:0},{opacity:1,offset:.1},{opacity:1,offset:.8},{opacity:0}],{duration:2600});
      }
    };
  };
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
};
