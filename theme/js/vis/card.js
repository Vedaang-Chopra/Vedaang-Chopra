/* Shared project-card renderer. A project module calls RSCard.mount(el, data, {tabs}).
   data.card = { problem:{strip:[{text,tone}], line}, flow:{nodes:[{label,sub,isNew}], loop, note},
                 early:{title, unit, bars:[{label,value,shown,kind}], max, caveat, notShown} }
   tabs = [{id, label, render(panel, data)}]; a tab renders once, the first time it is opened. */
window.RSVis = window.RSVis || {mods:{}};
(function(){
var NS='http://www.w3.org/2000/svg';
var CSS=
'.rc{font-family:var(--font-sans);color:var(--text-color);font-size:15px;line-height:1.5}'+
'.rc *{box-sizing:border-box}'+
'.rc-sec{border-top:1px solid var(--border-color);margin-top:16px;padding-top:12px}'+
'.rc-lab{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 6px;font-weight:600}'+
'.rc-line{margin:6px 0 0;color:var(--mut);font-size:14px}'+
'.rc svg{display:block;width:100%;height:auto}'+
'.rc svg text{fill:var(--text-color);font-family:var(--font-sans);font-size:12px}'+
'.rc svg .m{fill:var(--mut)}'+
'.rc-tabs{display:flex;flex-wrap:wrap;gap:18px;border-bottom:1px solid var(--border-color);margin:2px 0 12px}'+
'.rc-tab{background:none;border:0;border-bottom:2px solid transparent;padding:4px 0 6px;font:inherit;font-size:14px;color:var(--mut);cursor:pointer}'+
'.rc-tab[aria-selected="true"]{color:var(--text-color);border-bottom-color:var(--text-color);font-weight:600}'+
'.rc button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}'+
'.rc-panel[hidden]{display:none}'+
'.rc-bars{list-style:none;margin:0;padding:0;display:grid;gap:8px}'+
'.rc-bar{display:grid;grid-template-columns:110px 1fr auto;gap:10px;align-items:center;font-size:14px}'+
'.rc-track{height:14px;background:var(--surface);border:1px solid var(--border-color);border-radius:3px;overflow:hidden}'+
'.rc-fill{display:block;height:100%;background:var(--mut)}'+
'.rc-bar.is-ours .rc-fill{background:var(--ok)}'+
'.rc-flag{font-size:12px;color:var(--bad,#b3261e);text-transform:none;letter-spacing:0;font-weight:600;margin-left:6px}'+
'.rc-ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0;font-size:14px}'+
'.rc-seg{display:inline-flex;border:1px solid var(--border-color);border-radius:6px;overflow:hidden}'+
'.rc-seg button{font:inherit;font-size:13px;background:var(--btn-bg);color:var(--text-color);border:0;padding:5px 12px;cursor:pointer}'+
'.rc-seg button[aria-pressed="true"]{background:var(--text-color);color:var(--bg-color)}';

function el(tag,cls,text){var e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;}
function svg(tag,attrs,text){var e=document.createElementNS(NS,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);if(text!=null)e.textContent=text;return e;}
function section(root,label){var s=el('section','rc-sec');s.appendChild(el('h4','rc-lab',label));root.appendChild(s);return s;}
function injectCss(){if(document.getElementById('rc-css'))return;var s=el('style');s.id='rc-css';s.textContent=CSS;document.head.appendChild(s);}

function problem(root,p,badge){
  var s=section(root,'1 · Problem');if(badge)s.firstChild.appendChild(el('span','rc-flag',badge));var n=p.strip.length,w=700,gap=36,bw=(w-gap*(n-1))/n;
  var v=svg('svg',{viewBox:'0 0 '+w+' 80',role:'img','aria-label':p.strip.map(function(x){return x.text;}).join(', then ')});
  p.strip.forEach(function(it,i){
    var x=i*(bw+gap);
    var stroke=it.tone==='bad'?'var(--bad,#b3261e)':it.tone==='ok'?'var(--ok)':'var(--border-color)';
    v.appendChild(svg('rect',{x:x,y:8,width:bw,height:60,rx:6,fill:'var(--surface)',stroke:stroke,'stroke-width':it.tone?2:1}));
    wrap(v,it.text,x+bw/2,32,bw-12,16);
    if(i<n-1)v.appendChild(svg('text',{x:x+bw+gap/2,y:42,'text-anchor':'middle','font-size':18,class:'m'},'→'));
  });
  s.appendChild(v);s.appendChild(el('p','rc-line',p.line));
}
function wrap(parent,text,cx,y,maxw,lh){
  var words=String(text).split(' '),lines=[],cur='',cpl=Math.max(8,Math.floor(maxw/6.6));
  words.forEach(function(w){if((cur+' '+w).trim().length>cpl&&cur){lines.push(cur);cur=w;}else cur=(cur+' '+w).trim();});
  if(cur)lines.push(cur);
  var y0=y-((lines.length-1)*lh)/2+4;
  lines.forEach(function(l,i){parent.appendChild(svg('text',{x:cx,y:y0+i*lh,'text-anchor':'middle'},l));});
}
function flow(root,f){
  var s=section(root,'2 · What I built'),n=f.nodes.length,w=700,gap=18,bw=(w-gap*(n-1))/n;
  var v=svg('svg',{viewBox:'0 0 '+w+' '+(f.loop?136:84),role:'img','aria-label':f.nodes.map(function(x){return x.label;}).join(', then ')});
  f.nodes.forEach(function(it,i){
    var x=i*(bw+gap);
    v.appendChild(svg('rect',{x:x,y:6,width:bw,height:68,rx:6,fill:it.isNew?'var(--ok-bg,var(--surface))':'var(--surface)',stroke:it.isNew?'var(--ok)':'var(--border-color)','stroke-width':it.isNew?2:1}));
    wrap(v,it.label,x+bw/2,28,bw-10,15);
    if(it.sub)v.appendChild(svg('text',{x:x+bw/2,y:66,'text-anchor':'middle',class:'m','font-size':11},it.sub));
    if(i<n-1)v.appendChild(svg('path',{d:'M'+(x+bw)+' 40H'+(x+bw+gap-2),stroke:'var(--mut)',fill:'none'}));
  });
  if(f.loop){
    v.appendChild(svg('path',{d:'M'+(w-bw/2)+' 76V106H'+(bw/2)+'V76',stroke:'var(--ok)','stroke-width':2,fill:'none','stroke-dasharray':'5 4'}));
    v.appendChild(svg('text',{x:w/2,y:126,'text-anchor':'middle',class:'m'},f.loop));
  }
  s.appendChild(v);if(f.note)s.appendChild(el('p','rc-line',f.note));
}
function tabs(root,list,data){
  var s=section(root,'3 · See it'),bar=el('div','rc-tabs'),panels=[],tabEls=[];bar.setAttribute('role','tablist');
  var uid='rc'+Math.random().toString(36).slice(2,7);
  function show(i){tabEls.forEach(function(t,j){t.setAttribute('aria-selected',String(i===j));t.tabIndex=i===j?0:-1;panels[j].hidden=i!==j;});
    var p=panels[i];if(!p.getAttribute('data-done')){p.setAttribute('data-done','1');list[i].render(p,data);}}
  list.forEach(function(t,i){
    var b=el('button','rc-tab',t.label);b.type='button';b.setAttribute('role','tab');b.id=uid+'t'+i;b.setAttribute('aria-controls',uid+'p'+i);
    b.addEventListener('click',function(){show(i);});
    b.addEventListener('keydown',function(e){var d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;if(d){var k=(i+d+list.length)%list.length;show(k);tabEls[k].focus();e.preventDefault();}});
    bar.appendChild(b);tabEls.push(b);
    var p=el('div','rc-panel');p.id=uid+'p'+i;p.setAttribute('role','tabpanel');p.setAttribute('aria-labelledby',b.id);p.hidden=true;panels.push(p);
  });
  s.appendChild(bar);panels.forEach(function(p){s.appendChild(p);});show(0);
}
function early(root,e){
  var s=section(root,'4 · Early numbers');
  var flag=el('span','rc-flag','early, not final');s.firstChild.appendChild(flag);
  if(e.title)s.appendChild(el('p','',e.title));
  var bars=e.bars||[],ul=el('ul','rc-bars'),max=e.max||Math.max.apply(null,bars.map(function(b){return b.value;}));
  bars.forEach(function(b){
    var li=el('li','rc-bar'+(b.kind==='ours'?' is-ours':''));
    li.appendChild(el('span','',b.label));
    var tr=el('span','rc-track'),f=el('span','rc-fill');f.style.width=(b.value/max*100).toFixed(1)+'%';tr.appendChild(f);li.appendChild(tr);
    li.appendChild(el('span','',b.shown||String(b.value)));ul.appendChild(li);
  });
  if(bars.length)s.appendChild(ul);
  if(e.unit)s.appendChild(el('p','rc-line',e.unit));
  if(e.caveat)s.appendChild(el('p','rc-line',e.caveat));
  if(e.notShown)s.appendChild(el('p','rc-line','Not shown yet: '+e.notShown));
}
window.RSCard={
  el:el,svg:svg,
  /* segmented one-control helper for tabs: opts=[{id,label}], onPick(id) */
  seg:function(opts,initial,onPick){var d=el('div','rc-seg'),bs=[];opts.forEach(function(o){var b=el('button','',o.label);b.type='button';b.setAttribute('aria-pressed',String(o.id===initial));
    b.addEventListener('click',function(){bs.forEach(function(x){x.setAttribute('aria-pressed','false');});b.setAttribute('aria-pressed','true');onPick(o.id);});bs.push(b);d.appendChild(b);});return d;},
  mount:function(host,data,cfg){
    injectCss();var c=data.card,root=el('div','rc');host.appendChild(root);
    problem(root,c.problem,c.badge);flow(root,c.flow);tabs(root,cfg.tabs,data);early(root,c.early);
  }
};
})();
