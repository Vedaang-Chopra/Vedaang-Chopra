window.RSVis = window.RSVis || {mods:{}};
(function(){
var ID='shastra', NS='http://www.w3.org/2000/svg';
var CSS=
'.rs-vis[data-vis="shastra"]{font-family:var(--font-sans);color:var(--text-color);font-size:15px;line-height:1.5}'+
'.rs-vis[data-vis="shastra"] *{box-sizing:border-box}'+
'.rs-vis[data-vis="shastra"] button{font:inherit;color:var(--text-color);cursor:pointer}'+
'.rs-vis[data-vis="shastra"] button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}'+
'.rs-vis[data-vis="shastra"] .sh-tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}'+
'.rs-vis[data-vis="shastra"] .sh-tab{background:var(--btn-bg);border:1px solid var(--border-color);border-radius:6px;padding:7px 12px}'+
'.rs-vis[data-vis="shastra"] .sh-tab[aria-selected="true"]{border-color:var(--text-color);box-shadow:inset 0 -3px 0 var(--text-color);font-weight:600}'+
'.rs-vis[data-vis="shastra"] .sh-note{color:var(--mut);margin:0 0 10px;font-size:14px}'+
'.rs-vis[data-vis="shastra"] .sh-steps{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}'+
'.rs-vis[data-vis="shastra"] .sh-step{background:var(--surface);border:1px solid var(--border-color);border-radius:16px;padding:4px 12px;font-size:14px}'+
'.rs-vis[data-vis="shastra"] .sh-step[aria-current="step"]{background:var(--btn-bg);border-color:var(--text-color);font-weight:600}'+
'.rs-vis[data-vis="shastra"] .sh-stage{background:var(--surface);border:1px solid var(--border-color);border-radius:8px;padding:12px;min-height:150px}'+
'.rs-vis[data-vis="shastra"] .sh-stage h4{margin:0 0 6px;font-size:15px}'+
'.rs-vis[data-vis="shastra"] .sh-mono{font-family:var(--font-mono);font-size:13px;overflow-wrap:anywhere}'+
'.rs-vis[data-vis="shastra"] .sh-plain{margin:8px 0 0;padding:8px;border-left:3px solid var(--ok)}'+
'.rs-vis[data-vis="shastra"] ol,.rs-vis[data-vis="shastra"] ul{margin:4px 0;padding-left:20px}'+
'.rs-vis[data-vis="shastra"] .sh-blk{border:1px solid var(--border-color);border-radius:6px;padding:8px;margin:6px 0;background:var(--bg-color)}'+
'.rs-vis[data-vis="shastra"] .sh-cap{display:inline-block;border:1px solid var(--border-color);border-radius:10px;padding:0 8px;margin:2px 4px 2px 0;font-family:var(--font-mono);font-size:12.5px}'+
'.rs-vis[data-vis="shastra"] .sh-nav{display:flex;gap:8px;margin-top:10px}'+
'.rs-vis[data-vis="shastra"] .sh-nav button,.rs-vis[data-vis="shastra"] .sh-row{background:var(--btn-bg);border:1px solid var(--border-color);border-radius:6px;padding:6px 14px}'+
'.rs-vis[data-vis="shastra"] .sh-nav button[disabled]{opacity:.5;cursor:default}'+
'.rs-vis[data-vis="shastra"] .sh-gwrap{overflow-x:auto}'+
'.rs-vis[data-vis="shastra"] .sh-gwrap svg{display:block;width:100%;min-width:420px;height:auto}'+
'.rs-vis[data-vis="shastra"] svg text{fill:var(--text-color);font-family:var(--font-sans)}'+
'.rs-vis[data-vis="shastra"] .sh-ans{font-size:20px;font-family:var(--font-mono);font-weight:600}'+
'.rs-vis[data-vis="shastra"] .sh-ladder{display:flex;flex-direction:column;gap:6px}'+
'.rs-vis[data-vis="shastra"] .sh-row{display:flex;align-items:center;gap:10px;text-align:left;width:100%;padding:8px 10px;min-height:44px}'+
'.rs-vis[data-vis="shastra"] .sh-row[aria-pressed="true"]{border-color:var(--text-color);box-shadow:inset 0 0 0 1px var(--text-color)}'+
'.rs-vis[data-vis="shastra"] .sh-shape{flex:none;display:block}'+
'.rs-vis[data-vis="shastra"] .sh-name{flex:1;min-width:0}'+
'.rs-vis[data-vis="shastra"] .sh-badge{font-weight:600;font-size:14px}'+
'.rs-vis[data-vis="shastra"] .sh-cnt{display:block;color:var(--mut);font-size:13.5px}'+
'@media (min-width:640px){.rs-vis[data-vis="shastra"] .sh-cnt{display:inline;margin-left:8px}}';

function el(t,c,x){var e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e}
function sv(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);return e}
function stxt(x,y,s,a){var e=sv('text',a||{});e.setAttribute('x',x);e.setAttribute('y',y);e.textContent=s;return e}

var COL={Built:'var(--ok)',Partial:'var(--warn)',Proposed:'var(--text-color)',Tried:'var(--bad)'};
var LAB={Built:'Built',Partial:'Partial',Proposed:'Proposed',Tried:'Tried and did not work'};

function shape(st,w,h,uid){
  var s=sv('svg',{width:w,height:h,viewBox:'0 0 '+w+' '+h,'class':'sh-shape','aria-hidden':'true',focusable:'false'});
  var a={x:1.5,y:1.5,width:w-3,height:h-3,rx:4,stroke:COL[st],'stroke-width':2};
  if(st==='Built'){a.fill=COL.Built;a['fill-opacity']=.35}
  else if(st==='Partial'){a.fill=COL.Partial;a['fill-opacity']=.3;var r0=sv('rect',a);s.appendChild(r0);
    s.appendChild(sv('rect',{x:1.5,y:1.5,width:(w-3)/2,height:h-3,rx:4,fill:COL.Partial,'fill-opacity':.7}));
    s.appendChild(sv('rect',a));s.lastChild.setAttribute('fill','none');return s}
  else if(st==='Proposed'){a.fill='none';a['stroke-dasharray']='5 4'}
  else{var d=sv('defs',{}),p=sv('pattern',{id:uid,width:7,height:7,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'});
    p.appendChild(sv('rect',{width:7,height:7,fill:'none'}));p.appendChild(sv('line',{x1:0,y1:0,x2:0,y2:7,stroke:COL.Tried,'stroke-width':3}));
    d.appendChild(p);s.appendChild(d);a.fill='url(#'+uid+')'}
  s.appendChild(sv('rect',a));return s}

function mount(root,data,ctx){
  if(!document.getElementById('rs-vis-'+ID+'-css')){var st=el('style');st.id='rs-vis-'+ID+'-css';st.textContent=CSS;document.head.appendChild(st)}
  root.textContent='';
  var views=[['One trace becomes a graph',viewTrace],['Is the idea real yet?',viewLadder]];
  var tabs=el('div','sh-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Shastra views');
  var body=el('div'),btns=[],cur=0;
  function show(i,focus){cur=i;btns.forEach(function(b,j){b.setAttribute('aria-selected',j===i);b.tabIndex=j===i?0:-1});
    body.textContent='';body.setAttribute('role','tabpanel');body.setAttribute('aria-labelledby',btns[i].id);views[i][1](body,data,ctx);if(focus)btns[i].focus()}
  views.forEach(function(v,i){var b=el('button','sh-tab',v[0]);b.type='button';b.id='sh-tab-'+i+'-'+Math.floor(Math.random()*1e6);
    b.setAttribute('role','tab');
    b.addEventListener('click',function(){show(i)});
    b.addEventListener('keydown',function(e){var n=null;
      if(e.key==='ArrowRight')n=(i+1)%views.length;else if(e.key==='ArrowLeft')n=(i+views.length-1)%views.length;
      else if(e.key==='Home')n=0;else if(e.key==='End')n=views.length-1;
      if(n!=null){e.preventDefault();show(n,true)}});
    btns.push(b);tabs.appendChild(b)});
  root.appendChild(tabs);root.appendChild(body);show(0);
}

function viewTrace(box,data){
  var ex=data.example,names=['Trace','Blocks','Graph','Replay'],i=0;
  box.appendChild(el('p','sh-note','One real trace, shown as an example. It is not a result.'));
  var steps=el('div','sh-steps');steps.setAttribute('role','group');steps.setAttribute('aria-label','Stages');
  var sb=[];names.forEach(function(n,k){var b=el('button','sh-step',(k+1)+'. '+n);b.type='button';b.addEventListener('click',function(){go(k)});sb.push(b);steps.appendChild(b)});
  var stage=el('div','sh-stage');stage.setAttribute('aria-live','polite');
  var nav=el('div','sh-nav'),prev=el('button',null,'Previous'),next=el('button',null,'Next');
  prev.type=next.type='button';
  prev.addEventListener('click',function(){go(i-1)});next.addEventListener('click',function(){go(i+1)});
  nav.appendChild(prev);nav.appendChild(next);
  box.appendChild(steps);box.appendChild(stage);box.appendChild(nav);
  function go(k){if(k<0||k>3)return;i=k;sb.forEach(function(b,j){if(j===k)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current')});
    prev.disabled=k===0;next.disabled=k===3;stage.textContent='';[tr,bl,gr,rp][k](stage)}
  function tr(s){
    s.appendChild(el('h4',null,'The task, as the agent received it'));
    s.appendChild(el('div','sh-mono',ex.task));
    s.appendChild(el('p','sh-plain','Read backwards, it says: '+ex.plain));
    s.appendChild(el('h4',null,'Events in the trace'));
    var ol=el('ol');ex.events.forEach(function(t){ol.appendChild(el('li',null,t))});s.appendChild(ol)}
  function bl(s){
    s.appendChild(el('h4',null,'The events are grouped into blocks, and each block is labelled'));
    ex.blocks.forEach(function(b){var d=el('div','sh-blk');d.appendChild(el('strong',null,b.name));d.appendChild(el('p',null,b.text));
      b.caps.forEach(function(c){d.appendChild(el('span','sh-cap',c))});s.appendChild(d)})}
  function gr(s){
    s.appendChild(el('h4',null,'Task-dependency graph'));
    var w=el('div','sh-gwrap');w.setAttribute('role','img');
    w.setAttribute('aria-label','Two steps, both llm. Step one points to step two with an observed order edge. Neither reads or writes recorded data.');
    var g=sv('svg',{viewBox:'0 0 560 160'});
    g.appendChild(sv('defs',{})).appendChild(sv('marker',{id:'sh-arr',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'})).appendChild(sv('path',{d:'M0 0L10 5L0 10z',fill:'currentColor'}));
    g.style.color='var(--text-color)';
    [20,330].forEach(function(x,k){var n=ex.nodes[k];
      g.appendChild(sv('rect',{x:x,y:20,width:210,height:120,rx:8,fill:'var(--bg-color)',stroke:'var(--text-color)','stroke-width':2}));
      g.appendChild(stxt(x+12,46,n.id+' ('+n.op+')',{'font-size':16,'font-weight':600}));
      g.appendChild(stxt(x+12,76,'reads: '+(n.reads.length?n.reads.join(', '):'nothing recorded'),{'font-size':14}));
      g.appendChild(stxt(x+12,102,'writes: '+(n.writes.length?n.writes.join(', '):'nothing recorded'),{'font-size':14}))});
    g.appendChild(sv('line',{x1:230,y1:80,x2:326,y2:80,stroke:'var(--text-color)','stroke-width':2.5,'marker-end':'url(#sh-arr)'}));
    g.appendChild(stxt(280,68,ex.edge.kind,{'font-size':14,'text-anchor':'middle'}));
    g.appendChild(stxt(280,104,ex.edge.trust,{'font-size':14,'text-anchor':'middle'}));
    w.appendChild(g);s.appendChild(w);
    s.appendChild(el('p',null,'Each box is a step; the arrow is an edge saying step two comes after step one. This edge is only about order, and it was observed in the trace.'))}
  function rp(s){
    s.appendChild(el('h4',null,'The graph is compiled to LangGraph code and run again'));
    s.appendChild(el('p',null,'Replay status: '+ex.replay.status+'. Replayed answer:'));
    s.appendChild(el('div','sh-ans',ex.replay.answer));
    s.appendChild(el('p',null,ex.replay.match?'It matches the answer in the original trace. This is one trace, not a measure of how often replay works.':''))}
  go(0);
}

function viewLadder(box,data){
  box.appendChild(el('p','sh-note','Each stage is marked by text and shape. Select a stage for what it means.'));
  var lad=el('div','sh-ladder');lad.setAttribute('role','group');lad.setAttribute('aria-label','Pipeline stages and their status');
  var det=el('div','sh-stage');det.setAttribute('aria-live','polite');det.style.minHeight='0';det.style.marginTop='10px';
  det.appendChild(el('p',null,'Select a stage to read what it means.'));
  var rows=[];
  data.stages.forEach(function(s,k){
    var b=el('button','sh-row');b.type='button';b.setAttribute('aria-pressed','false');
    b.appendChild(shape(s.status,40,26,'sh-hatch-'+k));
    var nm=el('span','sh-name');nm.appendChild(el('strong',null,s.name));nm.appendChild(el('br'));
    var bd=el('span','sh-badge',LAB[s.status]);nm.appendChild(bd);
    if(s.count)nm.appendChild(el('span','sh-cnt',s.count));
    b.appendChild(nm);
    b.addEventListener('click',function(){rows.forEach(function(r){r.setAttribute('aria-pressed','false')});b.setAttribute('aria-pressed','true');
      det.textContent='';det.appendChild(el('strong',null,s.name+' ('+LAB[s.status].toLowerCase()+')'));det.appendChild(el('p',null,s.mean));
      if(s.note)det.appendChild(el('p','sh-note',s.note))});
    rows.push(b);lad.appendChild(b)});
  box.appendChild(lad);box.appendChild(det);
  var key=el('p','sh-note','Key: solid fill is built, half fill is partial, dashed outline is proposed, hatched is tried and did not work. The owner’s longer-term goal, parameterized deterministic workflows as procedural memory, is not built.');
  key.style.marginTop='10px';box.appendChild(key);
}
RSVis.mods[ID]=mount;
})();
