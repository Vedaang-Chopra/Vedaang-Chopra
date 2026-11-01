window.RSVis = window.RSVis || {mods:{}};
(function(){
var ID='shastra';
var CSS=
'.sh-cap{font-size:14px;color:var(--mut);margin:0 0 8px}'+
'.sh-blk{border:1px solid var(--border-color);border-radius:6px;padding:8px 10px;margin:6px 0;background:var(--surface)}'+
'.sh-blk.on{border-color:var(--text-color);border-width:2px;padding:7px 9px}'+
'.sh-blk.off{opacity:.55}'+
'.sh-blk h5{margin:0 0 4px;font-size:14px}'+
'.sh-blk ol{margin:0;padding-left:20px;font-size:14px}'+
'.sh-chip{display:inline-block;border:1px solid var(--border-color);border-radius:10px;padding:0 8px;margin:4px 4px 0 0;font-family:var(--font-mono);font-size:12px}'+
'.sh-read{margin:8px 0 0;font-size:14px;min-height:2.6em}'+
'.sh-task{font-size:14px;margin:0 0 6px}.sh-task code{font-family:var(--font-mono);font-size:13px;overflow-wrap:anywhere}'+
'.sh-res{border:1px solid var(--border-color);border-radius:6px;padding:10px;background:var(--surface);font-size:14px;margin-top:8px}'+
'.sh-res b{font-family:var(--font-mono)}';

function blocksTab(panel,data){
  var C=RSCard,ex=data.example;
  panel.appendChild(C.el('p','sh-cap','One short run, cut into 3 blocks. Pick a block to see which events it holds and the labels it gets.'));
  panel.appendChild(C.el('p','sh-task','Task (typed backwards): '+ex.plain));
  var groups=[[0,1],[2],[3,4]],names=['Block 1','Block 2','Block 3'];
  var ctl=C.el('div','rc-ctl'),list=C.el('div'),read=C.el('p','sh-read');
  read.setAttribute('aria-live','polite');
  var cards=groups.map(function(g,i){
    var b=ex.blocks[i],c=C.el('div','sh-blk');c.appendChild(C.el('h5','',names[i]+': '+b.text));
    var ol=C.el('ol');ol.start=g[0]+1;g.forEach(function(k){ol.appendChild(C.el('li','',ex.events[k]));});c.appendChild(ol);
    var chips=C.el('div');b.caps.forEach(function(x){chips.appendChild(C.el('span','sh-chip',x));});c.appendChild(chips);
    list.appendChild(c);return c;});
  function pick(i){cards.forEach(function(c,j){c.className='sh-blk '+(i===j?'on':'off');});
    var n=ex.blocks[i].caps.length;
    read.textContent=names[i]+' holds '+groups[i].length+' of the 5 events and gets '+n+' labels. Labels are written by a language model and are not yet validated.';}
  ctl.appendChild(C.el('span','','Show:'));
  ctl.appendChild(C.seg(names.map(function(n,i){return {id:i,label:n};}),0,pick));
  panel.appendChild(ctl);panel.appendChild(list);panel.appendChild(read);pick(0);
}

function graphTab(panel,data){
  var C=RSCard,ex=data.example,n=ex.nodes;
  panel.appendChild(C.el('p','sh-cap','The observed workflow graph for the same run: two steps joined in the order they happened. Meaning of each step is not yet validated.'));
  var ctl=C.el('div','rc-ctl'),read=C.el('p','sh-read');read.setAttribute('aria-live','polite');
  var host=C.el('div');
  var plain=['Step one: a model call','Step two: a model call'];
  function draw(mode){
    host.textContent='';
    var v=C.svg('svg',{viewBox:'0 0 700 130',role:'img','aria-label':'Two steps joined by an arrow labelled order, observed'});
    var xs=[40,420];
    n.forEach(function(nd,i){
      v.appendChild(C.svg('rect',{x:xs[i],y:20,width:240,height:70,rx:8,fill:'var(--surface)',stroke:'var(--text-color)','stroke-width':1.5}));
      v.appendChild(C.svg('text',{x:xs[i]+120,y:50,'text-anchor':'middle','font-size':14},mode==='plain'?plain[i]:nd.id));
      v.appendChild(C.svg('text',{x:xs[i]+120,y:72,'text-anchor':'middle',class:'m'},mode==='plain'?'reads and writes: none recorded':'op: '+nd.op+' · reads: none · writes: none'));
    });
    v.appendChild(C.svg('path',{d:'M282 55H416',stroke:'var(--text-color)','stroke-width':2,fill:'none'}));
    v.appendChild(C.svg('path',{d:'M408 49L418 55L408 61',stroke:'var(--text-color)','stroke-width':2,fill:'none'}));
    v.appendChild(C.svg('text',{x:350,y:44,'text-anchor':'middle',class:'m'},mode==='plain'?'then':'edge: '+ex.edge.kind));
    v.appendChild(C.svg('text',{x:350,y:118,'text-anchor':'middle',class:'m'},'Link kind: '+ex.edge.kind+' ('+ex.edge.trust+', seen in the trace, not inferred)'));
    host.appendChild(v);
    read.textContent=mode==='plain'?'Plain words: each step is one call to the language model.':'Stored names: the labels exactly as saved in the database.';
  }
  ctl.appendChild(C.el('span','','Labels:'));
  ctl.appendChild(C.seg([{id:'plain',label:'Plain words'},{id:'raw',label:'Stored names'}],'plain',draw));
  panel.appendChild(ctl);panel.appendChild(host);panel.appendChild(read);draw('plain');
}

function replayTab(panel,data){
  var C=RSCard,ex=data.example,rp=ex.replay;
  var rs=data.stages.filter(function(s){return s.name==='Replay';})[0];
  panel.appendChild(C.el('p','sh-cap','The graph is turned into LangGraph code (a workflow library) and run again. The question: does the replay give the same answer?'));
  var ctl=C.el('div','rc-ctl'),box=C.el('div','sh-res');box.setAttribute('aria-live','polite');
  function show(id){
    box.textContent='';
    if(id==='orig'){
      box.appendChild(C.el('p','','Original run: the model read the task, planned, and answered "right".'));
    }else{
      var p=C.el('p','');p.appendChild(document.createTextNode('Replay from generated code: status '+rp.status+', answer '));
      p.appendChild(C.el('b','',rp.answer));p.appendChild(document.createTextNode(rp.match?', which matches the original.':', which does not match.'));box.appendChild(p);
      box.appendChild(C.el('p','rc-line','This is the one worked example. Overall: '+rs.count+'. Status: '+rs.status+'.'));
    }
  }
  ctl.appendChild(C.el('span','','Show:'));
  ctl.appendChild(C.seg([{id:'orig',label:'Original run'},{id:'replay',label:'Replayed as code'}],'replay',show));
  panel.appendChild(ctl);panel.appendChild(box);show('replay');
}

RSVis.mods[ID]=function(el,data){
  if(!document.getElementById('sh-css')){var s=document.createElement('style');s.id='sh-css';s.textContent=CSS;document.head.appendChild(s);}
  RSCard.mount(el,data,{tabs:[
    {id:'blocks',label:'Trace to blocks',render:blocksTab},
    {id:'graph',label:'Workflow graph',render:graphTab},
    {id:'replay',label:'Replayed as code',render:replayTab}]});
};
})();
