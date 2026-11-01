window.RSVis = window.RSVis || {mods:{}};
(function(){
  var ID = 'ar';
  var P = '.rs-vis[data-vis="ar"]';
  var CSS = [
    '@{font-family:var(--font-sans);color:var(--text-color);margin:1rem 0}',
    '@ *{box-sizing:border-box}',
    '@ button{font:inherit;font-size:14px;color:var(--text-color);background:var(--surface);border:1.5px solid var(--border-color);border-radius:6px;padding:.5rem .75rem;cursor:pointer;text-align:left;min-height:40px}',
    '@ button:hover{border-color:var(--text-color)}',
    '@ button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    '@ button[aria-pressed="true"]{border-color:var(--text-color);background:color-mix(in srgb,var(--bg-color) 80%,var(--text-color));font-weight:600}',
    '@ .idea{font-size:15px;line-height:1.5;margin:0 0 .75rem}',
    '@ .note{font-size:13px;line-height:1.5;margin:.5rem 0}',
    '@ .detail{font-size:14px;line-height:1.5;border-left:4px solid var(--text-color);background:var(--surface);padding:.6rem .8rem;margin:.75rem 0;min-height:3.2em}',
    '@ .st{display:inline-block;font-size:12px;line-height:1.2;padding:.15rem .45rem;border-radius:4px;border:1.5px solid var(--text-color);margin-left:.4rem;font-weight:600;vertical-align:middle}',
    '@ .st.built{background:color-mix(in srgb,var(--ok) 12%,var(--bg-color));border-color:var(--ok)}',
    '@ .st.part{background:color-mix(in srgb,var(--warn) 14%,var(--bg-color));border-color:var(--warn)}',
    '@ .st.gone{background:none;border-style:dashed}',
    '@ .bar{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}',
    '@ .lanes{display:grid;grid-template-columns:1fr;gap:1rem}',
    '@ .lane{border:1.5px solid var(--border-color);border-radius:8px;padding:.75rem}',
    '@ .lane h4{margin:0 0 .6rem;font-size:15px}',
    '@ .lane ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:.4rem}',
    '@ .lane li{position:relative}',
    '@ .lane li button{width:100%}',
    '@ .lane li+li::before{content:"";display:block;height:.35rem;margin-left:1.2rem;border-left:2px solid var(--border-color)}',
    '@ .lane li.cut button{border-style:dashed}',
    '@ .phone{width:240px;max-width:100%;margin:.5rem auto 0;border:3px solid var(--text-color);border-radius:22px;padding:.8rem .7rem;background:var(--surface)}',
    '@ .screen{min-height:150px;border:1.5px solid var(--border-color);border-radius:10px;background:var(--bg-color);padding:.6rem;font-size:13px;line-height:1.45}',
    '@ .screen .tag{font-weight:700;margin-bottom:.3rem}',
    '@ .screen pre{margin:0;font-family:var(--font-mono);font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}',
    '@ .screen p{margin:.15rem 0}',
    '@ .ill{font-size:13px;text-align:center;margin:.5rem 0 0}',
    '@ .play{display:block;margin:.6rem auto 0}',
    '@ .stage{transition:opacity .2s}',
    '@media (min-width:760px){@ .lanes.both{grid-template-columns:1fr 1fr}}'
  ].join('\n').replace(/@(?!media)/g, P);

  function h(tag, cls, text){
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function statusCls(s){ return /replaced/i.test(s) ? 'gone' : /prototype/i.test(s) ? 'part' : 'built'; }

  RSVis.mods[ID] = function mount(el, data, ctx){
    if (!document.getElementById('rs-vis-ar-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-ar-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    var reduce = ctx && ctx.reduceMotion;
    el.textContent = '';
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', data.title + ': earlier and current pipeline');

    el.appendChild(h('p', 'idea', data.idea));

    var mode = el.clientWidth >= 760 ? 'both' : 'new';
    var bar = h('div', 'bar');
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Choose which pipeline to show');
    var modes = [['old', 'Earlier pipeline'], ['new', 'Current pipeline'], ['both', 'Side by side']];
    var modeBtns = {};
    modes.forEach(function(m){
      var b = h('button', null, m[1]);
      b.type = 'button';
      b.addEventListener('click', function(){ mode = m[0]; render(); });
      modeBtns[m[0]] = b;
      bar.appendChild(b);
    });
    el.appendChild(bar);

    var lanesEl = h('div', 'lanes');
    el.appendChild(lanesEl);

    var detail = h('div', 'detail', 'Select a step to see what it does.');
    detail.setAttribute('aria-live', 'polite');
    el.appendChild(detail);

    el.appendChild(h('p', 'note', data.partial));

    var phone = h('div', 'phone');
    var screen = h('div', 'screen');
    screen.setAttribute('aria-live', 'polite');
    phone.appendChild(screen);
    el.appendChild(phone);
    el.appendChild(h('p', 'ill', data.phoneLabel));
    var play = h('button', 'play', 'Play the flow');
    play.type = 'button';
    el.appendChild(play);

    function render(){
      Object.keys(modeBtns).forEach(function(k){ modeBtns[k].setAttribute('aria-pressed', k === mode ? 'true' : 'false'); });
      lanesEl.className = 'lanes' + (mode === 'both' ? ' both' : '');
      lanesEl.textContent = '';
      data.lanes.forEach(function(l){
        if (mode !== 'both' && l.id !== mode) return;
        var lane = h('div', 'lane');
        var hd = h('h4', null, l.name);
        hd.appendChild(h('span', 'st ' + statusCls(l.status), l.status));
        lane.appendChild(hd);
        var ol = h('ol');
        l.steps.forEach(function(s){
          var li = h('li', /OCR/.test(s.label) ? 'cut' : '');
          var b = h('button', null, s.label);
          b.type = 'button';
          b.addEventListener('click', function(){
            detail.textContent = l.name + ', ' + s.label + ': ' + s.detail;
          });
          li.appendChild(b);
          ol.appendChild(li);
        });
        lane.appendChild(ol);
        lanesEl.appendChild(lane);
      });
    }

    function showFrame(i){
      var f = data.frames[i];
      screen.textContent = '';
      var box = h('div', 'stage');
      box.appendChild(h('div', 'tag', f.stage));
      if (f.json) {
        box.appendChild(h('pre', null, f.json.join('\n')));
      } else {
        f.lines.forEach(function(t){ box.appendChild(h('p', null, t)); });
      }
      screen.appendChild(box);
    }
    function showAll(){
      screen.textContent = '';
      data.frames.forEach(function(f){
        var box = h('div', 'stage');
        box.appendChild(h('div', 'tag', f.stage));
        if (f.json) box.appendChild(h('pre', null, f.json.join('\n')));
        else f.lines.forEach(function(t){ box.appendChild(h('p', null, t)); });
        screen.appendChild(box);
      });
    }

    var timer = null;
    play.addEventListener('click', function(){
      if (timer) { clearTimeout(timer); timer = null; }
      if (reduce) { showAll(); return; }
      var i = 0;
      (function step(){
        showFrame(i);
        i++;
        if (i < data.frames.length) timer = setTimeout(step, 1600);
        else timer = null;
      })();
    });

    render();
    if (reduce) showAll(); else showFrame(0);
  };
})();
