window.RSVis = window.RSVis || {mods:{}};
(function(){
  var S = '.rs-vis[data-vis="cerberus"]';
  var CSS = [
    S+'{font-family:var(--font-sans);color:var(--text-color);}',
    S+' *{box-sizing:border-box;}',
    S+' button{font:inherit;color:inherit;cursor:pointer;}',
    S+' button:focus-visible,'+S+' input:focus-visible{outline:3px solid var(--focus);outline-offset:2px;}',
    S+' .cb-title{font-size:1.05rem;font-weight:600;margin:0 0 .6rem;}',
    S+' .cb-tabs{display:flex;gap:.4rem;flex-wrap:wrap;border-bottom:1px solid var(--border-color);margin-bottom:.9rem;}',
    S+' .cb-tab{background:none;border:1px solid transparent;border-bottom:0;border-radius:6px 6px 0 0;padding:.5rem .8rem;font-size:.95rem;}',
    S+' .cb-tab[aria-selected="true"]{border-color:var(--border-color);background:var(--surface);font-weight:600;margin-bottom:-1px;}',
    S+' .cb-lab{font-size:.9rem;color:var(--mut);margin:.2rem 0 .5rem;}',
    S+' .cb-row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;margin-bottom:.9rem;}',
    S+' .cb-arrow{align-self:center;color:var(--mut);font-size:1.1rem;}',
    S+' .cb-box{flex:1 1 150px;min-width:130px;text-align:left;background:var(--surface);border:1px solid var(--border-color);border-radius:8px;padding:.6rem .7rem;display:flex;flex-direction:column;gap:.45rem;}',
    S+' .cb-box[aria-pressed="true"]{border-color:var(--text-color);box-shadow:0 0 0 2px var(--text-color) inset;}',
    S+' .cb-box.proposed{background:none;border:2px dashed var(--text-color);}',
    S+' .cb-name{font-size:.95rem;font-weight:600;line-height:1.25;}',
    S+' .cb-chip{align-self:flex-start;display:inline-flex;align-items:center;gap:.35rem;font-size:.85rem;padding:.1rem .5rem;border-radius:4px;border:1px solid var(--text-color);color:var(--text-color);}',
    S+' .cb-chip.built{background:color-mix(in srgb,var(--ok) 12%,var(--bg-color));}',
    S+' .cb-chip.partial{background:color-mix(in srgb,var(--warn) 14%,var(--bg-color));border-style:double;border-width:3px;}',
    S+' .cb-chip.proposed{background:none;border-style:dashed;border-width:2px;}',
    S+' .cb-chip.failed{background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 20%,var(--bg-color)) 0 4px,var(--bg-color) 4px 8px);}',
    S+' .cb-chip span{background:var(--bg-color);padding:0 .25rem;border-radius:3px;}',
    S+' .cb-detail{border:1px solid var(--border-color);border-radius:8px;padding:.7rem .9rem;font-size:.95rem;line-height:1.5;min-height:3.6rem;background:var(--surface);}',
    S+' .cb-range{width:100%;margin:.4rem 0;accent-color:var(--text-color);}',
    S+' .cb-stops{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:.95rem;margin-bottom:.8rem;}',
    S+' .cb-stops button{background:none;border:0;padding:.2rem .4rem;text-decoration:underline;}',
    S+' .cb-stops button[aria-pressed="true"]{font-weight:700;}',
    S+' .cb-vec{display:grid;grid-template-columns:repeat(32,1fr);gap:2px;margin:.3rem 0;}',
    S+' .cb-cell{height:26px;border-radius:2px;border:1px solid var(--text-color);}',
    S+' .cb-cell.on{background:var(--text-color);}',
    S+' .cb-cell.off{background:none;border-style:dashed;opacity:.4;}',
    S+' .cb-key{display:flex;justify-content:space-between;gap:.5rem;font-size:.9rem;margin-bottom:.8rem;}',
    S+' .cb-big{font-size:2.2rem;font-weight:700;line-height:1.1;font-family:var(--font-mono);}',
    S+' .cb-bar{height:10px;border:1px solid var(--text-color);border-radius:5px;margin:.4rem 0;overflow:hidden;}',
    S+' .cb-bar i{display:block;height:100%;background:var(--text-color);}',
    S+' .cb-metric{font-size:.9rem;color:var(--mut);}',
    S+' .cb-cap{margin:.8rem 0 0;font-size:.95rem;font-weight:600;}',
    S+' .cb-idea{margin-top:1.4rem;border-top:1px solid var(--border-color);padding-top:1rem;}',
    S+' .cb-idea h3{margin:0 0 .3rem;font-size:1.05rem;display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;}',
    S+' .cb-cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin-top:.6rem;}',
    S+' .cb-case{border:1px solid var(--border-color);border-radius:8px;padding:.7rem;background:var(--surface);}',
    S+' .cb-q{font-weight:600;font-size:.95rem;margin:0 0 .5rem;}',
    S+' .cb-grid{display:grid;gap:3px;margin-bottom:.5rem;}',
    S+' .cb-p{aspect-ratio:1/1;border-radius:3px;border:1px solid var(--text-color);}',
    S+' .cb-p.k{background:var(--text-color);}',
    S+' .cb-p.d{border-style:dashed;opacity:.35;}',
    S+' .cb-note{font-size:.9rem;margin:0;}',
    '@media (max-width:560px){'+S+' .cb-arrow{display:none;}'+S+' .cb-box{flex-basis:100%;}'+S+' .cb-big{font-size:1.9rem;}}',
    '@media (prefers-reduced-motion:no-preference){'+S+' .cb-bar i{transition:width .25s ease;}'+S+' .cb-cell{transition:opacity .2s ease;}}'
  ].join('\n');

  function el(tag, cls, text, parent){
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  function chip(parent, status, label){
    var c = el('span', 'cb-chip ' + status, null, parent);
    var glyph = {built:'✓', partial:'◐', proposed:'□', failed:'✕'}[status];
    el('span', null, glyph, c).setAttribute('aria-hidden', 'true');
    el('b', null, label, c).style.fontWeight = '600';
    return c;
  }

  function mountPipeline(panel, data, detail){
    var boxes = [];
    function select(btn, item){
      boxes.forEach(function(b){ b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'); });
      detail.textContent = item.name + ' (' + data.statuses[item.status] + '). ' + item.text;
    }
    function row(label, items, arrows){
      el('p', 'cb-lab', label, panel);
      var g = el('div', 'cb-row', null, panel);
      g.setAttribute('role', 'group');
      g.setAttribute('aria-label', label);
      items.forEach(function(it, i){
        if (arrows && i) el('span', 'cb-arrow', '→', g).setAttribute('aria-hidden', 'true');
        var b = el('button', 'cb-box' + (it.status === 'proposed' ? ' proposed' : ''), null, g);
        b.type = 'button';
        b.setAttribute('aria-pressed', 'false');
        el('span', 'cb-name', it.name, b);
        chip(b, it.status, data.statuses[it.status]);
        b.addEventListener('click', function(){ select(b, it); });
        boxes.push(b);
      });
    }
    row(data.pipelineLabel, data.pipeline, true);
    row(data.sideLabel, data.side, false);
  }

  function mountShrink(panel, data){
    var s = data.shrink, N = 32;
    var wrap = el('div', null, null, panel);
    var lab = el('label', 'cb-lab', s.label, wrap);
    var range = el('input', 'cb-range', null, wrap);
    range.type = 'range'; range.min = 0; range.max = s.sizes.length - 1; range.step = 1; range.value = 0;
    range.id = 'cb-range-' + Math.random().toString(36).slice(2, 7);
    lab.htmlFor = range.id;
    var stops = el('div', 'cb-stops', null, wrap);
    stops.setAttribute('role', 'group'); stops.setAttribute('aria-label', s.label + ' stops');
    var sb = s.sizes.map(function(sz, i){
      var b = el('button', null, String(sz), stops);
      b.type = 'button';
      b.addEventListener('click', function(){ range.value = i; render(); });
      return b;
    });
    el('p', 'cb-lab', s.vectorLabel, wrap);
    var vec = el('div', 'cb-vec', null, wrap);
    vec.setAttribute('aria-hidden', 'true');
    var cells = [];
    for (var i = 0; i < N; i++) cells.push(el('div', 'cb-cell', null, vec));
    var key = el('div', 'cb-key', null, wrap);
    var kl = el('span', null, null, key), dl = el('span', null, null, key);
    el('p', 'cb-metric', s.note, wrap);
    var big = el('div', 'cb-big', null, wrap);
    var bar = el('div', 'cb-bar', null, wrap); bar.setAttribute('aria-hidden', 'true');
    var fill = el('i', null, null, bar);
    el('div', 'cb-metric', s.metric, wrap);
    var live = el('p', 'cb-lab', null, wrap);
    live.setAttribute('aria-live', 'polite');
    el('p', 'cb-cap', s.caption, panel);
    var total = s.sizes[0];
    function render(){
      var i = +range.value, sz = s.sizes[i], keep = Math.round(sz / total * N);
      cells.forEach(function(c, k){ c.className = 'cb-cell ' + (k < keep ? 'on' : 'off'); });
      sb.forEach(function(b, k){ b.setAttribute('aria-pressed', k === i ? 'true' : 'false'); });
      kl.textContent = '✓ ' + s.keptText + ': first ' + sz + ' of ' + total;
      dl.textContent = sz === total ? '' : '✕ ' + s.droppedText;
      big.textContent = s.recall[i].toFixed(1);
      fill.style.width = s.recall[i] + '%';
      range.setAttribute('aria-valuetext', sz + ' dimensions, recall at 5 is ' + s.recall[i].toFixed(1));
      live.textContent = 'At ' + sz + ' dimensions, ' + s.metric + ': ' + s.recall[i].toFixed(1);
    }
    range.addEventListener('input', render);
    render();
  }

  function mountIdea(root, data){
    var d = data.idea;
    var sec = el('section', 'cb-idea', null, root);
    sec.setAttribute('aria-label', d.title);
    var h = el('h3', null, d.title, sec);
    chip(h, 'proposed', d.chip);
    el('p', 'cb-lab', d.badge + '. ' + d.mock, sec);
    var cases = el('div', 'cb-cases', null, sec);
    d.cases.forEach(function(c, ci){
      var box = el('div', 'cb-case', null, cases);
      el('p', 'cb-q', c.q, box);
      var g = el('div', 'cb-grid', null, box);
      g.style.gridTemplateColumns = 'repeat(' + d.cols + ',1fr)';
      g.setAttribute('role', 'img');
      g.setAttribute('aria-label', 'Mock patch grid for the question ' + c.q + '. ' + c.note);
      for (var r = 0; r < d.rows; r++) for (var q = 0; q < d.cols; q++) {
        var keep;
        if (c.kept === 'most') keep = !((r === 0 && q < 3) || (r === d.rows - 1 && q > d.cols - 3) || (r + q) % 7 === 0);
        else keep = c.kept.some(function(p){ return p[0] === r && p[1] === q; });
        el('div', 'cb-p ' + (keep ? 'k' : 'd'), null, g);
      }
      el('p', 'cb-note', c.note, box);
    });
  }

  RSVis.mods['cerberus'] = function mount(root, data, ctx){
    if (!document.getElementById('rs-vis-cerberus-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-cerberus-css'; st.textContent = CSS;
      document.head.appendChild(st);
    }
    root.textContent = '';
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', data.title);
    el('p', 'cb-title', data.title, root);
    var tl = el('div', 'cb-tabs', null, root);
    tl.setAttribute('role', 'tablist'); tl.setAttribute('aria-label', data.title);
    var panels = [], tabs = [];
    var uid = 'cb' + Math.random().toString(36).slice(2, 6);
    data.tabs.forEach(function(name, i){
      var t = el('button', 'cb-tab', name, tl);
      t.type = 'button'; t.id = uid + 't' + i;
      t.setAttribute('role', 'tab');
      t.setAttribute('aria-controls', uid + 'p' + i);
      tabs.push(t);
    });
    data.tabs.forEach(function(n, i){
      var p = el('div', null, null, root);
      p.id = uid + 'p' + i; p.setAttribute('role', 'tabpanel');
      p.setAttribute('aria-labelledby', uid + 't' + i);
      panels.push(p);
    });
    var detail = el('div', 'cb-detail', data.detailDefault, panels[0]);
    detail.setAttribute('aria-live', 'polite');
    var first = panels[0];
    var holder = el('div', null, null, first);
    first.insertBefore(holder, detail);
    mountPipeline(holder, data, detail);
    mountShrink(panels[1], data);
    function show(i, focus){
      tabs.forEach(function(t, k){
        t.setAttribute('aria-selected', k === i ? 'true' : 'false');
        t.tabIndex = k === i ? 0 : -1;
        panels[k].hidden = k !== i;
      });
      if (focus) tabs[i].focus();
    }
    tabs.forEach(function(t, i){
      t.addEventListener('click', function(){ show(i, false); });
      t.addEventListener('keydown', function(e){
        var n = tabs.length, k = null;
        if (e.key === 'ArrowRight') k = (i + 1) % n;
        else if (e.key === 'ArrowLeft') k = (i + n - 1) % n;
        else if (e.key === 'Home') k = 0;
        else if (e.key === 'End') k = n - 1;
        if (k != null) { e.preventDefault(); show(k, true); }
      });
    });
    show(0, false);
    mountIdea(root, data);
  };
})();
