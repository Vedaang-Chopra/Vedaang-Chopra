window.RSVis = window.RSVis || {mods:{}};
(function(){
  var ID = 'cerberus';
  var S = '.rs-vis[data-vis="cerberus"]';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var CSS = [
    S+'{font-family:var(--font-sans);color:var(--text-color);}',
    S+' *{box-sizing:border-box;}',
    S+' button{font:inherit;font-size:14px;color:var(--text-color);background:var(--surface);border:1.5px solid var(--border-color);border-radius:6px;padding:.45rem .75rem;cursor:pointer;min-height:40px;}',
    S+' button:hover{border-color:var(--text-color);}',
    S+' button:disabled{cursor:default;opacity:.55;}',
    S+' button:focus-visible,'+S+' input:focus-visible,'+S+' summary:focus-visible{outline:3px solid var(--focus);outline-offset:2px;}',
    S+' button[aria-pressed="true"]{border-color:var(--text-color);background:color-mix(in srgb,var(--bg-color) 80%,var(--text-color));font-weight:600;}',
    S+' .cb-nav{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin:0 0 .8rem;}',
    S+' .cb-dots{display:flex;gap:.35rem;flex:1 1 auto;justify-content:center;}',
    S+' .cb-dot{min-width:40px;padding:.3rem .5rem;border-radius:20px;text-align:center;font-family:var(--font-mono);}',
    S+' .cb-dot[aria-current="step"]{border-color:var(--text-color);background:var(--text-color);color:var(--bg-color);font-weight:700;}',
    S+' .cb-count{font-size:13px;color:var(--mut);margin:0 0 .2rem;}',
    S+' .cb-title{font-size:17px;font-weight:600;line-height:1.35;margin:0 0 .4rem;}',
    S+' .cb-text{font-size:15px;line-height:1.5;margin:0 0 .8rem;max-width:62ch;}',
    S+' .cb-panel.cb-anim{animation:cbFade .25s ease-out;}',
    '@keyframes cbFade{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:none;}}',
    S+' .cb-pic{max-width:460px;}',
    S+' .cb-pic svg{display:block;width:100%;height:auto;border:1px solid var(--border-color);border-radius:6px;background:var(--surface);}',
    S+' .cb-sky{fill:var(--surface);}',
    S+' .cb-ground{fill:color-mix(in srgb,var(--ok) 13%,var(--bg-color));}',
    S+' .cb-sun{fill:color-mix(in srgb,var(--warn) 55%,var(--bg-color));}',
    S+' .cb-leaf{fill:color-mix(in srgb,var(--ok) 60%,var(--bg-color));}',
    S+' .cb-wood{fill:color-mix(in srgb,var(--text-color) 55%,var(--bg-color));}',
    S+' .cb-house{fill:color-mix(in srgb,var(--base-bar) 45%,var(--bg-color));stroke:var(--text-color);stroke-width:1.5;}',
    S+' .cb-board{fill:color-mix(in srgb,var(--bad) 60%,var(--bg-color));stroke:var(--text-color);stroke-width:2;}',
    S+' .cb-p{stroke:var(--border-color);stroke-width:1;fill:color-mix(in srgb,var(--bg-color) 74%,transparent);}',
    S+' .cb-p.on{fill:color-mix(in srgb,var(--focus) 14%,transparent);stroke:var(--text-color);stroke-width:2.5;}',
    S+' .cb-anim-ok .cb-p{transition:fill .25s,stroke-width .25s;}',
    S+' .cb-counter{font-size:15px;margin:.6rem 0 0;}',
    S+' .cb-counter b{font-size:17px;}',
    S+' .cb-qs{display:flex;flex-wrap:wrap;gap:.5rem;margin:0 0 .7rem;}',
    S+' .cb-stamp{display:inline-block;font-size:13px;font-weight:600;border:2px dashed var(--text-color);border-radius:4px;padding:.2rem .5rem;margin:0 0 .7rem;}',
    S+' figure.cb-fig{margin:0;}',
    S+' .cb-fig img{display:block;width:100%;max-width:760px;height:auto;border:1px solid var(--border-color);border-radius:6px;}',
    S+' .cb-fig figcaption{font-size:14px;line-height:1.5;margin-top:.5rem;max-width:70ch;}',
    S+' .cb-tag{display:inline-block;font-size:12px;font-weight:600;border:1.5px dashed var(--text-color);border-radius:4px;padding:.05rem .4rem;margin-right:.4rem;}',
    S+' .cb-sizes{display:flex;gap:.5rem;flex-wrap:wrap;margin:0 0 .4rem;font-family:var(--font-mono);}',
    S+' .cb-range{width:100%;max-width:420px;margin:.2rem 0 .7rem;accent-color:var(--text-color);display:block;}',
    S+' .cb-vec{display:flex;gap:2px;margin:.2rem 0 .35rem;max-width:620px;}',
    S+' .cb-cell{flex:1 1 0;height:26px;border-radius:2px;border:1.5px solid var(--text-color);background:color-mix(in srgb,var(--rs-link) 70%,var(--bg-color));}',
    S+' .cb-cell.off{background:none;border:1.5px dashed var(--mut);}',
    S+' .cb-anim-ok .cb-cell{transition:opacity .2s,background .2s;}',
    S+' .cb-mark{font-size:13px;margin:0 0 .7rem;}',
    S+' .cb-read{border-left:4px solid var(--text-color);background:var(--surface);padding:.55rem .8rem;margin:0 0 .6rem;max-width:620px;}',
    S+' .cb-read .cb-num{font-family:var(--font-mono);font-size:22px;font-weight:700;}',
    S+' .cb-bar{height:10px;border:1px solid var(--text-color);border-radius:3px;margin:.35rem 0;max-width:420px;}',
    S+' .cb-bar i{display:block;height:100%;background:var(--text-color);}',
    S+' .cb-gloss{font-size:13px;line-height:1.45;margin:.2rem 0 0;}',
    S+' .cb-cap{font-size:14px;font-weight:600;margin:0;}',
    S+' .cb-chips{display:flex;flex-direction:column;gap:.5rem;max-width:620px;margin:0 0 .7rem;}',
    S+' .cb-chip{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.4rem;text-align:left;}',
    S+' .cb-chip .cb-n{font-weight:600;}',
    S+' .st{display:inline-block;font-size:12px;line-height:1.25;padding:.15rem .45rem;border-radius:4px;border:1.5px solid var(--text-color);font-weight:600;color:var(--text-color);}',
    S+' .st.built{background:color-mix(in srgb,var(--ok) 13%,var(--bg-color));border-color:var(--ok);}',
    S+' .st.partial{background:color-mix(in srgb,var(--warn) 13%,var(--bg-color));border-color:var(--warn);}',
    S+' .st.proposed{background:none;border-style:dashed;}',
    S+' .st.failed{border-color:var(--bad);background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 13%,var(--bg-color)) 0 4px,var(--bg-color) 4px 8px);}',
    S+' .st.failed span{background:var(--bg-color);padding:0 .2rem;border-radius:2px;}',
    S+' .cb-detail{font-size:14px;line-height:1.5;border-left:4px solid var(--text-color);background:var(--surface);padding:.6rem .8rem;min-height:3.2em;max-width:620px;}',
    S+' details.cb-more{margin-top:1rem;border-top:1px solid var(--border-color);padding-top:.6rem;}',
    S+' details.cb-more summary{cursor:pointer;font-size:14px;font-weight:600;padding:.3rem 0;}',
    S+' details.cb-more ul{margin:.4rem 0 0;padding-left:1.1rem;font-size:14px;line-height:1.5;}',
    S+' details.cb-more li{margin:.25rem 0;}',
    S+' .cb-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;}',
    '@media (max-width:480px){'+S+' .cb-dots{order:3;flex-basis:100%;}'+S+' .cb-nav{justify-content:space-between;}}'
  ].join('\n');

  function h(tag, cls, text){
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function sv(tag, attrs){
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function statusTag(st, text){
    var t = h('span', 'st ' + st);
    t.appendChild(h('span', null, text));
    return t;
  }

  function scene(grid, label){
    var C = 50, W = grid.cols * C, H = grid.rows * C;
    var svg = sv('svg', {viewBox:'0 0 '+W+' '+H, role:'img', 'aria-label':label});
    svg.appendChild(sv('rect', {'class':'cb-sky', x:0, y:0, width:W, height:H}));
    svg.appendChild(sv('rect', {'class':'cb-ground', x:0, y:210, width:W, height:H-210}));
    svg.appendChild(sv('circle', {'class':'cb-sun', cx:62, cy:58, r:30}));
    svg.appendChild(sv('rect', {'class':'cb-wood', x:94, y:160, width:14, height:62}));
    svg.appendChild(sv('circle', {'class':'cb-leaf', cx:101, cy:150, r:38}));
    svg.appendChild(sv('rect', {'class':'cb-house', x:150, y:150, width:84, height:64}));
    svg.appendChild(sv('path', {'class':'cb-house', d:'M142 152 L192 112 L242 152 Z'}));
    svg.appendChild(sv('rect', {'class':'cb-wood', x:296, y:118, width:8, height:98}));
    svg.appendChild(sv('rect', {'class':'cb-board', x:260, y:66, width:80, height:52, rx:4}));
    var cells = [];
    for (var r = 0; r < grid.rows; r++) {
      for (var c = 0; c < grid.cols; c++) {
        var p = sv('rect', {'class':'cb-p', x:c*C, y:r*C, width:C, height:C});
        p.dataset.rc = r + ',' + c;
        svg.appendChild(p);
        cells.push(p);
      }
    }
    return {svg:svg, cells:cells};
  }
  function keptSet(grid, kind){
    var set = {};
    if (kind === 'sign') {
      grid.sign.forEach(function(rc){ set[rc[0]+','+rc[1]] = 1; });
    } else {
      for (var r = 0; r < grid.rows; r++) for (var c = 0; c < grid.cols; c++) set[r+','+c] = 1;
      if (kind === 'most') { delete set['0,0']; delete set['0,'+(grid.cols-1)]; delete set['0,3']; }
    }
    return set;
  }
  function light(sc, set, label){
    sc.cells.forEach(function(p){ p.setAttribute('class', set[p.dataset.rc] ? 'cb-p on' : 'cb-p'); });
    sc.svg.setAttribute('aria-label', label);
  }

  function stepAll(panel, st, data){
    var sc = scene(data.grid, st.gridLabel);
    light(sc, keptSet(data.grid, 'all'), st.gridLabel);
    var pic = h('div', 'cb-pic'); pic.appendChild(sc.svg); panel.appendChild(pic);
    var cnt = h('p', 'cb-counter', st.counterLabel + ' ');
    cnt.appendChild(h('b', null, st.counterValue));
    panel.appendChild(cnt);
  }

  function stepChoose(panel, st, data){
    panel.appendChild(h('div', 'cb-stamp', st.stamp));
    var qs = h('div', 'cb-qs');
    qs.setAttribute('role', 'group');
    qs.setAttribute('aria-label', st.questionsLabel);
    panel.appendChild(qs);
    var sc = scene(data.grid, st.questions[0].gridLabel);
    var pic = h('div', 'cb-pic'); pic.appendChild(sc.svg); panel.appendChild(pic);
    var cnt = h('p', 'cb-counter');
    cnt.setAttribute('aria-live', 'polite');
    panel.appendChild(cnt);
    var btns = st.questions.map(function(q, i){
      var b = h('button', null, q.q);
      b.type = 'button';
      b.addEventListener('click', function(){ pick(i); });
      qs.appendChild(b);
      return b;
    });
    function pick(i){
      var q = st.questions[i];
      btns.forEach(function(b, j){ b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
      light(sc, keptSet(data.grid, q.kept), q.gridLabel);
      cnt.textContent = st.counterLabel + ' ';
      cnt.appendChild(h('b', null, q.count));
    }
    pick(0);
  }

  function stepFigure(panel, st){
    var fig = h('figure', 'cb-fig');
    var img = h('img');
    img.src = st.img;
    img.alt = st.alt;
    img.loading = 'lazy';
    img.decoding = 'async';
    fig.appendChild(img);
    var cap = h('figcaption');
    cap.appendChild(h('span', 'cb-tag', st.tag));
    cap.appendChild(document.createTextNode(st.caption));
    fig.appendChild(cap);
    panel.appendChild(fig);
  }

  function stepShrink(panel, st){
    var N = 32, max = st.sizes[0];
    var sizes = h('div', 'cb-sizes');
    sizes.setAttribute('role', 'group');
    sizes.setAttribute('aria-label', st.sizeLabel);
    panel.appendChild(sizes);
    var range = h('input', 'cb-range');
    range.type = 'range'; range.min = 0; range.max = st.sizes.length - 1; range.step = 1; range.value = 0;
    range.setAttribute('aria-label', st.sizeLabel);
    panel.appendChild(range);
    var vec = h('div', 'cb-vec');
    vec.setAttribute('role', 'img');
    panel.appendChild(vec);
    var cells = [];
    for (var i = 0; i < N; i++) { var c = h('span', 'cb-cell'); vec.appendChild(c); cells.push(c); }
    var mark = h('p', 'cb-mark');
    panel.appendChild(mark);
    var read = h('div', 'cb-read');
    read.setAttribute('aria-live', 'polite');
    var line = h('div');
    var name = h('span', null, st.metricName + ': ');
    var num = h('span', 'cb-num');
    line.appendChild(name); line.appendChild(num);
    var bar = h('div', 'cb-bar'); var fill = h('i'); bar.appendChild(fill);
    bar.setAttribute('aria-hidden', 'true');
    read.appendChild(line); read.appendChild(bar);
    read.appendChild(h('p', 'cb-gloss', st.metricGloss));
    panel.appendChild(read);
    panel.appendChild(h('p', 'cb-cap', st.caption));
    var btns = st.sizes.map(function(s, i){
      var b = h('button', null, String(s));
      b.type = 'button';
      b.addEventListener('click', function(){ set(i); });
      sizes.appendChild(b);
      return b;
    });
    function set(i){
      var s = st.sizes[i], keep = Math.max(1, Math.round(N * s / max));
      range.value = i;
      range.setAttribute('aria-valuetext', String(s));
      btns.forEach(function(b, j){ b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
      cells.forEach(function(c, j){ c.className = j < keep ? 'cb-cell' : 'cb-cell off'; });
      var m = s === max ? st.keptText + ': all ' + s + ' numbers. Nothing ' + st.droppedText + '.'
        : st.keptText + ': the first ' + s + ' of ' + max + ' numbers. Dashed cells: ' + st.droppedText + '.';
      mark.textContent = m;
      vec.setAttribute('aria-label', st.vectorLabel + ' ' + m);
      num.textContent = String(st.recall[i]);
      fill.style.width = st.recall[i] + '%';
    }
    range.addEventListener('input', function(){ set(+range.value); });
    set(0);
  }

  function stepChips(panel, st, data){
    var box = h('div', 'cb-chips');
    box.setAttribute('role', 'group');
    box.setAttribute('aria-label', st.title);
    panel.appendChild(box);
    var det = h('div', 'cb-detail');
    det.setAttribute('aria-live', 'polite');
    panel.appendChild(det);
    var btns = st.chips.map(function(c, i){
      var b = h('button', 'cb-chip');
      b.type = 'button';
      b.appendChild(h('span', 'cb-n', c.name));
      b.appendChild(statusTag(c.status, c.statusText));
      b.addEventListener('click', function(){ pick(i); });
      box.appendChild(b);
      return b;
    });
    function pick(i){
      btns.forEach(function(b, j){ b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
      det.textContent = st.chips[i].text;
    }
    pick(0);
  }

  var BUILD = {all:stepAll, choose:stepChoose, figure:stepFigure, shrink:stepShrink, chips:stepChips};

  function mount(el, data, ctx){
    ctx = ctx || {};
    if (!document.getElementById('rs-vis-' + ID + '-css')) {
      var style = h('style', null, CSS);
      style.id = 'rs-vis-' + ID + '-css';
      document.head.appendChild(style);
    }
    while (el.firstChild) el.removeChild(el.firstChild);
    var root = h('div', ctx.reduceMotion ? '' : 'cb-anim-ok');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', data.label);
    el.appendChild(root);
    var steps = data.steps, n = steps.length, cur = 0;

    var nav = h('div', 'cb-nav');
    var back = h('button', null, data.back); back.type = 'button';
    var next = h('button', null, data.next); next.type = 'button';
    var dots = h('div', 'cb-dots');
    dots.setAttribute('role', 'group');
    dots.setAttribute('aria-label', 'Steps');
    var dotBtns = steps.map(function(s, i){
      var d = h('button', 'cb-dot', String(i + 1));
      d.type = 'button';
      d.setAttribute('aria-label', data.stepWord + ' ' + (i + 1) + ': ' + s.title);
      d.addEventListener('click', function(){ go(i); });
      dots.appendChild(d);
      return d;
    });
    nav.appendChild(back); nav.appendChild(dots); nav.appendChild(next);
    root.appendChild(nav);

    var head = h('div');
    head.setAttribute('aria-live', 'polite');
    var count = h('p', 'cb-count');
    var title = h('p', 'cb-title');
    var text = h('p', 'cb-text');
    head.appendChild(count); head.appendChild(title); head.appendChild(text);
    root.appendChild(head);
    var panel = h('div', 'cb-panel');
    root.appendChild(panel);

    var more = h('details', 'cb-more');
    more.appendChild(h('summary', null, data.more.summary));
    var ul = h('ul');
    data.more.items.forEach(function(it){
      var li = h('li');
      li.appendChild(statusTag(it.status, data.statuses[it.status]));
      li.appendChild(document.createTextNode(' '));
      li.appendChild(h('b', null, it.name));
      li.appendChild(document.createTextNode(': ' + it.text));
      ul.appendChild(li);
    });
    more.appendChild(ul);
    root.appendChild(more);

    back.addEventListener('click', function(){ go(cur - 1); });
    next.addEventListener('click', function(){ go(cur + 1); });

    function go(i){
      if (i < 0 || i >= n) return;
      cur = i;
      var st = steps[i];
      count.textContent = data.stepWord + ' ' + (i + 1) + ' ' + data.ofWord + ' ' + n;
      title.textContent = st.title;
      text.textContent = st.text;
      back.disabled = i === 0;
      next.disabled = i === n - 1;
      dotBtns.forEach(function(d, j){
        if (j === i) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
      });
      while (panel.firstChild) panel.removeChild(panel.firstChild);
      panel.className = 'cb-panel';
      if (!ctx.reduceMotion) { void panel.offsetWidth; panel.className = 'cb-panel cb-anim'; }
      BUILD[st.kind](panel, st, data);
      if (document.activeElement === back && back.disabled) next.focus();
      if (document.activeElement === next && next.disabled) back.focus();
    }
    go(0);
  }

  RSVis.mods[ID] = mount;
})();
