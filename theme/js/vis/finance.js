window.RSVis = window.RSVis || {mods:{}};
(function(){
  var ID = 'finance';
  var P = '.rs-vis[data-vis="finance"]';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  var CSS = [
    '@{font-family:var(--font-sans);color:var(--text-color);margin:1rem 0}',
    '@ *{box-sizing:border-box}',
    '@ button{font:inherit;color:var(--text-color);background:var(--surface);border:1.5px solid var(--border-color);border-radius:6px;cursor:pointer}',
    '@ button:hover{border-color:var(--text-color)}',
    '@ button:focus-visible,@ summary:focus-visible,@ svg .mk:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    '@ .tag{display:inline-block;font-size:13px;line-height:1.4;padding:.2rem .55rem;border:1.5px dashed var(--warn);border-radius:4px;color:var(--text-color);margin:0 0 .75rem}',
    '@ .nav{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin:0 0 .6rem}',
    '@ .nav .mv{font-size:14px;padding:.45rem .8rem;min-height:40px}',
    '@ .nav .mv:disabled{opacity:.5;cursor:default}',
    '@ .dots{display:flex;gap:.35rem}',
    '@ .dot{width:40px;height:40px;padding:0;font-size:14px;font-weight:600;text-align:center;border-radius:50%}',
    '@ .dot[aria-current="step"]{background:var(--text-color);color:var(--bg-color);border-color:var(--text-color)}',
    '@ .count{font-size:13px;color:var(--mut);margin:0 0 .2rem}',
    '@ h4{font-size:17px;line-height:1.35;margin:0 0 .4rem}',
    '@ .body{font-size:15px;line-height:1.5;margin:0 0 .8rem;max-width:44rem}',
    '@ .cap{font-size:13px;line-height:1.5;color:var(--mut);margin:.5rem 0 0;max-width:44rem}',
    '@ .rows{list-style:none;margin:0;padding:0;display:grid;gap:.3rem;max-width:40rem}',
    '@ .row{display:grid;grid-template-columns:7.5rem 1fr 2.8rem;align-items:center;gap:.5rem;font-size:14px}',
    '@ .row .lab{text-align:right;color:var(--mut)}',
    '@ .row .trk{height:20px;position:relative}',
    '@ .row .bar{height:100%;background:var(--base-bar);border-radius:2px}',
    '@ .row .num{font-family:var(--font-mono);font-size:13px}',
    '@ .row.hi .bar{background:var(--text-color)}',
    '@ .row.hi .lab,@ .row.hi .num{color:var(--text-color);font-weight:700}',
    '@ .wide .row{grid-template-columns:minmax(8rem,15rem) 1fr 9.5rem}',
    '@ .wide .row .lab{color:var(--text-color);text-align:left}',
    '@ .wide .row .num{font-family:var(--font-sans);color:var(--mut)}',
    '@ .wide .row .trk{background:color-mix(in srgb,var(--border-color) 45%,transparent);border-radius:2px}',
    '@ .note{font-size:14px;font-weight:600;margin:.6rem 0 0}',
    '@ details{margin:.8rem 0 0;max-width:44rem}',
    '@ summary{cursor:pointer;font-size:14px;padding:.3rem 0;min-height:32px}',
    '@ .legend{display:flex;flex-wrap:wrap;gap:.4rem 1rem;font-size:13px;margin:.4rem 0 .6rem;padding:0;list-style:none}',
    '@ .legend li{display:flex;align-items:center;gap:.35rem}',
    '@ .sw{display:inline-block;width:18px;height:14px;border:1px solid var(--text-color)}',
    '@ .s-u{background:var(--text-color)}',
    '@ .s-m{background:var(--base-bar)}',
    '@ .s-c{background:repeating-linear-gradient(45deg,var(--text-color) 0 1.5px,transparent 1.5px 5px)}',
    '@ .stack{list-style:none;margin:0;padding:0;display:grid;gap:.45rem}',
    '@ .stack li{display:grid;grid-template-columns:minmax(7rem,11rem) 1fr;gap:.5rem;align-items:center;font-size:14px}',
    '@ .sb{display:flex;height:18px;border:1px solid var(--border-color)}',
    '@ .sb span{display:block;height:100%}',
    '@ .sb .s-c,@ .sb .s-m{border-left:1px solid var(--bg-color)}',
    '@ .scroll{overflow-x:auto;max-width:min(100%,780px);border:1px solid var(--border-color);border-radius:6px;background:var(--surface)}',
    '@ .scroll svg{display:block;width:100%;max-width:760px;min-width:560px;height:auto}',
    '@ svg text{fill:var(--text-color);font-family:var(--font-sans)}',
    '@ svg .ax{stroke:var(--border-color);stroke-width:1}',
    '@ svg .ln{fill:none;stroke:var(--text-color);stroke-width:2}',
    '@ svg .gd{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 3;opacity:.6}',
    '@ svg .mk{cursor:pointer}',
    '@ svg .mk circle{fill:color-mix(in srgb,var(--ok) 12%,var(--bg-color));stroke:var(--ok);stroke-width:2}',
    '@ svg .mk[aria-pressed="true"] circle{fill:var(--ok)}',
    '@ .live{font-size:14px;line-height:1.5;border-left:4px solid var(--text-color);background:var(--surface);padding:.5rem .75rem;margin:.6rem 0 0;min-height:2.6em;max-width:44rem}',
    '@ .chips{list-style:none;margin:0;padding:0;display:grid;gap:.5rem;max-width:44rem}',
    '@ .chip{font-size:14px;line-height:1.45;padding:.55rem .75rem;border-radius:6px;border:2px solid var(--ok);background:color-mix(in srgb,var(--ok) 10%,var(--bg-color))}',
    '@ .chip b{display:inline-block;font-size:12px;margin-right:.45rem;padding:.05rem .4rem;border:1.5px solid currentColor;border-radius:4px}',
    '@ .chip.tried{border-color:var(--bad);background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 10%,var(--bg-color)) 0 5px,var(--bg-color) 5px 10px)}',
    '@ .stage.anim{animation:rsfin .18s ease-out}',
    '@keyframes rsfin{from{opacity:0}to{opacity:1}}',
    '@media (max-width:520px){@ .row{grid-template-columns:5.6rem 1fr 2.6rem}@ .wide .row{grid-template-columns:1fr 6.8rem}@ .wide .row .lab{grid-column:1 / -1}@ .stack li{grid-template-columns:1fr}}'
  ].join('\n').split('@ ').join(P + ' ').split('@{').join(P + '{');

  function h(tag, cls, text){
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function s(tag, attrs){
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function parse(d){ var p = d.split('-'); return Date.UTC(+p[0], +p[1] - 1, +p[2]); }
  function monthYear(d){ var p = d.split('-'); return MONTHS[+p[1] - 1] + ' ' + p[0]; }
  function dayMonthYear(d){ var p = d.split('-'); return (+p[2]) + ' ' + MONTHS[+p[1] - 1].slice(0, 3) + ' ' + p[0]; }
  function rupees(v){ return String.fromCharCode(0x20b9) + v.toLocaleString('en-IN'); }

  function stepOverlap(data){
    var f = document.createDocumentFragment();
    var max = 0;
    data.hist.forEach(function(r){ if (r[1] > max) max = r[1]; });
    var ul = h('ul', 'rows');
    ul.setAttribute('aria-label', 'Number of stocks by how many fund houses hold them');
    data.hist.forEach(function(r){
      var li = h('li', 'row' + (r[0] === 1 ? ' hi' : ''));
      li.appendChild(h('span', 'lab', r[0] === 1 ? 'Just one' : 'Held by ' + r[0]));
      var t = h('span', 'trk'); var b = h('span', 'bar'); b.style.display = 'block';
      b.style.width = (100 * r[1] / max) + '%'; t.appendChild(b); li.appendChild(t);
      li.appendChild(h('span', 'num', String(r[1])));
      ul.appendChild(li);
    });
    f.appendChild(ul);
    f.appendChild(h('p', 'cap', 'Each bar counts stocks. The label says how many of the ' + data.namcs + ' fund houses hold that stock.'));
    return f;
  }

  function stepFavourites(data){
    var f = document.createDocumentFragment();
    var ul = h('ul', 'rows wide');
    ul.setAttribute('aria-label', 'Most widely held stocks');
    data.most_held.forEach(function(m){
      var li = h('li', 'row');
      li.appendChild(h('span', 'lab', m.name.replace(/ Limited$/, '')));
      var t = h('span', 'trk'); var b = h('span', 'bar'); b.style.display = 'block';
      b.style.width = (100 * m.amcs / data.namcs) + '%'; t.appendChild(b); li.appendChild(t);
      li.appendChild(h('span', 'num', m.amcs + ' of ' + data.namcs + ' fund houses'));
      ul.appendChild(li);
    });
    f.appendChild(ul);
    var top = 0;
    data.hist.forEach(function(r){ if (r[1] > 0 && r[0] > top) top = r[0]; });
    if (top < data.namcs) f.appendChild(h('p', 'note', 'No stock is held by all ' + data.namcs + '.'));

    var det = h('details');
    det.appendChild(h('summary', null, 'See each fund'));
    det.appendChild(h('p', 'cap', 'Each bar is one small-cap fund. It splits the fund into stocks only this fund house owns, and stocks it shares.'));
    var lg = h('ul', 'legend');
    [['s-u', 'Only this fund house (solid)'], ['s-m', 'Shared with a few (grey)'], ['s-c', 'Shared with many (striped)']].forEach(function(x){
      var li = h('li'); li.appendChild(h('span', 'sw ' + x[0])); li.appendChild(h('span', null, x[1])); lg.appendChild(li);
    });
    det.appendChild(lg);
    var seen = {}, total = {};
    data.funds.forEach(function(fd){ total[fd.label] = (total[fd.label] || 0) + 1; });
    var st = h('ul', 'stack');
    data.funds.forEach(function(fd){
      seen[fd.label] = (seen[fd.label] || 0) + 1;
      var name = fd.label + (total[fd.label] > 1 ? (seen[fd.label] === 1 ? ' (first fund)' : ' (second fund)') : '');
      var li = h('li'); li.appendChild(h('span', null, name));
      var bar = h('span', 'sb');
      var parts = [['s-u', fd.u, 'only this fund house'], ['s-m', fd.m, 'a few'], ['s-c', fd.c, 'many']];
      var sum = fd.u + fd.m + fd.c;
      parts.forEach(function(p){ var sp = h('span', p[0]); sp.style.width = (100 * p[1] / sum) + '%'; bar.appendChild(sp); });
      var order = parts.slice().sort(function(a, b){ return b[1] - a[1]; });
      bar.setAttribute('role', 'img');
      bar.setAttribute('aria-label', name + ': largest part shared with ' + (order[0][2] === 'only this fund house' ? 'nobody, only this fund house' : order[0][2]) + ', smallest part ' + order[2][2]);
      li.appendChild(bar);
      st.appendChild(li);
    });
    det.appendChild(st);
    det.appendChild(h('p', 'cap', 'Funds sorted by their share held by nobody else, largest first.'));
    f.appendChild(det);
    return f;
  }

  function stepAther(data, live){
    var a = data.ather;
    var f = document.createDocumentFragment();
    var W = 640, H = 300, L = 100, R = 20, T = 20, B = 92;
    var x0 = Math.min(parse(a.prices[0][0]), parse(a.events[0].d));
    var x1 = Math.max(parse(a.prices[a.prices.length - 1][0]), parse(a.events[a.events.length - 1].d));
    var lo = Infinity, hi = -Infinity, loI = 0, hiI = 0;
    a.prices.forEach(function(p, i){ if (p[1] < lo) { lo = p[1]; loI = i; } if (p[1] > hi) { hi = p[1]; hiI = i; } });
    var plotB = H - B;
    function X(t){ return L + (W - L - R) * (t - x0) / (x1 - x0); }
    function Y(v){ return plotB - (plotB - T) * (v - lo) / (hi - lo); }
    var svg = s('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'group'});
    svg.setAttribute('aria-label', a.label + ' weekly price from ' + dayMonthYear(a.prices[0][0]) + ' to ' + dayMonthYear(a.prices[a.prices.length - 1][0]) + ', with markers where fund houses first reported holding it');
    svg.appendChild(s('line', {'class': 'ax', x1: L, y1: T, x2: L, y2: plotB}));
    svg.appendChild(s('line', {'class': 'ax', x1: L, y1: plotB, x2: W - R, y2: plotB}));
    [[lo, a.prices[loI][0]], [hi, a.prices[hiI][0]]].forEach(function(v){
      var t = s('text', {x: L - 6, y: Y(v[0]) + 4, 'text-anchor': 'end', 'font-size': 13}); t.textContent = rupees(v[0]); svg.appendChild(t);
      svg.appendChild(s('line', {'class': 'gd', x1: L, y1: Y(v[0]), x2: W - R, y2: Y(v[0])}));
    });
    var yl = s('text', {x: 14, y: (T + plotB) / 2, 'font-size': 13, 'text-anchor': 'middle', transform: 'rotate(-90 14 ' + ((T + plotB) / 2) + ')'});
    yl.textContent = 'Weekly price'; svg.appendChild(yl);
    var d = '';
    a.prices.forEach(function(p, i){ d += (i ? 'L' : 'M') + X(parse(p[0])).toFixed(1) + ' ' + Y(p[1]).toFixed(1); });
    var path = s('path', {'class': 'ln', d: d}); path.setAttribute('aria-hidden', 'true'); svg.appendChild(path);
    [[a.prices[0][0], 'start'], [a.prices[a.prices.length - 1][0], 'end']].forEach(function(v){
      var t = s('text', {x: X(parse(v[0])), y: plotB + 16, 'text-anchor': v[1], 'font-size': 13}); t.textContent = dayMonthYear(v[0]); svg.appendChild(t);
    });
    var laneY = H - 34;
    var lt = s('text', {x: L - 20, y: laneY + 4, 'text-anchor': 'end', 'font-size': 13}); lt.textContent = 'Joined'; svg.appendChild(lt);
    var marks = [];
    function show(i){
      marks.forEach(function(m, j){ m.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
      var ev = a.events[i];
      live.textContent = monthYear(ev.d) + ': ' + ev.amcs.join(', ');
    }
    a.events.forEach(function(ev, i){
      var cx = X(parse(ev.d));
      svg.appendChild(s('line', {'class': 'gd', x1: cx, y1: T, x2: cx, y2: laneY}));
      var g = s('g', {'class': 'mk', tabindex: 0, role: 'button', 'aria-pressed': 'false'});
      g.setAttribute('aria-label', monthYear(ev.d) + ', ' + ev.amcs.length + (ev.amcs.length === 1 ? ' fund house' : ' fund houses'));
      g.appendChild(s('circle', {cx: cx, cy: laneY, r: 5 + 2.5 * ev.amcs.length}));
      g.addEventListener('click', function(){ show(i); });
      g.addEventListener('focus', function(){ show(i); });
      g.addEventListener('keydown', function(e){
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(i); }
        else if (e.key === 'ArrowRight' && i < marks.length - 1) { e.preventDefault(); marks[i + 1].focus(); }
        else if (e.key === 'ArrowLeft' && i > 0) { e.preventDefault(); marks[i - 1].focus(); }
      });
      marks.push(g); svg.appendChild(g);
    });
    var box = h('div', 'scroll'); box.appendChild(svg); f.appendChild(box);
    f.appendChild(h('p', 'cap', 'Bigger circle, more fund houses that month. Tab to a circle or click it.'));
    live.className = 'live';
    live.textContent = 'Pick a circle to see which fund houses first reported holding ' + a.label + ' that month.';
    f.appendChild(live);
    f.appendChild(h('p', 'cap', 'Funds report once a month, so a marker sits on the month-end of the portfolio, not the day they bought. Description only, not a prediction.'));
    return f;
  }

  function stepTakeaway(){
    var f = document.createDocumentFragment();
    var ul = h('ul', 'chips');
    [['', 'Finding', 'Fund managers disagree a lot.'],
     ['', 'Finding', 'Some stocks attract many funds over time.'],
     [' tried', 'Tried, did not work', "Whether funds' picks predict returns: not shown. My backtest so far is negative and under review."]
    ].forEach(function(c){
      var li = h('li', 'chip' + c[0]); li.appendChild(h('b', null, c[1])); li.appendChild(document.createTextNode(c[2])); ul.appendChild(li);
    });
    f.appendChild(ul);
    return f;
  }

  RSVis.mods[ID] = function mount(el, data, ctx){
    ctx = ctx || {};
    if (!document.getElementById('rs-vis-finance-css')) {
      var st = document.createElement('style'); st.id = 'rs-vis-finance-css'; st.textContent = CSS; document.head.appendChild(st);
    }
    while (el.firstChild) el.removeChild(el.firstChild);
    var steps = [
      {t: 'Do fund managers agree on what to buy? Mostly not.',
       b: 'Indian mutual funds publish their holdings every month. I looked at ' + data.nfunds + ' small-cap funds from ' + data.namcs + ' fund houses. Of ' + data.nstocks + ' stocks, ' + data.hist[0][1] + ' (about half) are held by just one fund house.',
       f: function(){ return stepOverlap(data); }},
      {t: 'Even the favourite stocks are held by about three in four.',
       b: 'These are the most widely held stocks. Even the top ones miss a few fund houses.',
       f: function(){ return stepFavourites(data); }},
      {t: 'When funds do agree, here is what it looked like: ' + data.ather.label + '.',
       b: 'The line is the weekly share price. Each circle is a month when new fund houses first reported owning it.',
       f: function(){ return stepAther(data, h('p')); }},
      {t: 'What I take from it, and what I do not.',
       b: 'Two things the data shows, and one thing it does not.',
       f: stepTakeaway}
    ];
    var cur = 0;
    var root = h('div');
    var tag = h('p', 'tag', data.tag); root.appendChild(tag);
    var nav = h('div', 'nav');
    nav.setAttribute('role', 'group');
    nav.setAttribute('aria-label', 'Fund overlap story steps');
    var back = h('button', 'mv', 'Back'); back.type = 'button';
    var dots = h('div', 'dots');
    var next = h('button', 'mv', 'Next'); next.type = 'button';
    var dotBtns = steps.map(function(st, i){
      var b = h('button', 'dot', String(i + 1)); b.type = 'button';
      b.setAttribute('aria-label', 'Step ' + (i + 1) + ': ' + st.t);
      b.addEventListener('click', function(){ go(i, true); });
      dots.appendChild(b); return b;
    });
    nav.appendChild(back); nav.appendChild(dots); nav.appendChild(next);
    root.appendChild(nav);
    var head = h('div'); head.setAttribute('aria-live', 'polite');
    var count = h('p', 'count'); var title = h('h4'); var body = h('p', 'body');
    head.appendChild(count); head.appendChild(title); head.appendChild(body);
    root.appendChild(head);
    var stage = h('div', 'stage'); root.appendChild(stage);
    el.appendChild(root);

    function go(i, focusTitle){
      cur = Math.max(0, Math.min(steps.length - 1, i));
      var st = steps[cur];
      count.textContent = 'Step ' + (cur + 1) + ' of ' + steps.length;
      title.textContent = st.t; body.textContent = st.b;
      dotBtns.forEach(function(b, j){ if (j === cur) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
      back.disabled = cur === 0; next.disabled = cur === steps.length - 1;
      while (stage.firstChild) stage.removeChild(stage.firstChild);
      stage.classList.remove('anim');
      stage.appendChild(st.f());
      if (!ctx.reduceMotion) { void stage.offsetWidth; stage.classList.add('anim'); }
      if (focusTitle === 'next' && next.disabled) back.focus();
      if (focusTitle === 'back' && back.disabled) next.focus();
    }
    back.addEventListener('click', function(){ go(cur - 1, 'back'); });
    next.addEventListener('click', function(){ go(cur + 1, 'next'); });
    go(0);
  };
})();
