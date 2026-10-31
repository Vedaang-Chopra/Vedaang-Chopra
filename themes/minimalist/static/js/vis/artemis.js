window.RSVis = window.RSVis || { mods: {} };
(function () {
  'use strict';
  var ID = 'artemis';
  var P = '.rs-vis[data-vis="artemis"] ';
  var CSS = [
    P + '.av{font-family:var(--font-sans);color:var(--text-color);font-size:.92rem;line-height:1.45}',
    P + '.av-tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .8rem}',
    P + 'button{font:inherit;color:var(--text-color);background:var(--btn-bg);border:1px solid var(--border-color);border-radius:6px;padding:.35rem .7rem;cursor:pointer}',
    P + 'button:focus-visible,' + P + 'select:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    P + '.av-tab[aria-selected="true"]{border-color:var(--text-color);font-weight:600;box-shadow:inset 0 -3px 0 var(--text-color)}',
    P + '.av-head{font-size:1.05rem;font-weight:600;margin:0 0 .3rem}',
    P + '.av-mut{color:var(--mut)}',
    P + '.av-small{font-size:.85rem;color:var(--mut);margin:.3rem 0}',
    P + '.av-live{min-height:2.6em;margin:.4rem 0 .6rem;padding:.45rem .6rem;border-left:3px solid var(--plain-bar);background:var(--surface)}',
    P + '.av-scroll{overflow-x:auto;max-width:100%}',
    P + '.av-grid{min-width:600px}',
    P + '.av-row{display:grid;grid-template-columns:minmax(150px,1.6fr) repeat(5,minmax(70px,1fr)) minmax(96px,1fr);align-items:center;width:100%;text-align:left;border-radius:4px;padding:.18rem .3rem;background:transparent;border:1px solid transparent}',
    P + '.av-hdr{font-size:.8rem;color:var(--mut);font-weight:600;padding-bottom:.3rem;border-bottom:1px solid var(--border-color)}',
    P + '.av-hdr span{text-align:center;padding:0 .2rem}',
    P + '.av-hdr span:first-child{text-align:left}',
    P + 'button.av-row:hover,' + P + 'button.av-row[aria-pressed="true"]{border-color:var(--text-color);background:var(--surface)}',
    P + '.av-cell{display:flex;justify-content:center}',
    P + '.av-dot{flex:none;box-sizing:border-box;width:14px;height:14px;border-radius:50%;border:1px solid var(--base-bar)}',
    P + '.av-dot.on{width:18px;height:18px;background:var(--text-color);border-color:var(--text-color)}',
    P + '.av-dot.lg-cheap,' + P + '.av-row.cheap .av-dot.on{background:var(--ok);border-color:var(--ok);border-radius:3px}',
    P + '.av-dot.lg-dead,' + P + '.av-row.dead .av-dot.on{background:transparent;border:2px dashed var(--mut)}',
    P + '.av-tag{font-size:.78rem;color:var(--ok);font-weight:600}',
    P + '.av-tag.dead{color:var(--mut);font-weight:400}',
    P + '.av-row.cheap{box-shadow:inset 4px 0 0 var(--ok)}',
    P + '.av-tier{font-weight:600;margin:.7rem 0 .2rem;font-size:.88rem}',
    P + '.av-legend{display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;margin:.6rem 0 0;padding:0;list-style:none;font-size:.85rem;color:var(--mut)}',
    P + '.av-legend li{display:flex;align-items:center;gap:.4rem}',
    P + '.av-ctl{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:.3rem 0 .6rem}',
    P + 'select{font:inherit;color:var(--text-color);background:var(--bg-color);border:1px solid var(--border-color);border-radius:6px;padding:.3rem .4rem;max-width:100%}',
    P + '.av-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.4rem;margin:.4rem 0;padding:0;list-style:none}',
    P + '.av-steps button{width:100%;text-align:left;font-size:.85rem}',
    P + '.av-steps button[aria-current="step"]{border-color:var(--text-color);font-weight:600;background:var(--surface)}',
    P + '.av-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.5rem;margin:.6rem 0;padding:0;list-style:none}',
    P + '.av-card{border:1px solid var(--border-color);border-radius:6px;padding:.5rem .6rem;background:var(--bg-color);display:grid;gap:.25rem}',
    P + '.av-card.pick{border:3px solid var(--ok)}',
    P + '.av-cname{font-weight:600}',
    P + '.av-blab{font-size:.78rem;color:var(--mut)}',
    P + '.av-bar{height:8px;background:color-mix(in srgb,var(--base-bar) 30%,transparent);border-radius:2px}',
    P + '.av-bar i{display:block;height:100%;background:var(--plain-bar);border-radius:2px}',
    P + '.av-badge{font-size:.78rem;font-weight:600}',
    P + '.av-badge.ok{color:var(--ok)}',
    P + '.av-panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.7rem;margin-top:1rem}',
    P + '.av-panel{border:1px solid var(--border-color);border-radius:6px;padding:.6rem .75rem;background:var(--surface)}',
    P + '.av-panel h4{margin:0 0 .3rem;font-size:.92rem}',
    P + '.av-panel p{margin:.2rem 0}',
    P + '.av-st{list-style:none;margin:0;padding:0;display:grid;gap:.25rem}',
    P + '.av-st li{display:flex;gap:.45rem;align-items:baseline;flex-wrap:wrap}',
    P + '.av-sw{display:inline-block;width:12px;height:12px;flex:none;transform:translateY(1px)}',
    P + '.av-sw.built{background:var(--ok)}',
    P + '.av-sw.partial{background:linear-gradient(90deg,var(--warn) 50%,transparent 50%);outline:1.5px solid var(--warn);outline-offset:-1.5px}',
    P + '.av-big{font-size:1.3rem;font-weight:700}',
    P + '.av-note{font-size:.85rem;color:var(--mut);font-style:italic}',
    P + '.av-req{border:1px dashed var(--plain-bar);border-radius:6px;padding:.45rem .6rem;margin:.4rem 0}',
    P + '.anim .av-card,' + P + '.anim .av-row{transition:border-color .15s,background-color .15s}'
  ].join('\n');

  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'text') e.textContent = attrs[k];
      else if (k === 'cls') e.className = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }

  function mount(el, data, ctx) {
    ctx = ctx || {};
    if (!document.getElementById('rs-vis-artemis-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-artemis-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    var models = data.models.slice().sort(function (a, b) { return a.price_rank - b.price_rank; });
    var byId = {};
    models.forEach(function (m) { byId[m.id] = m; });
    var top = models[models.length - 1];
    var tierById = {};
    data.tiers.forEach(function (t) { tierById[t.id] = t; });
    function cheaper(t) { return t.tier !== 'dead' && byId[t.pick].price_rank < top.price_rank; }

    var root = h('div', { cls: 'av' + (ctx.reduceMotion ? '' : ' anim') });
    el.appendChild(root);

    var tabs = h('div', { cls: 'av-tabs', role: 'tablist', 'aria-label': 'ARTEMIS views' });
    var views = [
      { id: 'grid', name: 'Which model for which task' },
      { id: 'route', name: 'Route one request' }
    ];
    var panels = {}, tabBtns = [];
    views.forEach(function (v, i) {
      var b = h('button', { cls: 'av-tab', role: 'tab', id: 'av-tab-' + v.id, 'aria-controls': 'av-pan-' + v.id, 'aria-selected': i === 0 ? 'true' : 'false', tabindex: i === 0 ? '0' : '-1', type: 'button', text: v.name });
      b.addEventListener('click', function () { select(i); });
      b.addEventListener('keydown', function (e) {
        var n = null;
        if (e.key === 'ArrowRight') n = (i + 1) % views.length;
        else if (e.key === 'ArrowLeft') n = (i + views.length - 1) % views.length;
        else if (e.key === 'Home') n = 0;
        else if (e.key === 'End') n = views.length - 1;
        if (n !== null) { e.preventDefault(); select(n); tabBtns[n].focus(); }
      });
      tabBtns.push(b);
      tabs.appendChild(b);
    });
    root.appendChild(tabs);
    function select(i) {
      views.forEach(function (v, j) {
        tabBtns[j].setAttribute('aria-selected', j === i ? 'true' : 'false');
        tabBtns[j].tabIndex = j === i ? 0 : -1;
        panels[v.id].hidden = j !== i;
      });
    }

    /* View 1: grid */
    var g = h('div', { role: 'tabpanel', id: 'av-pan-grid', 'aria-labelledby': 'av-tab-grid' });
    panels.grid = g;
    g.appendChild(h('p', { cls: 'av-head', text: 'No single model is best everywhere.' }));
    g.appendChild(h('p', { cls: 'av-small', text: 'Rows are the ' + data.counts.families + ' task families, grouped by how well the best model does. Columns are the ' + data.counts.models + ' models, cheapest on the left. The filled mark is the model the per-task lookup picks. On a narrow screen the grid scrolls sideways.' }));
    var live = h('p', { cls: 'av-live', 'aria-live': 'polite', text: 'Hover, focus or click a task row to see its tier and chosen model.' });
    g.appendChild(live);
    var scroll = h('div', { cls: 'av-scroll' });
    var grid = h('div', { cls: 'av-grid', role: 'group', 'aria-label': 'Per-task model choice grid' });
    var hdr = h('div', { cls: 'av-row av-hdr', 'aria-hidden': 'true' }, [h('span', { text: 'Task family' })]);
    models.forEach(function (m) { hdr.appendChild(h('span', { text: m.name })); });
    hdr.appendChild(h('span', { text: 'Pick vs priciest' }));
    grid.appendChild(hdr);
    var rowBtns = [];
    function describe(t) {
      var tr = tierById[t.tier], m = byId[t.pick];
      var s = t.label + '. Tier: ' + tr.name.toLowerCase() + ' (' + tr.desc + '). Lookup picks ' + m.name + '.';
      if (t.tier === 'dead') s += ' Any pick here is weak, so it does not count as a win.';
      else if (cheaper(t)) s += ' That is cheaper than the priciest model, ' + top.name + '.';
      else s += ' That is the priciest model.';
      return s;
    }
    data.tiers.forEach(function (tr) {
      grid.appendChild(h('p', { cls: 'av-tier', text: tr.name + ': ' + tr.desc }));
      data.tasks.filter(function (t) { return t.tier === tr.id; }).forEach(function (t) {
        var cls = 'av-row' + (cheaper(t) ? ' cheap' : '') + (t.tier === 'dead' ? ' dead' : '');
        var b = h('button', { cls: cls, type: 'button', 'aria-pressed': 'false', 'aria-label': describe(t) });
        b.appendChild(h('span', { text: t.label }));
        models.forEach(function (m) {
          b.appendChild(h('span', { cls: 'av-cell' }, [h('span', { cls: 'av-dot' + (m.id === t.pick ? ' on' : '') })]));
        });
        var tag = t.tier === 'dead' ? h('span', { cls: 'av-tag dead', text: 'no model does well' }) :
          cheaper(t) ? h('span', { cls: 'av-tag', text: 'cheaper wins' }) : h('span', { cls: 'av-tag dead', text: 'priciest' });
        b.appendChild(tag);
        function show() { live.textContent = describe(t); }
        b.addEventListener('mouseenter', show);
        b.addEventListener('focus', show);
        b.addEventListener('click', function () {
          rowBtns.forEach(function (o) { o.setAttribute('aria-pressed', 'false'); });
          b.setAttribute('aria-pressed', 'true');
          show();
        });
        rowBtns.push(b);
        grid.appendChild(b);
      });
    });
    scroll.appendChild(grid);
    g.appendChild(scroll);
    function sw(cls) { return h('span', { cls: 'av-dot ' + cls, 'aria-hidden': 'true' }); }
    var lg = h('ul', { cls: 'av-legend', 'aria-label': 'Legend' });
    lg.appendChild(h('li', null, [sw('on'), h('span', { text: 'filled circle: lookup picks the priciest model' })]));
    lg.appendChild(h('li', null, [sw('on lg-cheap'), h('span', { text: 'green square and edge bar: a model cheaper than ' + top.name + ' wins' })]));
    lg.appendChild(h('li', null, [sw('on lg-dead'), h('span', { text: 'dashed circle: dead task, no model does well' })]));
    lg.appendChild(h('li', null, [sw(''), h('span', { text: 'empty circle: not picked' })]));
    g.appendChild(lg);
    root.appendChild(g);

    /* View 2: route one request */
    var r = h('div', { role: 'tabpanel', id: 'av-pan-route', 'aria-labelledby': 'av-tab-route', hidden: '' });
    panels.route = r;
    r.appendChild(h('p', { cls: 'av-head', text: 'Route one request to the cheapest model likely to be right.' }));
    r.appendChild(h('p', { cls: 'av-note', text: 'Illustration of the decision rule; scores are not shown.' }));
    var ctl = h('div', { cls: 'av-ctl' });
    var sel = h('select', { id: 'av-task-sel' });
    data.tasks.slice().sort(function (a, b) { return a.label < b.label ? -1 : 1; }).forEach(function (t) {
      var o = h('option', { value: t.id, text: t.label });
      if (t.id === 'chart_reasoning') o.selected = true;
      sel.appendChild(o);
    });
    ctl.appendChild(h('label', { 'for': 'av-task-sel', text: 'Task family of the request' }));
    ctl.appendChild(sel);
    r.appendChild(ctl);

    var steps = [
      'The request arrives',
      'The router scores each model for being correct',
      'Cost and latency penalties are subtracted',
      'The chosen model answers'
    ];
    var stepNow = 0;
    var stepList = h('ol', { cls: 'av-steps', role: 'group', 'aria-label': 'Routing steps' });
    var stepBtns = steps.map(function (s, i) {
      var b = h('button', { type: 'button', text: (i + 1) + '. ' + s });
      b.addEventListener('click', function () { stepNow = i; renderRoute(); });
      stepList.appendChild(h('li', null, [b]));
      return b;
    });
    r.appendChild(stepList);
    var nav = h('div', { cls: 'av-ctl', role: 'group', 'aria-label': 'Step navigation' });
    var prev = h('button', { type: 'button', text: 'Back' });
    var next = h('button', { type: 'button', text: 'Next step' });
    prev.addEventListener('click', function () { if (stepNow > 0) { stepNow--; renderRoute(); } });
    next.addEventListener('click', function () { if (stepNow < steps.length - 1) { stepNow++; renderRoute(); } });
    nav.appendChild(prev); nav.appendChild(next);
    r.appendChild(nav);
    var rLive = h('p', { cls: 'av-live', 'aria-live': 'polite' });
    r.appendChild(rLive);
    var req = h('p', { cls: 'av-req' });
    r.appendChild(req);
    r.appendChild(h('p', { cls: 'av-small', text: 'Cards run from cheapest to most expensive. Bars are ranks, not measurements.' }));
    var cardList = h('ul', { cls: 'av-cards', 'aria-label': 'Candidate models, cheapest first' });
    r.appendChild(cardList);
    var n = models.length;
    var cards = models.map(function (m) {
      var pb = h('i', { style: 'width:' + Math.round(100 * m.price_rank / n) + '%' });
      var sb = h('i', { style: 'width:' + Math.round(100 * (n + 1 - m.speed_rank) / n) + '%' });
      var badge = h('span', { cls: 'av-badge' });
      var c = h('li', { cls: 'av-card' }, [
        h('span', { cls: 'av-cname', text: m.name }),
        m.strongest ? h('span', { cls: 'av-blab', text: 'Strongest single model overall' }) : null,
        h('span', { cls: 'av-blab', text: 'relative price, pinned price table' }),
        h('span', { cls: 'av-bar', 'aria-hidden': 'true' }, [pb]),
        h('span', { cls: 'av-blab', text: 'relative speed' }),
        h('span', { cls: 'av-bar', 'aria-hidden': 'true' }, [sb]),
        badge
      ]);
      cardList.appendChild(c);
      return { m: m, el: c, badge: badge };
    });

    function renderRoute() {
      var t = data.tasks.filter(function (x) { return x.id === sel.value; })[0];
      var pick = byId[t.pick];
      req.textContent = 'Request: an image plus a ' + t.label.toLowerCase() + ' question.';
      stepBtns.forEach(function (b, i) {
        if (i === stepNow) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      });
      prev.disabled = stepNow === 0;
      next.disabled = stepNow === steps.length - 1;
      var msg;
      if (stepNow === 0) msg = 'Step 1. A ' + t.label.toLowerCase() + ' request arrives. All ' + data.counts.models + ' models could answer it; they differ widely in price and speed.';
      else if (stepNow === 1) msg = 'Step 2. The router estimates, for each model, how likely it is to answer correctly. The scores are not shown here.';
      else if (stepNow === 2) msg = 'Step 3. Cost and latency penalties are subtracted from each score at decision time, so a cheaper or faster model can overtake a stronger one.';
      else {
        msg = 'Step 4. ' + pick.name + ' answers. This is the per-task lookup’s pick for ' + t.label.toLowerCase() + ' (the learned router did not beat this lookup on the test split).';
        if (t.tier === 'dead') msg += ' This is a dead task: no model does well, whichever is picked.';
        else if (pick.price_rank < top.price_rank) msg += ' It is cheaper than ' + top.name + '.';
      }
      rLive.textContent = msg;
      cards.forEach(function (c) {
        var isPick = stepNow === 3 && c.m.id === pick.id;
        c.el.className = 'av-card' + (isPick ? ' pick' : '');
        c.badge.className = 'av-badge' + (isPick ? ' ok' : '');
        c.badge.textContent = stepNow === 0 ? 'Candidate' :
          stepNow === 1 ? 'Correctness score: hidden' :
          stepNow === 2 ? 'Score minus cost and latency penalty' :
          isPick ? 'Chosen: answers the request' : 'Not chosen';
      });
    }
    sel.addEventListener('change', renderRoute);
    renderRoute();
    root.appendChild(r);

    /* Static panels */
    var pans = h('div', { cls: 'av-panels' });
    pans.appendChild(h('section', { cls: 'av-panel', 'aria-label': 'Leakage fix' }, [
      h('h4', { text: 'The leak that was removed' }),
      h('p', { text: 'Earlier splits shared images between train and test. Splitting by image removed the leak and retired the old numbers.' }),
      h('p', null, [h('span', { cls: 'av-big', text: data.counts.shared_images }), h('span', { cls: 'av-mut', text: ' shared images found' })]),
      h('p', { cls: 'av-small', text: 'Benchmark: ' + data.counts.questions + ' questions across ' + data.counts.families + ' task families.' })
    ]));
    var sl = h('ul', { cls: 'av-st' });
    data.status.forEach(function (s) {
      var k = s[1].toLowerCase();
      var li = h('li', null, [h('span', { cls: 'av-sw ' + k, 'aria-hidden': 'true' }), h('span', { text: s[0] + ': ' }), h('strong', { text: s[1] })]);
      if (s[2]) li.appendChild(h('span', { cls: 'av-mut', text: '(' + s[2] + ')' }));
      sl.appendChild(li);
    });
    pans.appendChild(h('section', { cls: 'av-panel', 'aria-label': 'Build status' }, [h('h4', { text: 'What exists' }), sl]));
    root.appendChild(pans);
  }

  if (window.RSVis.register) window.RSVis.register(ID, mount);
  window.RSVis.mods = window.RSVis.mods || {};
  window.RSVis.mods[ID] = mount;
})();
