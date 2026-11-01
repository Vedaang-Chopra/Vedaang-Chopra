window.RSVis = window.RSVis || { mods: {} };
(function () {
  'use strict';
  var ID = 'artemis';
  var P = '.rs-vis[data-vis="artemis"] ';
  var CSS = [
    P + '.aw{container-type:inline-size;font-family:var(--font-sans);color:var(--text-color);font-size:15px;line-height:1.5;margin:1rem 0}',
    P + '.aw *{box-sizing:border-box}',
    P + 'button{font:inherit;font-size:14px;color:var(--text-color);background:var(--surface);border:1.5px solid var(--border-color);border-radius:6px;padding:.45rem .75rem;cursor:pointer;min-height:40px}',
    P + 'button:hover:not([disabled]){border-color:var(--text-color)}',
    P + 'button[disabled]{opacity:.55;cursor:default}',
    P + 'button:focus-visible,' + P + 'summary:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    P + '.aw-nav{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem;margin:0 0 .6rem}',
    P + '.aw-dots{display:flex;gap:.3rem;margin:0;padding:0;list-style:none}',
    P + '.aw-dot{min-width:36px;padding:.3rem .4rem;text-align:center}',
    P + '.aw-dot[aria-current="step"]{border-color:var(--text-color);background:color-mix(in srgb,var(--bg-color) 80%,var(--text-color));font-weight:700}',
    P + '.aw-count{font-size:13px;color:var(--mut);margin:0}',
    P + '.aw-title{font-size:1.1rem;font-weight:700;margin:.1rem 0 .35rem;line-height:1.35}',
    P + '.aw-text{margin:0 0 .7rem;max-width:62ch}',
    P + '.aw-body.fade{animation:awfade .25s ease-out}',
    '@keyframes awfade{from{opacity:0}to{opacity:1}}',
    P + '.aw-mut{color:var(--mut);font-size:13px;margin:.4rem 0}',
    P + '.aw-st{display:inline-block;font-size:12px;line-height:1.2;padding:.15rem .45rem;border-radius:4px;border:1.5px solid var(--text-color);font-weight:600;color:var(--text-color);vertical-align:middle;margin-right:.35rem}',
    P + '.aw-st.built{background:color-mix(in srgb,var(--ok) 13%,var(--bg-color));border-color:var(--ok)}',
    P + '.aw-st.part{background:color-mix(in srgb,var(--warn) 13%,var(--bg-color));border-color:var(--warn)}',
    P + '.aw-st.tried{border-color:var(--bad);background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 13%,var(--bg-color)) 0 4px,var(--bg-color) 4px 8px)}',
    P + '.aw-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr));gap:.5rem;margin:0;padding:0;list-style:none}',
    P + '.aw-card{border:1.5px solid var(--border-color);border-radius:6px;padding:.55rem .65rem;background:var(--surface);display:grid;gap:.25rem}',
    P + '.aw-card.pick{border:3px solid var(--ok)}',
        P + '.aw-cname{font-weight:700;font-size:14px}',
    P + '.aw-lab{font-size:13px;color:var(--mut);display:flex;justify-content:space-between;gap:.4rem}',
    P + '.aw-bar{height:9px;background:color-mix(in srgb,var(--base-bar) 25%,transparent);border-radius:2px;overflow:hidden}',
    P + '.aw-bar i{display:block;height:100%;background:var(--plain-bar)}',
    P + '.aw-tag{font-size:13px;font-weight:700;color:var(--ok)}',
    P + '.aw-tag.str{color:var(--text-color);font-weight:600}',
    P + '.aw-axis{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);margin:.35rem 0 0}',
    P + '.aw-pick{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .6rem}',
    P + '.aw-pick button[aria-pressed="true"]{border-color:var(--text-color);background:color-mix(in srgb,var(--bg-color) 80%,var(--text-color));font-weight:700}',
    P + '.aw-live{font-size:14px;border-left:4px solid var(--text-color);background:var(--surface);padding:.5rem .75rem;margin:.6rem 0;min-height:2.8em}',
    P + '.aw-none{border:2px dashed var(--mut);border-radius:6px;padding:.4rem .6rem;font-weight:600;margin:.5rem 0 0;font-size:14px}',
    P + 'details{margin:.7rem 0 0;font-size:14px}',
    P + 'summary{cursor:pointer;font-weight:600;padding:.3rem 0;min-height:32px}',
    P + '.aw-tier{font-weight:700;margin:.7rem 0 .2rem;font-size:14px}',
    P + '.aw-list{margin:0;padding:0;list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:.15rem .9rem}',
    P + '.aw-list li{display:flex;justify-content:space-between;gap:.5rem;border-bottom:1px solid var(--border-color);padding:.2rem 0}',
    P + '.aw-list span:last-child{color:var(--mut);text-align:right}',
    P + '.aw-req{border-left:4px solid var(--plain-bar);padding:.3rem .7rem;margin:0 0 .3rem;font-size:14px;font-style:italic}',
    P + '.aw-arrow{font-size:13px;color:var(--mut);margin:.1rem 0 .4rem}',
    P + '.aw-rule{display:grid;grid-template-columns:minmax(96px,1.3fr) repeat(3,minmax(54px,1fr));gap:.3rem .5rem;align-items:center;font-size:13px}',
    P + '.aw-rule .h{color:var(--mut);font-weight:600;line-height:1.25}',
    P + '.aw-rule .n{font-weight:600;line-height:1.25}',
    P + '.aw-rule .win{color:var(--ok);font-weight:700}',
    P + '.aw-rule .aw-bar{height:14px}',
    P + '.aw-rule .q i{background:var(--plain-bar)}',
    P + '.aw-rule .c i{background:repeating-linear-gradient(90deg,var(--mut) 0 3px,transparent 3px 6px)}',
    P + '.aw-rule .t i{background:var(--base-bar)}',
    P + '.aw-rule .t.win i{background:var(--ok)}',
    P + '.aw-ill{display:inline-block;font-size:13px;border:1.5px dashed var(--mut);border-radius:4px;padding:.1rem .45rem;margin:.6rem 0 0;color:var(--text-color)}',
    P + '.aw-lad{display:grid;gap:.55rem;margin:.2rem 0}',
    P + '.aw-lrow{display:grid;grid-template-columns:minmax(120px,1fr) 3fr;gap:.6rem;align-items:center}',
    P + '.aw-lname{font-weight:600;font-size:14px;line-height:1.25}',
    P + '.aw-lname small{display:block;font-weight:400;font-size:13px;color:var(--mut)}',
    P + '.aw-ltrack{display:flex;align-items:center;gap:.45rem}',
    P + '.aw-lbar{height:22px;border-radius:3px;flex:none}',
    P + '.aw-lbar.base{background:var(--base-bar)}',
    P + '.aw-lbar.ok{background:var(--ok)}',
    P + '.aw-lbar.tried{border:2px solid var(--bad);background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 45%,var(--bg-color)) 0 4px,var(--bg-color) 4px 8px)}',
    P + '.aw-lbar.bound{border:2px solid var(--plain-bar);background:color-mix(in srgb,var(--plain-bar) 15%,var(--bg-color))}',
    P + '.aw-lval{font-family:var(--font-mono);font-size:14px;font-weight:600}',
    P + '.aw-key{font-weight:700;margin:.6rem 0 0}',
    P + '.aw-ba{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:.7rem}',
    P + '.aw-panel{border:1.5px solid var(--border-color);border-radius:6px;padding:.6rem .7rem;background:var(--surface)}',
    P + '.aw-panel h5{margin:0 0 .45rem;font-size:14px;font-weight:700}',
    P + '.aw-split{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}',
    P + '.aw-box{border:1.5px solid var(--text-color);border-radius:4px;padding:.35rem}',
    P + '.aw-box b{display:block;font-size:13px;margin:0 0 .3rem}',
    P + '.aw-imgs{display:flex;flex-wrap:wrap;gap:4px}',
    P + '.aw-img{width:18px;height:18px;border-radius:3px;background:var(--base-bar)}',
    P + '.aw-img.shared{background:repeating-linear-gradient(45deg,var(--bad) 0 3px,var(--bg-color) 3px 6px);border:2px solid var(--bad);border-radius:50%}',
    P + '.aw-leg{display:flex;align-items:center;gap:.4rem;font-size:13px;margin:.45rem 0 0}',
    '@container (max-width:520px){' + P + '.aw-rule{grid-template-columns:repeat(3,1fr);row-gap:.2rem}' + P + '.aw-rule .h:first-child{display:none}' + P + '.aw-rule .n{grid-column:1/-1;margin-top:.35rem}' + P + '.aw-lrow{grid-template-columns:1fr;gap:.2rem}}',
    P + '.aw-status{margin:.3rem 0 0;padding:0;list-style:none;display:grid;gap:.25rem}'
  ].join('\n');

  function h(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function stBadge(s) {
    var c = s === 'Built' ? 'built' : s === 'Partial' ? 'part' : 'tried';
    return h('span', 'aw-st ' + c, s);
  }
  function bar(frac, cls) {
    var b = h('div', 'aw-bar' + (cls ? ' ' + cls : ''));
    b.setAttribute('aria-hidden', 'true');
    var i = h('i');
    i.style.width = Math.max(4, Math.round(frac * 100)) + '%';
    b.appendChild(i);
    return b;
  }

  var PRICE = ['lowest', 'low', 'middle', 'high', 'highest'];
  var SPEED = ['fastest', 'fast', 'middle', 'slow', 'slowest'];

  RSVis.mods[ID] = function mount(el, data, ctx) {
    ctx = ctx || {};
    if (!document.getElementById('rs-vis-' + ID + '-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-' + ID + '-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    var models = data.models.slice().sort(function (a, b) { return a.price_rank - b.price_rank; });
    var byId = {};
    models.forEach(function (m) { byId[m.id] = m; });
    var tasks = {};
    data.tasks.forEach(function (t) { tasks[t.id] = t; });
    var strongest = models.filter(function (m) { return m.strongest; })[0];
    var C = data.counts;

    function modelCards(pickId) {
      var ul = h('ul', 'aw-cards');
      models.forEach(function (m) {
        var li = h('li', 'aw-card' + (m.id === pickId ? ' pick' : ''));
        li.appendChild(h('span', 'aw-cname', m.name));
        var pl = h('span', 'aw-lab');
        pl.appendChild(h('span', null, 'Price'));
        pl.appendChild(h('span', null, PRICE[m.price_rank - 1]));
        li.appendChild(pl);
        li.appendChild(bar(m.price_rank / 5));
        var sl = h('span', 'aw-lab');
        sl.appendChild(h('span', null, 'Speed'));
        sl.appendChild(h('span', null, SPEED[m.speed_rank - 1]));
        li.appendChild(sl);
        li.appendChild(bar((6 - m.speed_rank) / 5));
        if (m.id === pickId) li.appendChild(h('span', 'aw-tag', 'Lookup pick'));
        else if (m.strongest) li.appendChild(h('span', 'aw-tag str', 'Strongest overall'));
        ul.appendChild(li);
      });
      return ul;
    }

    function taskSentence(t) {
      var m = byId[t.pick];
      if (t.tier === 'dead') return t.label + ': no model does well here. Routing cannot fix a task that no model can do.';
      var s;
      if (m.strongest) s = 'For ' + t.label.toLowerCase() + ', the strongest and priciest model really is the best pick.';
      else s = 'For ' + t.label.toLowerCase() + ', the lookup picks ' + m.name + ', which is cheaper than the strongest model.';
      if (t.tier === 'weak') s += ' Even the best model is only middling here.';
      return s;
    }

    // Step builders: each returns a node holding one picture.
    function step1() {
      var w = h('div');
      w.appendChild(h('p', 'aw-text', 'Several vision-language models (AI that answers questions about images) can take the same request. They differ widely in price and speed. Always calling the biggest one wastes money.'));
      var cards = modelCards(null);
      cards.setAttribute('aria-label', 'Five models ordered from cheapest to most expensive');
      w.appendChild(cards);
      var ax = h('div', 'aw-axis');
      ax.appendChild(h('span', null, 'Cheapest'));
      ax.appendChild(h('span', null, 'Most expensive'));
      w.appendChild(ax);
      w.appendChild(h('p', 'aw-mut', 'Bars show order only, not measured amounts. Longer speed bar means faster.'));
      return w;
    }

    function step2() {
      var w = h('div');
      w.appendChild(h('p', 'aw-text', 'I built a benchmark of ' + C.questions + ' image questions across ' + C.families + ' kinds of task. A simple lookup table remembers which model did best at each kind. Pick a task.'));
      var grp = h('div', 'aw-pick');
      grp.setAttribute('role', 'group');
      grp.setAttribute('aria-label', 'Choose a kind of task');
      var live = h('p', 'aw-live');
      live.setAttribute('aria-live', 'polite');
      var holder = h('div');
      var btns = [];
      function show(id) {
        var t = tasks[id];
        btns.forEach(function (b) { b.setAttribute('aria-pressed', String(b._id === id)); });
        live.textContent = taskSentence(t);
        holder.textContent = '';
        var dead = t.tier === 'dead';
        holder.appendChild(modelCards(dead ? null : t.pick));
        if (dead) holder.appendChild(h('p', 'aw-none', 'No model does well here'));
      }
      data.featured.forEach(function (id) {
        var b = h('button', null, tasks[id].label);
        b.type = 'button';
        b._id = id;
        b.addEventListener('click', function () { show(id); });
        btns.push(b);
        grp.appendChild(b);
      });
      w.appendChild(grp);
      w.appendChild(live);
      w.appendChild(holder);
      var bs = h('p', 'aw-mut');
      bs.appendChild(stBadge('Built'));
      bs.appendChild(document.createTextNode('Benchmark, built offline.'));
      w.appendChild(bs);
      var d = h('details');
      d.appendChild(h('summary', null, 'See all ' + C.families + ' task families'));
      d.appendChild(h('p', 'aw-mut', 'Each row shows the model the lookup picks. Grouped by how well the best model does.'));
      data.tiers.forEach(function (tier) {
        d.appendChild(h('p', 'aw-tier', tier.name + ': ' + tier.desc));
        var ul = h('ul', 'aw-list');
        data.tasks.filter(function (t) { return t.tier === tier.id; }).forEach(function (t) {
          var li = h('li');
          li.appendChild(h('span', null, t.label));
          li.appendChild(h('span', null, t.tier === 'dead' ? 'no model does well' : byId[t.pick].name + (byId[t.pick].strongest ? ' (strongest)' : ' (cheaper)')));
          ul.appendChild(li);
        });
        d.appendChild(ul);
      });
      w.appendChild(d);
      show(data.featured[0]);
      return w;
    }

    // Hand-made shapes for the illustration only; never shown as numbers.
    var ILL = { deepseek_ocr: [0.15, 0.06], gemma_3_27b: [0.18, 0.16], qwen2_5_vl_3b: [0.6, 0.18], qwen2_5_vl_7b: [0.78, 0.25], qwen3_vl_8b_thinking: [0.66, 0.62] };
    function step3() {
      var w = h('div');
      w.appendChild(h('p', 'aw-text', 'A router guesses how likely each model is to answer well. It subtracts a penalty for price and delay, then sends the request to the best trade.'));
      w.appendChild(h('div', 'aw-req', 'Request: "How many apples are in this photo?"'));
      w.appendChild(h('p', 'aw-arrow', 'The router scores all five models:'));
      var g = h('div', 'aw-rule');
      g.setAttribute('role', 'img');
      var best = null, bestV = -1;
      models.forEach(function (m) {
        var v = ILL[m.id] ? ILL[m.id][0] - ILL[m.id][1] : 0;
        if (v > bestV) { bestV = v; best = m; }
      });
      g.setAttribute('aria-label', 'Illustration: each model gets a chance-of-good-answer score minus a price and delay penalty. ' + best.name + ' ends highest and is picked.');
      ['Model', 'Chance of a good answer', 'Minus price and delay', 'What is left'].forEach(function (s) { g.appendChild(h('span', 'h', s)); });
      models.forEach(function (m) {
        var q = ILL[m.id] || [0, 0];
        var win = m === best;
        g.appendChild(h('span', 'n' + (win ? ' win' : ''), m.name + (win ? ' (picked)' : '')));
        g.appendChild(bar(q[0], 'q'));
        g.appendChild(bar(q[1], 'c'));
        g.appendChild(bar(Math.max(0.02, q[0] - q[1]), 't' + (win ? ' win' : '')));
      });
      w.appendChild(g);
      w.appendChild(h('p', 'aw-ill', 'Illustration of the decision rule, not a recording'));
      var d = h('details');
      d.appendChild(h('summary', null, 'What exists today'));
      var ul = h('ul', 'aw-status');
      data.status.forEach(function (s) {
        var li = h('li');
        li.appendChild(stBadge(s[1]));
        li.appendChild(document.createTextNode(s[0] + (s[2] ? ': ' + s[2] : '')));
        ul.appendChild(li);
      });
      d.appendChild(ul);
      d.appendChild(h('p', 'aw-mut', 'A cascade tries a cheap model first and moves up only when the answer looks unsure.'));
      w.appendChild(d);
      return w;
    }

    function step4() {
      var w = h('div');
      w.appendChild(h('p', 'aw-text', 'I compared four ways of choosing a model, by how many test questions they got right. Longer is better.'));
      var lad = h('div', 'aw-lad');
      lad.setAttribute('role', 'list');
      var max = 0.7;
      data.ladder.forEach(function (r) {
        var row = h('div', 'aw-lrow');
        row.setAttribute('role', 'listitem');
        var nm = h('div', 'aw-lname', r.label);
        nm.appendChild(h('small', null, r.gloss));
        row.appendChild(nm);
        var tr = h('div', 'aw-ltrack');
        var b = h('span', 'aw-lbar ' + r.kind);
        b.style.width = Math.round(r.v / max * 78) + '%';
        b.setAttribute('aria-hidden', 'true');
        tr.appendChild(b);
        tr.appendChild(h('span', 'aw-lval', r.value));
        row.appendChild(tr);
        lad.appendChild(row);
      });
      w.appendChild(lad);
      w.appendChild(h('p', 'aw-key', 'My learned router did not beat a simple lookup table.'));
      w.appendChild(h('p', 'aw-mut', 'Accuracy on ' + C.test_questions + ' test questions. Early measurement. Hatched bar: did not beat the lookup.'));
      var d = h('details');
      d.appendChild(h('summary', null, 'What this does and does not show'));
      d.appendChild(h('p', null, 'All of this ran offline on stored answers. The oracle is not a real method: it peeks at the right answers, so it only marks the ceiling. The router and serving code have not been tested on live models.'));
      w.appendChild(d);
      return w;
    }

    function imgs(n, shared) {
      var d = h('div', 'aw-imgs');
      for (var i = 0; i < n; i++) d.appendChild(h('span', 'aw-img' + (shared.indexOf(i) >= 0 ? ' shared' : '')));
      return d;
    }
    function panel(title, trainShared, testShared, note) {
      var p = h('div', 'aw-panel');
      p.appendChild(h('h5', null, title));
      var s = h('div', 'aw-split');
      var a = h('div', 'aw-box'); a.appendChild(h('b', null, 'Training data')); a.appendChild(imgs(6, trainShared));
      var b = h('div', 'aw-box'); b.appendChild(h('b', null, 'Test data')); b.appendChild(imgs(4, testShared));
      s.appendChild(a); s.appendChild(b);
      s.setAttribute('aria-hidden', 'true');
      p.appendChild(s);
      p.appendChild(h('p', 'aw-mut', note));
      return p;
    }
    function step5() {
      var w = h('div');
      w.appendChild(h('p', 'aw-text', 'Earlier splits shared ' + C.shared_images + ' images between training and test data. Models were partly tested on pictures already seen. I regrouped the split by image and retired the earlier numbers.'));
      var ba = h('div', 'aw-ba');
      ba.appendChild(panel('Before', [1, 4], [0, 2], 'Some images sit on both sides, so test scores look better than they are.'));
      ba.appendChild(panel('After: grouped by image', [], [], 'Every image lives on one side only.'));
      w.appendChild(ba);
      var lg = h('p', 'aw-leg');
      lg.appendChild(h('span', 'aw-img shared'));
      lg.appendChild(document.createTextNode('Same image in both training and test'));
      w.appendChild(lg);
      var bs = h('p', 'aw-mut');
      bs.appendChild(stBadge('Built'));
      bs.appendChild(document.createTextNode('Image-grouped splits.'));
      w.appendChild(bs);
      return w;
    }

    var STEPS = [
      ['Five models, very different price and speed', step1],
      ['No single model is best at everything', step2],
      ['A router sends each request to a cheaper model', step3],
      ['Did it work? Not yet', step4],
      ['A leak I found and fixed', step5]
    ];

    el.textContent = '';
    var root = h('div', 'aw');
    root.setAttribute('role', 'region');
    root.setAttribute('aria-label', 'ARTEMIS walkthrough: routing image questions to cheaper models');
    var nav = h('div', 'aw-nav');
    nav.setAttribute('role', 'group');
    nav.setAttribute('aria-label', 'Walkthrough steps');
    var back = h('button', null, 'Back'); back.type = 'button';
    var next = h('button', null, 'Next'); next.type = 'button';
    var dots = h('ol', 'aw-dots');
    var dotBtns = STEPS.map(function (s, i) {
      var li = h('li');
      var b = h('button', 'aw-dot', String(i + 1));
      b.type = 'button';
      b.setAttribute('aria-label', 'Step ' + (i + 1) + ': ' + s[0]);
      b.addEventListener('click', function () { go(i); });
      li.appendChild(b);
      dots.appendChild(li);
      return b;
    });
    nav.appendChild(back); nav.appendChild(dots); nav.appendChild(next);
    var head = h('div');
    head.setAttribute('aria-live', 'polite');
    var count = h('p', 'aw-count');
    var title = h('p', 'aw-title');
    head.appendChild(count); head.appendChild(title);
    var body = h('div', 'aw-body');
    root.appendChild(nav); root.appendChild(head); root.appendChild(body);
    el.appendChild(root);

    var cur = 0;
    function go(i) {
      cur = Math.max(0, Math.min(STEPS.length - 1, i));
      count.textContent = 'Step ' + (cur + 1) + ' of ' + STEPS.length;
      title.textContent = STEPS[cur][0];
      dotBtns.forEach(function (b, j) {
        if (j === cur) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      });
      var f = document.activeElement;
      back.disabled = cur === 0;
      next.disabled = cur === STEPS.length - 1;
      if (f === next && next.disabled) back.focus();
      if (f === back && back.disabled) next.focus();
      body.textContent = '';
      body.classList.remove('fade');
      body.appendChild(STEPS[cur][1]());
      if (!ctx.reduceMotion) { void body.offsetWidth; body.classList.add('fade'); }
    }
    back.addEventListener('click', function () { go(cur - 1); });
    next.addEventListener('click', function () { go(cur + 1); });
    root.addEventListener('keydown', function (e) {
      if (e.target && e.target.className === 'aw-dot') {
        if (e.key === 'ArrowRight') { go(cur + 1); dotBtns[cur].focus(); e.preventDefault(); }
        if (e.key === 'ArrowLeft') { go(cur - 1); dotBtns[cur].focus(); e.preventDefault(); }
      }
    });
    go(0);
  };
})();
