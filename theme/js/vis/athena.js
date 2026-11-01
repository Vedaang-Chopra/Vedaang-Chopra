window.RSVis = window.RSVis || { mods: {} };
(function () {
  var ID = 'athena';
  var P = '.rs-vis[data-vis="athena"] ';
  var CSS = [
    P + '{font-family:var(--font-sans);color:var(--text-color)}',
    P + 'button{font:inherit;color:var(--text-color);cursor:pointer;text-align:left}',
    P + 'button:focus-visible,' + P + 'summary:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    P + '.at-t{font-size:1.05rem;font-weight:600;margin:0 0 .3rem}',
    P + '.at-p{font-size:.95rem;margin:0 0 .7rem;line-height:1.45}',
    P + '.at-m{font-size:.88rem;color:var(--mut);margin:.4rem 0 0}',
    P + '.at-nav{display:flex;align-items:center;gap:.5rem;margin:0 0 .8rem;flex-wrap:wrap}',
    P + '.at-nb{border:1px solid var(--border-color);background:var(--btn-bg);border-radius:6px;padding:.3rem .8rem;font-size:.9rem}',
    P + '.at-nb[disabled]{opacity:.5;cursor:default}',
    P + '.at-dots{display:flex;gap:.3rem}',
    P + '.at-dot{width:1.6rem;height:1.6rem;border-radius:50%;border:1px solid var(--mut);background:var(--bg-color);padding:0;text-align:center;font-size:.8rem}',
    P + '.at-dot[aria-current="step"]{background:var(--text-color);color:var(--bg-color);font-weight:700}',
    P + '.at-cnt{font-size:.88rem;color:var(--mut)}',
    P + '.at-row{list-style:none;margin:0 0 .6rem;padding:0;display:flex;flex-wrap:wrap;gap:.4rem}',
    P + '.at-box{border:2px solid var(--text-color);border-radius:6px;background:var(--surface);padding:.35rem .6rem;font-size:.9rem;line-height:1.3}',
    P + 'button.at-box[aria-pressed="true"]{outline:3px solid var(--focus);outline-offset:1px}',
    P + '.at-tool{border:1px solid var(--mut);border-radius:999px;background:var(--bg-color);padding:.15rem .6rem;font-size:.85rem}',
    P + 'button.at-tool[aria-pressed="true"]{border:2px solid var(--text-color);font-weight:600}',
    P + '.at-st{display:inline-flex;align-items:center;gap:.3rem;font-size:.85rem}',
    P + '.at-mk{display:inline-block;width:.7rem;height:.7rem;box-sizing:border-box;flex:none}',
    P + '.s-Built{background:var(--ok);border:2px solid var(--ok)}',
    P + '.s-Partial{border:2px solid var(--warn);background:linear-gradient(90deg,var(--warn) 50%,transparent 50%)}',
    P + '.s-Proposed{border:2px dashed var(--mut);background:transparent}',
    P + '.at-plan{border:3px double var(--text-color);border-radius:6px;padding:.5rem .7rem;margin:0 0 .6rem;background:var(--bg-color)}',
    P + '.at-flow{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin:0 0 .6rem;font-size:.9rem}',
    P + '.at-arr::before{content:"\\2193";color:var(--mut)}',
    P + '.at-dt{border:1px solid var(--border-color);border-radius:6px;padding:.4rem .6rem;margin:0 0 .6rem}',
    P + '.at-dt summary{cursor:pointer;font-size:.92rem}',
    P + '.at-in{list-style:none;margin:.5rem 0 0;padding:0;display:flex;flex-direction:column;gap:.4rem}',
    P + '.at-in li{border-left:3px solid var(--text-color);padding:.15rem .6rem}',
    P + '.at-per{font-size:.82rem;color:var(--mut);margin-left:.4rem}',
    P + '.at-chips{display:flex;flex-wrap:wrap;gap:.3rem;margin:.3rem 0 0}',
    P + '.at-ent{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .6rem}',
    P + 'button.at-ent-b{border:1px solid var(--border-color);background:var(--bg-color);border-radius:999px;padding:.25rem .8rem;font-size:.9rem}',
    P + 'button.at-ent-b[aria-pressed="true"]{border:2px solid var(--text-color);font-weight:600;background:var(--surface)}',
    P + '.at-skip{border-color:var(--border-color);background:transparent;color:var(--mut)}',
    P + '.at-skip .at-nm{text-decoration:line-through}',
    P + '.at-sk{font-size:.82rem;font-weight:600;margin-left:.3rem}',
    P + '.at-today{display:grid;gap:.5rem}',
    P + '.at-card{border:1px solid var(--border-color);border-radius:6px;padding:.5rem .7rem;background:var(--surface)}',
    P + '.at-card.c-Proposed{border:2px dashed var(--mut);background:transparent}',
    P + '.at-card.c-Partial{border:2px solid var(--warn)}',
    P + '.at-card.c-Built{border:2px solid var(--ok)}',
    P + '.at-detail{min-height:3.4rem;border-top:1px solid var(--border-color);padding-top:.5rem;font-size:.92rem}'
  ].join('\n');

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function status(s) {
    var w = el('span', 'at-st');
    w.appendChild(el('span', 'at-mk s-' + s));
    w.appendChild(el('span', null, s));
    return w;
  }

  RSVis.mods[ID] = function mount(root, data) {
    if (!document.getElementById('rs-vis-athena-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-athena-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    root.textContent = '';
    var steps = data.steps;
    var cur = 0;
    var selNode = 0;
    var selEntry = 0;
    var selInner = null;

    var wrap = el('div');
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', data.title);
    var nav = el('div', 'at-nav');
    var back = el('button', 'at-nb', 'Back');
    var next = el('button', 'at-nb', 'Next');
    var dots = el('div', 'at-dots');
    var cnt = el('span', 'at-cnt');
    var dotBtns = steps.map(function (s, i) {
      var b = el('button', 'at-dot', String(i + 1));
      b.type = 'button';
      b.setAttribute('aria-label', 'Go to step ' + (i + 1) + ': ' + s.title);
      b.addEventListener('click', function () { go(i); });
      dots.appendChild(b);
      return b;
    });
    back.type = 'button';
    next.type = 'button';
    back.addEventListener('click', function () { go(cur - 1); });
    next.addEventListener('click', function () { go(cur + 1); });
    nav.appendChild(back);
    nav.appendChild(dots);
    nav.appendChild(next);
    nav.appendChild(cnt);
    var stage = el('div');
    stage.setAttribute('aria-live', 'polite');
    wrap.appendChild(nav);
    wrap.appendChild(stage);
    wrap.appendChild(el('p', 'at-m', data.source_note));
    root.appendChild(wrap);

    function go(i) {
      if (i < 0 || i >= steps.length) return;
      cur = i;
      render();
    }

    function head(s) {
      stage.appendChild(el('h4', 'at-t', s.title));
      if (s.body) stage.appendChild(el('p', 'at-p', s.body));
    }

    function stepIdea(s) {
      head(s);
      var flow = el('div', 'at-flow');
      flow.appendChild(el('span', 'at-box', 'Your rough idea'));
      flow.appendChild(el('span', 'at-arr'));
      flow.appendChild(el('span', 'at-box', 'One language model reads it'));
      stage.appendChild(flow);
      var plan = el('div', 'at-plan');
      plan.appendChild(el('strong', null, s.plan_label));
      plan.appendChild(el('div', 'at-m', s.plan_note));
      stage.appendChild(plan);
    }

    function stepGraph(s) {
      head(s);
      var row = el('ol', 'at-row');
      var det = el('div', 'at-detail');
      var btns = [];
      function show(i) {
        selNode = i;
        var n = data.nodes[i];
        btns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
        det.textContent = '';
        var h = el('div');
        h.appendChild(el('strong', null, n.label + ' '));
        h.appendChild(status(n.status));
        det.appendChild(h);
        det.appendChild(el('div', null, n.what));
        det.appendChild(el('div', 'at-m', 'Produces: ' + n.makes));
      }
      data.nodes.forEach(function (n, i) {
        var li = el('li');
        var b = el('button', 'at-box', (i + 1) + '. ' + n.label);
        b.type = 'button';
        b.addEventListener('click', function () { show(i); });
        btns.push(b);
        li.appendChild(b);
        row.appendChild(li);
      });
      stage.appendChild(row);
      stage.appendChild(det);
      show(selNode);
    }

    function toolChips(tools) {
      var chips = el('div', 'at-chips');
      var out = el('div', 'at-m');
      out.setAttribute('aria-live', 'polite');
      var bs = [];
      tools.forEach(function (t, i) {
        var b = el('button', 'at-tool', t.label);
        b.type = 'button';
        b.setAttribute('aria-pressed', 'false');
        b.addEventListener('click', function () {
          bs.forEach(function (x, j) { x.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
          out.textContent = t.label + ': ' + t.what;
        });
        bs.push(b);
        chips.appendChild(b);
      });
      var f = document.createDocumentFragment();
      f.appendChild(chips);
      f.appendChild(out);
      return f;
    }

    function stepInner(s) {
      head(s);
      var d = document.createElement('details');
      d.className = 'at-dt';
      var sm = el('summary', null, s.details_label);
      d.appendChild(sm);
      var ol = el('ol', 'at-in');
      data.inner.forEach(function (it) {
        var li = el('li');
        var h = el('div');
        h.appendChild(el('strong', null, it.label));
        h.appendChild(el('span', 'at-per', it.per));
        h.appendChild(document.createTextNode(' '));
        h.appendChild(status(it.status));
        li.appendChild(h);
        li.appendChild(el('div', null, it.what));
        if (it.note) li.appendChild(el('div', 'at-m', it.note));
        if (it.tools) {
          li.appendChild(el('div', 'at-m', it.label === 'Research the scene' ? s.research_label : s.image_label));
          li.appendChild(toolChips(it.tools));
        }
        ol.appendChild(li);
      });
      d.appendChild(ol);
      stage.appendChild(d);
    }

    function stepEntry(s) {
      head(s);
      stage.appendChild(el('p', 'at-m', s.pick));
      var ent = el('div', 'at-ent');
      ent.setAttribute('role', 'group');
      ent.setAttribute('aria-label', s.pick);
      var row = el('ol', 'at-row');
      var note = el('div', 'at-detail');
      var eb = [];
      function show(i) {
        selEntry = i;
        var e = data.entries[i];
        eb.forEach(function (b, j) { b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
        row.textContent = '';
        data.nodes.forEach(function (n, k) {
          var skipped = e.skip.indexOf(n.id) >= 0;
          var partly = n.id === 'each' && e.inner_skip.length > 0;
          var li = el('li', 'at-box' + (skipped ? ' at-skip' : ''));
          li.appendChild(el('span', 'at-nm', (k + 1) + '. ' + n.label));
          if (skipped) li.appendChild(el('span', 'at-sk', 'skipped'));
          if (partly) li.appendChild(el('span', 'at-sk', 'skips ' + e.inner_skip.join(' and ').toLowerCase()));
          row.appendChild(li);
        });
        note.textContent = e.note || '';
      }
      data.entries.forEach(function (e, i) {
        var b = el('button', 'at-ent-b', e.label);
        b.type = 'button';
        b.addEventListener('click', function () { show(i); });
        eb.push(b);
        ent.appendChild(b);
      });
      stage.appendChild(ent);
      stage.appendChild(row);
      stage.appendChild(el('p', 'at-p', s.caption));
      stage.appendChild(note);
      show(selEntry);
    }

    function stepToday(s) {
      head(s);
      var g = el('div', 'at-today');
      data.today.forEach(function (t) {
        var c = el('div', 'at-card c-' + t.status);
        var h = el('div');
        h.appendChild(el('strong', null, t.label + ' '));
        h.appendChild(status(t.status));
        c.appendChild(h);
        c.appendChild(el('div', null, t.what));
        g.appendChild(c);
      });
      stage.appendChild(g);
    }

    var fns = [stepIdea, stepGraph, stepInner, stepEntry, stepToday];
    function render() {
      stage.textContent = '';
      fns[cur](steps[cur]);
      cnt.textContent = 'Step ' + (cur + 1) + ' of ' + steps.length;
      back.disabled = cur === 0;
      next.disabled = cur === steps.length - 1;
      dotBtns.forEach(function (b, i) {
        if (i === cur) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      });
    }
    render();
  };
})();
