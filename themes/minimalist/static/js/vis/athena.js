window.RSVis = window.RSVis || { mods: {} };
(function () {
  var ID = 'athena';
  var P = '.rs-vis[data-vis="athena"] ';
  var CSS = [
    P + '{container-type:inline-size;font-family:var(--font-sans);color:var(--text-color)}',
    P + '.at-h{font-size:1rem;margin:0 0 .2rem}',
    P + '.at-p{color:var(--mut);font-size:.88rem;margin:0 0 .6rem}',
    P + '.at-entry{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:0 0 .3rem}',
    P + '.at-entry-t{font-size:.88rem;color:var(--mut);margin-right:.2rem}',
    P + 'button{font:inherit;color:var(--text-color);cursor:pointer;text-align:left}',
    P + 'button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    P + '.at-eb{border:1px solid var(--border-color);background:var(--bg-color);border-radius:999px;padding:.25rem .75rem;font-size:.88rem}',
    P + '.at-eb[aria-pressed="true"]{border:2px solid var(--text-color);font-weight:600;background:var(--surface)}',
    P + '.at-cap{font-size:.85rem;color:var(--mut);margin:0 0 .7rem}',
    P + '.at-flow{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;align-items:flex-start;gap:.5rem 0}',
    P + '.at-flow>li{display:flex;align-items:flex-start}',
    P + '.at-ar{margin-top:.9rem;color:var(--mut);padding:0 .3rem;font-size:1rem}',
    P + '.at-ar::before{content:"\\2192"}',
    P + '.at-n{display:flex;flex-direction:column;gap:.15rem;min-width:6rem;max-width:9.5rem;padding:.4rem .5rem;background:var(--surface);font-size:.85rem;line-height:1.25}',
    P + '.at-n strong{font-size:.88rem}',
    P + '.at-code{font-family:var(--font-mono);font-size:.82rem;color:var(--mut);overflow-wrap:anywhere}',
    P + '.at-gsub{font-size:.82rem;color:var(--mut)}',
    P + '.k-graph{border:2px solid var(--text-color);border-radius:3px}',
    P + '.k-loop{border:1px solid var(--mut);border-radius:14px;background:var(--bg-color)}',
    P + '.k-gate{border:3px double var(--text-color);border-radius:3px;background:var(--bg-color)}',
    P + '.k-proposed{border:2px dashed var(--mut);border-radius:3px;background:transparent}',
    P + '.at-dia{display:inline-block;width:.7rem;height:.7rem;border:2px solid var(--text-color);transform:rotate(45deg);margin:0 .35rem 0 .1rem;vertical-align:-.05rem}',
    P + '.at-per{font-size:.82rem;font-weight:600;border:1px solid var(--mut);border-radius:3px;padding:0 .3rem;align-self:flex-start;color:var(--text-color)}',
    P + '.at-st{font-size:.82rem;display:inline-flex;align-items:center;gap:.3rem}',
    P + '.at-mk{display:inline-block;width:.7rem;height:.7rem;box-sizing:border-box;flex:none}',
    P + '.s-built{background:var(--ok);border:2px solid var(--ok)}',
    P + '.s-partial{border:2px solid var(--warn);background:linear-gradient(90deg,var(--warn) 50%,transparent 50%)}',
    P + '.s-proposed{border:2px dashed var(--mut);background:transparent}',
    P + '.at-skip{display:none;font-size:.82rem;font-weight:600;color:var(--mut);border:1px solid var(--mut);border-radius:3px;padding:0 .3rem;align-self:flex-start}',
    P + '.is-skip{background:transparent;border-color:var(--border-color);border-style:solid}',
    P + '.is-skip strong{text-decoration:line-through;color:var(--mut)}',
    P + '.is-skip .at-skip{display:inline-block}',
    P + '.is-start{box-shadow:0 0 0 3px color-mix(in srgb,var(--focus) 35%,transparent)}',
    P + '.at-n[aria-pressed="true"]{outline:2px solid var(--focus);outline-offset:1px}',
    P + '.at-loopwrap{flex-basis:100%;width:100%}',
    P + '.at-loop{flex:1;border:2px solid var(--text-color);border-radius:6px;padding:.5rem;background:color-mix(in srgb,var(--surface) 60%,var(--bg-color))}',
    P + '.at-loop>.at-n{max-width:none;min-width:0;border-width:2px;margin-bottom:.5rem;flex-direction:row;flex-wrap:wrap;align-items:baseline;gap:.2rem .6rem}',
    P + '.at-loopt{font-size:.85rem;color:var(--mut);margin:0 0 .4rem}',
    P + '.at-shot{border:1px solid var(--mut);border-radius:8px;padding:.35rem;display:flex;flex-direction:column;gap:.3rem}',
    P + '.at-shot>.at-shott{font-size:.82rem;color:var(--mut);margin:0}',
    P + '.at-shot .at-flow{flex-wrap:nowrap}',
    P + '.at-chips{display:flex;flex-direction:column;gap:.25rem;margin-top:.3rem}',
    P + '.at-chipt{font-size:.82rem;color:var(--mut)}',
    P + '.at-chip{font-size:.82rem;border:1px solid var(--mut);border-radius:4px;background:var(--bg-color);padding:.1rem .35rem;display:flex;align-items:center;gap:.3rem}',
    P + '.at-chip .at-mk{width:.55rem;height:.55rem}',
    P + '.at-chip[aria-pressed="true"]{outline:2px solid var(--focus)}',
    P + '.at-cell{display:flex;flex-direction:column;gap:.3rem}',
    P + '.at-det{margin:.8rem 0 0;border:1px solid var(--border-color);border-radius:6px;padding:.6rem .8rem;background:var(--bg-color);min-height:7rem}',
    P + '.at-det h4{margin:0 0 .3rem;font-size:.95rem}',
    P + '.at-det dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:.25rem .8rem;font-size:.88rem}',
    P + '.at-det dt{font-weight:600;color:var(--mut)}',
    P + '.at-det dd{margin:0}',
    P + '.at-leg{display:flex;flex-wrap:wrap;gap:.3rem 1rem;font-size:.82rem;color:var(--mut);margin:.7rem 0 0;padding:0;list-style:none}',
    P + '.at-leg li{display:inline-flex;align-items:center;gap:.35rem}',
    P + '.at-sw{display:inline-block;width:1.2rem;height:.8rem;box-sizing:border-box}',
    P + '.at-note{font-size:.82rem;color:var(--mut);margin:.5rem 0 0}',
    '@container (max-width:560px){' +
      P + '.at-flow{flex-direction:column;flex-wrap:nowrap}' +
      P + '.at-flow>li{flex-direction:column}' +
      P + '.at-ar{padding:.1rem 0 .1rem 1rem;margin:0}' +
      P + '.at-ar::before{content:"\\2193"}' +
      P + '.at-n{max-width:none}' +
      P + '.at-shot .at-flow{flex-wrap:nowrap}' +
    '}'
  ].join('\n');

  function h(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function statusMark(st, labels) {
    var s = h('span', 'at-st');
    s.appendChild(h('span', 'at-mk s-' + st));
    s.appendChild(document.createTextNode(labels[st]));
    return s;
  }

  RSVis.mods[ID] = function mount(el, data, ctx) {
    if (!document.getElementById('rs-vis-athena-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-athena-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    ctx = ctx || {};
    var N = data.nodes, T = data.tools, L = data.statuses;
    var btns = {}, chipBtns = {};
    var current = null, entry = data.entry_points[0];

    while (el.firstChild) el.removeChild(el.firstChild);
    var root = h('div', 'at');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', data.title);
    el.appendChild(root);

    root.appendChild(h('h3', 'at-h', data.title));
    root.appendChild(h('p', 'at-p', data.counts));

    var bar = h('div', 'at-entry');
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Entry point');
    bar.appendChild(h('span', 'at-entry-t', 'Start from:'));
    var entryBtns = [];
    data.entry_points.forEach(function (ep) {
      var b = h('button', 'at-eb', ep.label);
      b.type = 'button';
      b.addEventListener('click', function () { setEntry(ep, true); });
      entryBtns.push([b, ep]);
      bar.appendChild(b);
    });
    root.appendChild(bar);
    root.appendChild(h('p', 'at-cap', data.entry_caption));

    function nodeBtn(id) {
      var n = N[id];
      var b = h('button', 'at-n k-' + n.kind);
      b.type = 'button';
      if (n.kind === 'gate') {
        var t = h('strong');
        t.appendChild(h('span', 'at-dia'));
        t.appendChild(document.createTextNode(n.label));
        b.appendChild(t);
        b.appendChild(h('span', 'at-gsub', 'decides which nodes run'));
      } else {
        b.appendChild(h('strong', null, n.label));
        if (n.kind !== 'proposed') b.appendChild(h('span', 'at-code', id.replace(/_/g, '_\u200b')));
      }
      if (n.per) b.appendChild(h('span', 'at-per', n.per));
      b.appendChild(statusMark(n.status, L));
      b.appendChild(h('span', 'at-skip', 'skipped'));
      b.addEventListener('click', function () { show(id, 'node'); });
      btns[id] = b;
      return b;
    }

    function chips(list, title) {
      var box = h('div', 'at-chips');
      box.setAttribute('role', 'group');
      box.setAttribute('aria-label', title);
      box.appendChild(h('span', 'at-chipt', title));
      list.forEach(function (tid) {
        var c = h('button', 'at-chip');
        c.type = 'button';
        c.appendChild(h('span', 'at-mk s-' + T[tid].status));
        c.appendChild(document.createTextNode(T[tid].label));
        c.addEventListener('click', function () { show(tid, 'tool'); });
        chipBtns[tid] = c;
        box.appendChild(c);
      });
      return box;
    }

    function cell(id) {
      var wrap = h('div', 'at-cell');
      wrap.appendChild(nodeBtn(id));
      var n = N[id];
      if (n.tools) wrap.appendChild(chips(n.tools, id === 'scene_researcher' ? data.mcp_label : 'Image generation'));
      return wrap;
    }

    function arrow() { var a = h('span', 'at-ar'); a.setAttribute('aria-hidden', 'true'); return a; }

    var flow = h('ol', 'at-flow');
    flow.setAttribute('aria-label', 'LangGraph nodes, left to right');
    data.graph.forEach(function (id, i) {
      var li = h('li');
      if (i > 0) li.appendChild(arrow());
      if (id === 'scene_pipeline') {
        li.className = 'at-loopwrap';
        var lp = h('div', 'at-loop');
        lp.appendChild(nodeBtn(id));
        lp.appendChild(h('p', 'at-loopt', 'Inside the loop, for each scene in order:'));
        var inner = h('ol', 'at-flow');
        inner.setAttribute('aria-label', 'Modules run inside the scene pipeline');
        data.loop.forEach(function (item, j) {
          var il = h('li');
          if (j > 0) il.appendChild(arrow());
          if (Array.isArray(item)) {
            var sb = h('div', 'at-shot');
            sb.appendChild(h('p', 'at-shott', 'For each shot:'));
            var so = h('ol', 'at-flow');
            item.forEach(function (sid, k) {
              var sl = h('li');
              if (k > 0) sl.appendChild(arrow());
              sl.appendChild(cell(sid));
              so.appendChild(sl);
            });
            sb.appendChild(so);
            il.appendChild(sb);
          } else {
            il.appendChild(cell(item));
          }
          inner.appendChild(il);
        });
        lp.appendChild(inner);
        li.appendChild(lp);
      } else {
        li.appendChild(nodeBtn(id));
      }
      flow.appendChild(li);
    });
    root.appendChild(flow);

    var det = h('div', 'at-det');
    det.setAttribute('aria-live', 'polite');
    root.appendChild(det);

    var leg = h('ul', 'at-leg');
    leg.setAttribute('aria-label', 'Key');
    [['k-graph', 'LangGraph node'], ['k-loop', 'Runs only inside the scene loop'], ['k-gate', 'Plain-code router'],
     ['s-built', L.built], ['s-partial', L.partial], ['s-proposed', L.proposed]].forEach(function (k) {
      var li = h('li');
      var sw = h('span', (k[0].indexOf('s-') === 0 ? 'at-mk ' : 'at-sw ') + k[0]);
      sw.setAttribute('aria-hidden', 'true');
      li.appendChild(sw);
      li.appendChild(document.createTextNode(k[1]));
      leg.appendChild(li);
    });
    root.appendChild(leg);
    root.appendChild(h('p', 'at-note', data.source_note));

    function skipped(id) { return entry.skipped_nodes.indexOf(id) >= 0; }

    function row(dl, k, v) { dl.appendChild(h('dt', null, k)); dl.appendChild(h('dd', null, v)); }

    function show(id, type) {
      current = [id, type];
      Object.keys(btns).forEach(function (k) { btns[k].setAttribute('aria-pressed', String(type === 'node' && k === id)); });
      Object.keys(chipBtns).forEach(function (k) { chipBtns[k].setAttribute('aria-pressed', String(type === 'tool' && k === id)); });
      while (det.firstChild) det.removeChild(det.firstChild);
      var dl = h('dl');
      if (type === 'tool') {
        var t = T[id];
        det.appendChild(h('h4', null, t.label));
        row(dl, 'What it does', t.what);
        row(dl, 'Status', L[t.status]);
      } else {
        var n = N[id];
        det.appendChild(h('h4', null, n.label));
        row(dl, 'Purpose', n.purpose);
        row(dl, 'Writes', n.writes);
        if (n.per) row(dl, 'Runs', n.per + ', inside the scene pipeline');
        row(dl, 'Status', n.note || L[n.status]);
        if (skipped(id)) row(dl, 'This entry', 'Skipped: not invoked when starting from "' + entry.label + '".');
        else if (entry.starts_at === id && id !== 'intent_resolver') row(dl, 'This entry', 'Work starts here when starting from "' + entry.label + '". The intent node still runs first to classify the brief.');
      }
      det.appendChild(dl);
    }

    function setEntry(ep, announce) {
      entry = ep;
      entryBtns.forEach(function (p) { p[0].setAttribute('aria-pressed', String(p[1] === ep)); });
      Object.keys(btns).forEach(function (k) {
        var s = skipped(k);
        btns[k].classList.toggle('is-skip', s);
        btns[k].classList.toggle('is-start', ep.starts_at === k && k !== 'intent_resolver');
      });
      if (announce) {
        while (det.firstChild) det.removeChild(det.firstChild);
        det.appendChild(h('h4', null, 'Start from: ' + ep.label));
        var dl = h('dl');
        var sk = ep.skipped_nodes.map(function (k) { return N[k].label; });
        row(dl, 'Skipped', sk.length ? sk.join(', ') : 'None; every node runs.');
        row(dl, 'Work starts at', N[ep.starts_at].label + (ep.starts_at === 'intent_resolver' ? '' : ', after the intent node classifies the brief'));
        if (ep.note) row(dl, 'Note', ep.note);
        det.appendChild(dl);
        current = null;
        Object.keys(btns).forEach(function (k) { btns[k].setAttribute('aria-pressed', 'false'); });
        Object.keys(chipBtns).forEach(function (k) { chipBtns[k].setAttribute('aria-pressed', 'false'); });
      } else if (current) {
        show(current[0], current[1]);
      }
    }

    setEntry(entry, false);
    show('intent_resolver', 'node');
  };
})();
