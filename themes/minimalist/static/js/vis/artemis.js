window.RSVis = window.RSVis || { mods: {} };
(function () {
  'use strict';
  var ID = 'artemis';
  var CSS =
    '.ar-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr));gap:8px;margin:10px 0;padding:0;list-style:none}' +
    '.ar-tier{border:1.5px solid var(--border-color);border-radius:6px;padding:8px 10px;background:var(--surface);color:var(--mut);font-size:14px}' +
    '.ar-tier.on{border:3px solid var(--ok);color:var(--text-color);font-weight:600}' +
    '.ar-why{margin:6px 0 0;font-size:14px}' +
    '.ar-req{margin:8px 0 0;font-size:15px}' +
    '.ar-ill{font-size:12px;color:var(--mut);margin:6px 0 0}' +
    '.ar-tag{font-size:12px;font-weight:600}';

  function run(el, data) {
    var c = data.card, E = RSCard.el, S = RSCard.svg;
    if (!document.getElementById('rs-vis-' + ID + '-css')) {
      var st = E('style'); st.id = 'rs-vis-' + ID + '-css'; st.textContent = CSS; document.head.appendChild(st);
    }
    function routeTab(panel) {
      var reqs = c.requests, names = c.tierNames;
      var lab = E('label', '', 'Pick a request: '); lab.style.fontSize = '14px';
      var ctl = E('div', 'rc-ctl'); ctl.appendChild(lab);
      var reqP = E('p', 'ar-req'), strip = E('ul', 'ar-strip'), why = E('p', 'ar-why');
      why.setAttribute('aria-live', 'polite');
      var tiers = names.map(function (n) { var li = E('li', 'ar-tier', n); strip.appendChild(li); return li; });
      function pick(id) {
        var r = reqs.filter(function (x) { return x.id === id; })[0];
        reqP.textContent = r.text;
        tiers.forEach(function (t, i) { t.className = 'ar-tier' + (i === r.tier ? ' on' : ''); t.textContent = names[i] + (i === r.tier ? ' (chosen)' : ''); });
        why.textContent = 'Why: ' + r.why;
      }
      var seg = RSCard.seg(reqs.map(function (r) { return { id: r.id, label: r.label }; }), 'easy', pick);
      seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Sample request difficulty');
      ctl.appendChild(seg);
      panel.appendChild(ctl); panel.appendChild(reqP); panel.appendChild(strip); panel.appendChild(why);
      panel.appendChild(E('p', 'ar-ill', 'Illustration, not a real routed example: it shows the idea of matching model size to request difficulty.'));
      pick('easy');
    }
    var STATE = { base: 'Baseline', ok: 'Per-task lookup', tried: 'Tried and did not work', bound: 'Upper bound, uses the answers' };
    function ladderTab(panel) {
      var L = data.ladder, W = 700, rowH = 52, labW = 200, barW = 380, max = 0.7;
      var v = S('svg', { viewBox: '0 0 ' + W + ' ' + (L.length * rowH + 4), role: 'img', 'aria-label': 'Accuracy ladder: ' + L.map(function (r) { return r.label + ' ' + r.value; }).join(', ') });
      L.forEach(function (r, i) {
        var y = i * rowH + 4;
        v.appendChild(S('text', { x: 0, y: y + 16, 'font-weight': 600, 'font-size': 16 }, r.label));
        v.appendChild(S('text', { x: 0, y: y + 34, class: 'm', 'font-size': 13 }, STATE[r.kind] || ''));
        v.appendChild(S('rect', { x: labW, y: y + 6, width: barW, height: 22, rx: 3, fill: 'var(--surface)', stroke: 'var(--border-color)' }));
        var fill = r.kind === 'ok' ? 'var(--ok)' : r.kind === 'tried' ? 'var(--bad,#b3261e)' : 'var(--mut)';
        v.appendChild(S('rect', { x: labW, y: y + 6, width: barW * r.v / max, height: 22, rx: 3, fill: fill, opacity: r.kind === 'bound' ? 0.45 : 1 }));
        v.appendChild(S('text', { x: labW + barW + 10, y: y + 22, 'font-family': 'var(--font-mono)', 'font-size': 16 }, r.value));
      });
      panel.appendChild(v);
      panel.appendChild(E('p', 'rc-line', 'Accuracy on the test split, n=10,198 questions, five-model pool, offline. Early, not final: the learned router did not beat the per-task lookup.'));
    }
    RSCard.mount(el, data, { tabs: [
      { id: 'route', label: 'Route a request', render: routeTab },
      { id: 'ladder', label: 'The ladder', render: ladderTab }
    ] });
  }
  window.RSVis.register ? window.RSVis.register(ID, run) : (window.RSVis.mods[ID] = run);
})();
