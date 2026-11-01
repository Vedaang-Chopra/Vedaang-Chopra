/* CAD card: verifying a code-writing model step by step. Tabs render 3D parts on a canvas
   with a tiny painter's-algorithm renderer (orthographic, z up). Data: content/data/visuals/cad.json. */
window.RSVis = window.RSVis || { mods: {} };
(function () {
var ID = 'cad';
var CSS = [
'.cd-cv{display:block;width:100%;aspect-ratio:4/3;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-color)}',
'.cd-cv.drag{cursor:grab;touch-action:pan-y}',
'.cd-cv:focus-visible,.cd-range:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
'.cd-two,.cd-three{display:grid;gap:10px}',
'.cd-two{grid-template-columns:repeat(2,minmax(0,1fr));max-width:760px}',
'.cd-three{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}',
'.cd-fig{margin:0}',
'.cd-fig figcaption{font-size:13px;text-align:center;margin:4px 0 0}',
'.cd-fig figcaption b{font-weight:600}',
'.cd-range{flex:1 1 10rem;min-width:8rem;accent-color:var(--text-color)}',
'.cd-read{border:1px solid var(--border-color);border-radius:6px;padding:8px 10px;font-size:14px;margin:8px 0 0}',
'.cd-read p{margin:2px 0}',
'.cd-code{font-family:var(--font-mono);font-size:12.5px;overflow-wrap:anywhere}',
'.cd-st{font-weight:600}',
'.cd-st.bad{color:var(--bad,#b3261e)}',
'.cd-st.ok{color:var(--ok)}',
'.cd-sub{font-size:14px;font-weight:600;margin:16px 0 4px}',
'.cd-vl{list-style:none;margin:0;padding:0;display:grid;gap:4px;font-size:14px}',
'.cd-vl li{display:grid;grid-template-columns:5.6rem 1fr;gap:8px;align-items:baseline;border:1px solid var(--border-color);border-radius:6px;padding:5px 8px}',
'.cd-vl li.cause{border:2px solid var(--bad,#b3261e)}',
'.cd-vl li.missing{border-style:dashed}',
'.cd-vw{font-weight:600}',
'.cd-vl li.cause .cd-vw{color:var(--bad,#b3261e)}',
'.cd-vl .m{color:var(--mut)}',
'.cd-pre{font-family:var(--font-mono);font-size:12.5px;margin:6px 0 0;padding:8px;border:1px solid var(--border-color);border-radius:6px;background:var(--surface);overflow-x:auto;white-space:pre}',
'.cd-pre .hi{color:var(--ok);font-weight:600}',
'.cd-pre .lb{font-family:var(--font-sans);color:var(--mut);font-size:12px}',
'.cd-cardtxt{margin:8px 0 0;padding:8px 10px;border-left:3px solid var(--ok);background:var(--surface);font-size:14px}'
].join('\n');

var E = function (t, c, x) { return RSCard.el(t, c, x); };
var S = function (t, a, x) { return RSCard.svg(t, a, x); };

/* page colour variable -> [r,g,b]; probed inside the card because some variables are scoped to .rs */
function rgb(ctx, name, fb) {
  var probe = document.createElement('span'); probe.style.display = 'none';
  probe.style.color = 'var(' + name + (fb ? ',' + fb : '') + ')';
  ctx.appendChild(probe);
  var m = getComputedStyle(probe).color.match(/[\d.]+/g);
  ctx.removeChild(probe);
  return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
}
function mix(a, b, t) {
  return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' + Math.round(a[2] + (b[2] - a[2]) * t) + ')';
}
var TINTS = [0.14, 0.32, 0.22, 0.40, 0.18, 0.28, 0.36, 0.12, 0.26, 0.34, 0.20];

function normals(m) {
  if (m._n) return m._n;
  var v = m.v, f = m.f, n = [];
  for (var i = 0; i < f.length; i += 3) {
    var a = f[i] * 3, b = f[i + 1] * 3, c = f[i + 2] * 3;
    var ux = v[b] - v[a], uy = v[b + 1] - v[a + 1], uz = v[b + 2] - v[a + 2];
    var wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
    var nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    var l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    n.push(nx / l, ny / l, nz / l);
  }
  m._n = n;
  return n;
}
function bounds(list) {
  var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
  list.forEach(function (m) {
    for (var i = 0; i < m.v.length; i += 3) for (var k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], m.v[i + k]); hi[k] = Math.max(hi[k], m.v[i + k]);
    }
  });
  var c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  var r = Math.sqrt(Math.pow(hi[0] - lo[0], 2) + Math.pow(hi[1] - lo[1], 2) + Math.pow(hi[2] - lo[2], 2)) / 2;
  return { c: c, r: r || 1 };
}

/* scene = {mesh, b, ghosts:[mesh], tone:'bad'|null, empty:text}; view = {yaw, pitch} shared by viewers */
function Viewer(cv, view) { this.cv = cv; this.view = view; this.scene = null; }
Viewer.prototype.draw = function () {
  var cv = this.cv, sc = this.scene, w = cv.clientWidth, hgt = cv.clientHeight;
  if (!w || !hgt) return;
  var dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
  var g = cv.getContext('2d');
  if (!g) return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  var ctx = cv.parentNode || document.body;
  var bg = rgb(ctx, '--bg-color'), tx = rgb(ctx, '--text-color'), mu = rgb(ctx, '--mut'), bad = rgb(ctx, '--bad', '#b3261e');
  g.fillStyle = mix(bg, tx, 0); g.fillRect(0, 0, w, hgt);
  if (!sc || !sc.mesh) {
    g.fillStyle = mix(mu, mu, 0); g.font = '13px ' + (getComputedStyle(cv).fontFamily || 'sans-serif');
    g.textAlign = 'center'; g.fillText(sc && sc.empty || 'No solid yet', w / 2, hgt / 2);
    return;
  }
  var cy = Math.cos(this.view.yaw), sy = Math.sin(this.view.yaw), cp = Math.cos(this.view.pitch), sp = Math.sin(this.view.pitch);
  var s = Math.min(w, hgt) * (sc.zoom || 0.46) / sc.b.r, ox = w / 2, oy = hgt / 2, c = sc.b.c;
  function pr(v, i) {
    var x = v[i] - c[0], y = v[i + 1] - c[1], z = v[i + 2] - c[2];
    var x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    return [ox + x1 * s, oy - (z * cp + y1 * sp) * s, y1 * cp - z * sp];
  }
  var items = [];
  function addEdges(m, ghost) {
    var e = m.seg;
    for (var i = 0; i < e.length; i += 6) {
      var a = pr(e, i), b = pr(e, i + 3);
      items.push({ t: 1, d: ghost ? -1e9 : (a[2] + b[2]) / 2 - sc.b.r * 0.02, a: a, b: b, gh: ghost });
    }
  }
  var m = sc.mesh, v = m.v, f = m.f, nn = normals(m);
  for (var i = 0; i < f.length; i += 3) {
    var a = pr(v, f[i] * 3), b = pr(v, f[i + 1] * 3), cc = pr(v, f[i + 2] * 3);
    var nx = nn[i], ny = nn[i + 1], nz = nn[i + 2];
    var vy = nx * sy + ny * cy, vz = nz * cp + vy * sp, vx = nx * cy - ny * sy, vd = vy * cp - nz * sp;
    var lam = Math.abs(-0.35 * vx + 0.55 * vz - 0.75 * vd);
    var grp = m.g ? m.g[i / 3] : 0;
    items.push({ t: 0, d: (a[2] + b[2] + cc[2]) / 3, p: [a, b, cc], k: TINTS[grp % TINTS.length] + 0.16 * (1 - Math.min(1, lam)) });
  }
  addEdges(m, false);
  (sc.ghosts || []).forEach(function (gm) { addEdges(gm, true); });
  items.sort(function (p, q) { return q.d - p.d; });
  g.lineJoin = 'round';
  var edge = sc.tone === 'bad' ? mix(bg, bad, 0.95) : mix(bg, tx, 0.9);
  items.forEach(function (it) {
    if (it.t === 0) {
      var col = mix(bg, tx, it.k);
      g.fillStyle = col; g.strokeStyle = col; g.lineWidth = 0.6;
      g.beginPath(); g.moveTo(it.p[0][0], it.p[0][1]); g.lineTo(it.p[1][0], it.p[1][1]); g.lineTo(it.p[2][0], it.p[2][1]);
      g.closePath(); g.fill(); g.stroke();
    } else {
      g.strokeStyle = it.gh ? mix(bg, mu, 0.9) : edge;
      g.lineWidth = it.gh ? 1 : (sc.tone === 'bad' ? 1.8 : 1.3);
      g.setLineDash(it.gh ? [4, 3] : []);
      g.beginPath(); g.moveTo(it.a[0], it.a[1]); g.lineTo(it.b[0], it.b[1]); g.stroke();
    }
  });
  g.setLineDash([]);
};

/* redraw a set of viewers on resize (also covers first open of a closed card) and theme change */
function keepFresh(viewers) {
  var raf = 0;
  function later() { if (raf) return; raf = requestAnimationFrame(function () { raf = 0; viewers.forEach(function (v) { v.draw(); }); }); }
  if (window.ResizeObserver) { var ro = new ResizeObserver(later); viewers.forEach(function (v) { ro.observe(v.cv); }); }
  else window.addEventListener('resize', later);
  if (window.MutationObserver) {
    var mo = new MutationObserver(later);
    mo.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
  }
  return later;
}
function canvas(label) {
  var cv = E('canvas', 'cd-cv');
  cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', label);
  return cv;
}
function figure(cv, capNodes) {
  var fig = E('figure', 'cd-fig'), fc = E('figcaption');
  capNodes.forEach(function (n) { fc.appendChild(typeof n === 'string' ? document.createTextNode(n) : n); });
  fig.appendChild(cv); fig.appendChild(fc);
  return fig;
}

/* Tab 1: slide through build steps; target and first try side by side, then the verifier verdicts */
function findTab(panel, data) {
  var F = data.find, M = data.meshes, st = F.steps, n = st.length;
  panel.appendChild(E('p', 'rc-line', 'Sample ' + F.sample + '. Slide through the steps: target and the model’s first try match until step 7, where the handle turns the wrong way.'));
  var view = { yaw: -0.65, pitch: 0.5 };
  var used = [];
  st.forEach(function (s) { [s.mt, s.mg].forEach(function (k) { if (k != null) used.push(M[k]); }); });
  var b = bounds(used);
  var cvT = canvas('Target part at the chosen step'), cvG = canvas('Model’s first try at the chosen step');
  var vT = new Viewer(cvT, view), vG = new Viewer(cvG, view);
  var grid = E('div', 'cd-two');
  grid.appendChild(figure(cvT, [E('b', null, 'Target')]));
  grid.appendChild(figure(cvG, [E('b', null, 'Model’s first try'), ' (dashed: where the target has it)']));
  panel.appendChild(grid);

  var ctl = E('div', 'rc-ctl'), id = 'cdr' + Math.random().toString(36).slice(2, 7);
  var lab = E('label', null, 'Build step'); lab.setAttribute('for', id);
  var rg = document.createElement('input'); rg.type = 'range'; rg.id = id; rg.className = 'cd-range';
  rg.min = 1; rg.max = n; rg.step = 1; rg.value = n;
  ctl.appendChild(lab); ctl.appendChild(rg); panel.appendChild(ctl);

  var read = E('div', 'cd-read'); read.setAttribute('aria-live', 'polite'); panel.appendChild(read);
  var STATE = {
    none: ['No solid yet', ''], same: ['Same', 'ok'], first: ['First difference', 'bad'],
    carried: ['Still different, carried over from step 7', 'bad']
  };
  function line(label, code) {
    var p = E('p'); p.appendChild(E('span', 'rc-line', label + ': ')); p.appendChild(E('span', 'cd-code', code)); return p;
  }
  function show(i) {
    var s = st[i], sw = STATE[s.state], diff = s.state === 'first' || s.state === 'carried';
    rg.setAttribute('aria-valuetext', 'Step ' + (i + 1) + ' of ' + n + ', ' + s.op + ', ' + sw[0]);
    var base = s.base != null ? [M[s.base]] : [];
    vT.scene = { mesh: s.mt != null ? M[s.mt] : null, b: b, ghosts: base, zoom: 0.6 };
    vG.scene = { mesh: s.mg != null ? M[s.mg] : null, b: b, ghosts: diff && s.mt != null ? base.concat([M[s.mt]]) : base, tone: diff ? 'bad' : null, zoom: 0.6 };
    vT.draw(); vG.draw();
    read.textContent = '';
    var h = E('p'); h.appendChild(E('strong', null, 'Step ' + (i + 1) + ' of ' + n + ': ' + s.op + '. '));
    h.appendChild(E('span', 'cd-st ' + sw[1], sw[0] + '.'));
    read.appendChild(h);
    read.appendChild(line('Target code', s.t));
    read.appendChild(line('First try', s.g));
    if (i === n - 1) read.appendChild(E('p', 'rc-line', 'Finished first try overlaps the target at IoU ' + F.iou + ' (1 = identical).'));
  }
  rg.addEventListener('input', function () { show(+rg.value - 1); });
  keepFresh([vT, vG]);
  show(n - 1);

  var V = F.verdicts;
  panel.appendChild(E('p', 'cd-sub', 'The step checker’s verdicts on another run, sample ' + V.sample));
  var ul = E('ul', 'cd-vl'), WORD = { cause: 'Cause', replaced: 'Replaced', inherited: 'Inherited', missing: 'Missing' };
  ul.setAttribute('aria-label', 'Verdict for each target step');
  V.rows.forEach(function (r) {
    var li = E('li', r.v); li.appendChild(E('span', 'cd-vw', WORD[r.v]));
    var t = E('span', null, r.t + ' → ' + r.g); if (r.note) t.appendChild(E('span', 'm', ' (' + r.note + ')'));
    li.appendChild(t); ul.appendChild(li);
  });
  panel.appendChild(ul);
  panel.appendChild(E('p', 'rc-line', 'Cause: first step whose shape differs. Inherited: wrong only because of an earlier cause. Missing: a target step with no match.'));
  panel.appendChild(E('p', 'rc-line', 'It named the right root, the wrong sketch plane, yet both repair tries were rejected. One example, not a validation.'));
}

/* Tab 2: one learned card changes the first try on a sample it never saw; drag turns all three */
function reuseTab(panel, data) {
  var R = data.reuse;
  panel.appendChild(E('p', 'rc-line', 'Sample ' + R.sample + ', unseen by the card. One card, two samples: an example, not a rate.'));
  var view = { yaw: 2.2, pitch: 0.3 }, b = bounds(R.shapes.map(function (s) { return s.m; })), vs = [];
  var grid = E('div', 'cd-three');
  R.shapes.forEach(function (s) {
    var cv = canvas(s.alt + ' Drag, or focus and use the arrow keys, to turn all three.');
    cv.classList.add('drag'); cv.tabIndex = 0;
    var v = new Viewer(cv, view); v.scene = { mesh: s.m, b: b, tone: s.iou != null && s.iou < 0.9 ? 'bad' : null }; vs.push(v);
    var cap = [E('b', null, s.label)];
    if (s.iou != null) cap.push(E('br'), 'IoU ' + s.iou);
    grid.appendChild(figure(cv, cap));
  });
  panel.appendChild(grid);
  var redraw = keepFresh(vs);
  vs.forEach(function (v) { bindRotate(v.cv, view, redraw); });
  panel.appendChild(E('p', 'rc-line', 'Drag a shape, or focus it and use the arrow keys, to turn all three.'));

  var pre = E('pre', 'cd-pre');
  pre.setAttribute('aria-label', 'The one-line code change');
  R.code.forEach(function (c, i) {
    pre.appendChild(E('span', 'lb', c[0] + '\n'));
    if (i === R.code.length - 1) {
      var k = c[1].lastIndexOf('-90');
      pre.appendChild(document.createTextNode(c[1].slice(0, k)));
      pre.appendChild(E('span', 'hi', '-90'));
      pre.appendChild(document.createTextNode(c[1].slice(k + 3)));
    } else pre.appendChild(document.createTextNode(c[1] + '\n'));
  });
  panel.appendChild(pre);
  panel.appendChild(E('p', 'rc-line', 'The target lists the axis points the other way round, so the same “90” turns the opposite way. The card flipped one sign.'));
  var ct = E('div', 'cd-cardtxt');
  ct.appendChild(E('strong', null, 'The skill card, learned from one repair of sample ' + R.from + ': '));
  ct.appendChild(document.createTextNode(R.card.join(' ')));
  panel.appendChild(ct);
  panel.appendChild(E('p', 'rc-line', 'Added to the prompt; no second turn. Over 3 repeats: 1 of 3 correct with the card, 0 of 3 without. A second unseen sample went from 0 of 3 to 2 of 3.'));
}
function bindRotate(cv, view, redraw) {
  var drag = null;
  cv.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY]; if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', function (e) {
    if (!drag) return;
    view.yaw += (e.clientX - drag[0]) * 0.01;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY]; redraw();
  });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (nm) { cv.addEventListener(nm, function () { drag = null; }); });
  cv.addEventListener('keydown', function (e) {
    var k = e.key, st = 0.2;
    if (k === 'ArrowLeft') view.yaw -= st;
    else if (k === 'ArrowRight') view.yaw += st;
    else if (k === 'ArrowUp') view.pitch = Math.min(1.45, view.pitch + st);
    else if (k === 'ArrowDown') view.pitch = Math.max(-1.45, view.pitch - st);
    else return;
    e.preventDefault(); redraw();
  });
}

/* Tab 3: three systems on the same prompts; one switch picks the metric */
function compareTab(panel, data) {
  var C = data.compare;
  panel.appendChild(E('p', 'rc-line', 'Early, not final. Same ' + C.n + ' samples per group for every system, same model. Repair is scored against the true shape; crashed runs count as 0.'));
  var METRIC = {
    mean: { label: 'Mean IoU', get: function (r) { return r[0]; }, fmt: function (v) { return v.toFixed(3); } },
    ge9: { label: 'Share with IoU ≥ 0.9', get: function (r) { return r[1] / C.n; }, fmt: function (v) { return Math.round(v * 100) + '%'; } }
  };
  var host = E('div'), live = E('p', 'rc-line'); live.setAttribute('aria-live', 'polite');
  function draw(key) {
    var M = METRIC[key], W = 700, H = 250, l = 46, r = 8, t = 16, bt = 52, pw = W - l - r, ph = H - t - bt;
    var Y = function (v) { return t + ph * (1 - v); };
    var desc = [];
    var v = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (q) {
      v.appendChild(S('line', { x1: l, x2: W - r, y1: Y(q), y2: Y(q), stroke: 'var(--border-color)' }));
      v.appendChild(S('text', { x: l - 6, y: Y(q) + 4, 'text-anchor': 'end', class: 'm', 'font-size': 11 }, key === 'mean' ? q.toFixed(2) : Math.round(q * 100) + '%'));
    });
    var gw = pw / C.groups.length, bw = gw * 0.8 / C.systems.length;
    C.groups.forEach(function (g, gi) {
      var x0 = l + gi * gw + gw * 0.1;
      C.systems.forEach(function (sn, j) {
        var val = M.get(g.rows[j]), x = x0 + j * bw, y = Y(val), ours = j === C.systems.length - 1;
        v.appendChild(S('rect', { x: x + 3, y: y, width: bw - 6, height: t + ph - y, rx: 2,
          fill: ours ? 'var(--ok)' : 'var(--mut)', 'fill-opacity': ours ? 1 : 0.45 + 0.25 * j }));
        v.appendChild(S('text', { x: x + bw / 2, y: y - 5, 'text-anchor': 'middle', 'font-size': 12 }, M.fmt(val)));
        v.appendChild(S('text', { x: x + bw / 2, y: t + ph + 16, 'text-anchor': 'middle', class: 'm', 'font-size': 11 }, sn));
        desc.push(g.label + ' ' + sn + ' ' + M.fmt(val));
      });
      v.appendChild(S('text', { x: l + gi * gw + gw / 2, y: H - 10, 'text-anchor': 'middle', 'font-weight': 600 }, g.label + ' (n = ' + C.n + ')'));
    });
    v.setAttribute('aria-label', M.label + ': ' + desc.join('; ') + '.');
    host.textContent = ''; host.appendChild(v);
    live.textContent = 'Showing ' + M.label.toLowerCase() + '. ' + desc.join('; ') + '.';
  }
  var ctl = E('div', 'rc-ctl'); ctl.appendChild(E('span', null, 'Metric'));
  var seg = RSCard.seg([{ id: 'mean', label: 'Mean IoU' }, { id: 'ge9', label: 'IoU ≥ 0.9' }], 'mean', draw);
  seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Metric');
  ctl.appendChild(seg); panel.appendChild(ctl); panel.appendChild(host); panel.appendChild(live);
  panel.appendChild(E('p', 'rc-line', 'Single-shot: one try. Self-debug: fixes code that crashes. Repair: the scored retry loop above. IoU 1 means identical to the target.'));
  draw('mean');
}

function mount(el, data) {
  if (!document.getElementById('cd-css')) {
    var st = document.createElement('style'); st.id = 'cd-css'; st.textContent = CSS; document.head.appendChild(st);
  }
  RSCard.mount(el, data, { tabs: [
    { id: 'find', label: 'Find the error', render: findTab },
    { id: 'reuse', label: 'Reuse the lesson', render: reuseTab },
    { id: 'compare', label: 'Compare systems', render: compareTab }
  ] });
}
if (window.RSVis.register) window.RSVis.register(ID, mount); else window.RSVis.mods[ID] = mount;
})();
