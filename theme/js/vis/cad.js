window.RSVis = window.RSVis || { mods: {} };
(function () {
var ID = 'cad';
var P = '.rs-vis[data-vis="cad"] ';
var CSS = [
'&{font-family:var(--font-sans);color:var(--text-color);font-size:15px;line-height:1.45}',
'&button{font:inherit;color:var(--text-color);cursor:pointer}',
'&button:focus-visible,&summary:focus-visible,&canvas:focus-visible,&input:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
'&.cd-count{margin:0;color:var(--mut);font-size:14px}',
'&.cd-title{margin:.1rem 0 .7rem;font-size:1.1rem;line-height:1.3;font-weight:650}',
'&.cd-nav{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:.9rem 0 0}',
'&.cd>.cd-nav{border-top:1px solid var(--border-color);padding-top:.8rem}',
'&.cd-btn{border:1px solid var(--border-color);background:var(--btn-bg);border-radius:6px;padding:.35rem .85rem;min-height:2.2rem}',
'&.cd-btn:disabled{opacity:.5;cursor:default}',
'&.cd-dots{display:flex;gap:.3rem;margin:0 .3rem}',
'&.cd-dot{width:2rem;height:2rem;border-radius:50%;border:1px solid var(--border-color);background:var(--btn-bg);padding:0;font-size:13px}',
'&.cd-dot[aria-current="step"]{background:var(--text-color);color:var(--bg-color);border-color:var(--text-color);font-weight:700}',
'&.cd-pane p{margin:.35rem 0}',
'&.cd-note{color:var(--mut);font-size:14px}',
'&.cd-ill{display:inline-block;margin:0 0 .6rem;padding:.1rem .5rem;border:1px dashed var(--mut);border-radius:4px;color:var(--mut);font-size:13px}',
'&.cd-loop{list-style:none;margin:.2rem 0 .6rem;padding:0;display:flex;flex-wrap:wrap;gap:.35rem;align-items:center}',
'&.cd-loop li{display:flex;align-items:center;gap:.35rem}',
'&.cd-box{border:1px solid var(--border-color);background:var(--surface,var(--bg-color));border-radius:6px;padding:.4rem .6rem;font-weight:600;font-size:14px}',
'&.cd-box.again{border-style:dashed}',
'&.cd-arr{color:var(--mut)}',
'&.cd-v3{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:1rem;align-items:start}',
'&.cd-cv{display:block;width:100%;aspect-ratio:4/3;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-color);touch-action:pan-y;cursor:grab;max-width:560px}',
'&.cd-hint{font-size:13px;color:var(--mut);margin:.3rem 0 0;display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}',
'&.cd-hint .cd-btn{font-size:13px;padding:.2rem .6rem;min-height:1.9rem}',
'&.cd-read{border:1px solid var(--border-color);border-radius:6px;padding:.6rem .8rem;background:var(--bg-color)}',
'&.cd-read .op{font-weight:600;margin:0 0 .4rem}',
'&.cd-ctr{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.4rem;margin:.3rem 0}',
'&.cd-ctr div{border:1px solid var(--border-color);border-radius:6px;padding:.3rem .4rem}',
'&.cd-ctr dt{font-size:13px;color:var(--mut)}',
'&.cd-ctr dd{margin:0;font-family:var(--font-mono);font-size:15px;font-weight:600}',
'&.cd-ctr dd.none{font-family:var(--font-sans);font-size:13px;font-weight:400;color:var(--mut)}',
'&.cd-slide{display:flex;gap:.4rem;align-items:center;margin:.6rem 0 0;flex-wrap:wrap}',
'&.cd-slide input{flex:1 1 8rem;min-width:6rem;accent-color:var(--text-color)}',
'&.cd-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}',
'&.cd-chain{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:.35rem}',
'&.cd-chip{display:flex;flex-direction:column;align-items:flex-start;gap:.1rem;padding:.35rem .55rem;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-color);text-align:left;min-width:6.5rem}',
'&.cd-chip .nm{font-weight:600;font-size:14px}',
'&.cd-chip .vd{font-size:13px;display:flex;align-items:center;gap:.3rem}',
'&.cd-chip[aria-pressed="true"]{box-shadow:0 0 0 2px var(--text-color)}',
'&.cd-chip.correct .vd{color:var(--ok)}',
'&.cd-chip.cause{border:2px solid var(--bad)}',
'&.cd-chip.cause .vd{color:var(--bad);font-weight:600}',
'&.cd-chip.inherited{border-style:dashed;border-color:var(--warn)}',
'&.cd-det{border:1px solid var(--border-color);border-radius:6px;padding:.5rem .75rem;margin:.6rem 0 0;min-height:3rem}',
'&.mk{display:inline-block;width:10px;height:10px;flex:none;box-sizing:border-box}',
'&.mk.correct,&.mk.built{background:var(--ok);border-radius:50%}',
'&.mk.built{border-radius:2px}',
'&.mk.cause{background:var(--bad);transform:rotate(45deg)}',
'&.mk.inherited{border:2px dashed var(--warn);border-radius:50%}',
'&.mk.partial{border:2px solid var(--warn);background:linear-gradient(90deg,var(--warn) 50%,transparent 50%)}',
'&.mk.proposed{border:2px dashed var(--mut);background:transparent}',
'&.cd-trio{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.5rem}',
'&.cd-trio figure{margin:0}',
'&.cd-trio figcaption{font-size:13px;font-weight:600;text-align:center;margin:.2rem 0 0}',
'&.cd-iou{display:block;width:100%;max-width:420px;height:auto;margin:.5rem 0 0}',
'&.cd-iou text{fill:var(--text-color);font-family:var(--font-sans);font-size:12px}',
'&.cd-iou .lb{fill:var(--mut)}',
'&.cd-pane details{margin:.6rem 0 0}',
'&.cd-pane summary{cursor:pointer;color:var(--rs-link);font-size:14px}',
'&.cd-pane pre{font-family:var(--font-mono);font-size:12.5px;overflow-x:auto;border:1px solid var(--border-color);border-radius:6px;padding:.5rem;margin:.3rem 0;white-space:pre}',
'&.cd-stat{list-style:none;margin:.3rem 0;padding:0;display:flex;flex-direction:column;gap:.4rem}',
'&.cd-stat li{display:flex;gap:.5rem;align-items:baseline;border:1px solid var(--border-color);border-radius:6px;padding:.4rem .6rem}',
'&.cd-stat li.proposed{border-style:dashed}',
'&.cd-parts{margin:.3rem 0;padding-left:1.1rem;font-size:14px}',
'&.cd-parts li{margin:.2rem 0}',
'@media (max-width:640px){&.cd-v3{grid-template-columns:1fr}}'
].join('\n').replace(/&/g, P);

function h(tag, cls, text) {
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function btn(text, cls, label) {
  var b = h('button', cls || 'cd-btn', text);
  b.type = 'button';
  if (label) b.setAttribute('aria-label', label);
  return b;
}
function mark(kind) { var m = h('span', 'mk ' + kind); m.setAttribute('aria-hidden', 'true'); return m; }

/* resolve page colour variables to rgb via a probe element */
var probe = null;
function rgb(name) {
  if (!probe) { probe = document.createElement('span'); probe.style.display = 'none'; document.body.appendChild(probe); }
  probe.style.color = 'var(' + name + ')';
  var m = getComputedStyle(probe).color.match(/[\d.]+/g);
  return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
}
function mix(a, b, t) {
  return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' + Math.round(a[2] + (b[2] - a[2]) * t) + ')';
}
var TINTS = [0.14, 0.32, 0.22, 0.40, 0.18, 0.28, 0.36, 0.12, 0.26, 0.34, 0.20];

/* mesh prep: add feature edges for meshes without explicit segments */
function prep(m) {
  if (m._p) return m;
  var v = m.v, f = m.f, n = [], i;
  for (i = 0; i < f.length; i += 3) {
    var a = f[i] * 3, b = f[i + 1] * 3, c = f[i + 2] * 3;
    var ux = v[b] - v[a], uy = v[b + 1] - v[a + 1], uz = v[b + 2] - v[a + 2];
    var wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
    var nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    var l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    n.push(nx / l, ny / l, nz / l);
  }
  m._n = n;
  if (!m.seg) {
    var map = {}, seg = [];
    for (i = 0; i < f.length; i += 3) {
      for (var k = 0; k < 3; k++) {
        var p = f[i + k], q = f[i + (k + 1) % 3], key = p < q ? p + '_' + q : q + '_' + p;
        (map[key] = map[key] || []).push(i / 3);
      }
    }
    Object.keys(map).forEach(function (key) {
      var t = map[key], keep = t.length !== 2;
      if (!keep) {
        var d = n[t[0] * 3] * n[t[1] * 3] + n[t[0] * 3 + 1] * n[t[1] * 3 + 1] + n[t[0] * 3 + 2] * n[t[1] * 3 + 2];
        keep = Math.abs(d) < 0.85;
      }
      if (keep) {
        var pq = key.split('_');
        seg.push(v[pq[0] * 3], v[pq[0] * 3 + 1], v[pq[0] * 3 + 2], v[pq[1] * 3], v[pq[1] * 3 + 1], v[pq[1] * 3 + 2]);
      }
    });
    m.seg = seg;
  }
  m._p = true;
  return m;
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

var VIEW0 = { yaw: -0.65, pitch: 0.5 };
/* Tiny painter's-algorithm renderer: orthographic, z up, faces + edges depth-sorted together. */
function Viewer(canvas, view) {
  this.cv = canvas; this.view = view; this.scene = null;
}
Viewer.prototype.draw = function () {
  var cv = this.cv, sc = this.scene, w = cv.clientWidth, hgt = cv.clientHeight;
  if (!w || !hgt) return;
  var dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
  var g = cv.getContext('2d');
  if (!g) return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  var bg = rgb('--bg-color'), tx = rgb('--text-color'), mu = rgb('--mut');
  g.fillStyle = mix(bg, tx, 0); g.fillRect(0, 0, w, hgt);
  if (!sc || !sc.mesh) {
    g.fillStyle = mix(mu, mu, 0); g.font = '14px ' + (getComputedStyle(cv).fontFamily || 'sans-serif');
    g.textAlign = 'center'; g.fillText('Not solid yet: only a flat sketch', w / 2, hgt / 2);
    return;
  }
  var cy = Math.cos(this.view.yaw), sy = Math.sin(this.view.yaw), cp = Math.cos(this.view.pitch), sp = Math.sin(this.view.pitch);
  var s = Math.min(w, hgt) * 0.46 / sc.b.r, ox = w / 2, oy = hgt / 2, c = sc.b.c, off = sc.off || c;
  function pr(v, i) {
    var x = v[i] - off[0], y = v[i + 1] - off[1], z = v[i + 2] - off[2];
    var x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    return [ox + x1 * s, oy - (z * cp + y1 * sp) * s, y1 * cp - z * sp];
  }
  var items = [];
  var addEdges = function (m, ghost) {
    var e = m.seg;
    for (var i = 0; i < e.length; i += 6) {
      var a = pr(e, i), b = pr(e, i + 3);
      items.push({ t: 1, d: (a[2] + b[2]) / 2 - sc.b.r * 0.02, a: a, b: b, gh: ghost });
    }
  };
  var m = prep(sc.mesh), v = m.v, f = m.f, nn = m._n;
  for (var i = 0; i < f.length; i += 3) {
    var a = pr(v, f[i] * 3), b = pr(v, f[i + 1] * 3), cc = pr(v, f[i + 2] * 3);
    var j = i; var nx = nn[j], ny = nn[j + 1], nz = nn[j + 2];
    var vy = nx * sy + ny * cy, vz = nz * cp + vy * sp, vx = nx * cy - ny * sy, vd = vy * cp - nz * sp;
    var lam = Math.abs(-0.35 * vx + 0.55 * vz - 0.75 * vd);
    var grp = m.g ? m.g[i / 3] : 0;
    items.push({ t: 0, d: (a[2] + b[2] + cc[2]) / 3, p: [a, b, cc], k: TINTS[grp % TINTS.length] + 0.16 * (1 - Math.min(1, lam)) });
  }
  addEdges(m, false);
  if (sc.ghost) addEdges(prep(sc.ghost), true);
  items.sort(function (p, q) { return q.d - p.d; });
  g.lineJoin = 'round';
  items.forEach(function (it) {
    if (it.t === 0) {
      var col = mix(bg, tx, it.k);
      g.fillStyle = col; g.strokeStyle = col; g.lineWidth = 0.6;
      g.beginPath(); g.moveTo(it.p[0][0], it.p[0][1]); g.lineTo(it.p[1][0], it.p[1][1]); g.lineTo(it.p[2][0], it.p[2][1]);
      g.closePath(); g.fill(); g.stroke();
    } else {
      g.strokeStyle = it.gh ? mix(bg, mu, 0.8) : mix(bg, tx, 0.9);
      g.lineWidth = it.gh ? 1 : 1.3;
      g.setLineDash(it.gh ? [4, 3] : []);
      g.beginPath(); g.moveTo(it.a[0], it.a[1]); g.lineTo(it.b[0], it.b[1]); g.stroke();
    }
  });
  g.setLineDash([]);
};

/* drag + arrow keys on a canvas; all viewers sharing `view` redraw together */
function bindRotate(cv, view, redraw) {
  var drag = null;
  cv.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY]; if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', function (e) {
    if (!drag) return;
    view.yaw += (e.clientX - drag[0]) * 0.01;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY]; redraw();
  });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (n) { cv.addEventListener(n, function () { drag = null; }); });
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
function canvas3d(label, descId) {
  var cv = h('canvas', 'cd-cv');
  cv.tabIndex = 0;
  cv.setAttribute('role', 'img');
  cv.setAttribute('aria-label', label + '. Drag, or focus and use the arrow keys, to turn it.');
  if (descId) cv.setAttribute('aria-describedby', descId);
  return cv;
}
function hint(onReset) {
  var p = h('p', 'cd-hint');
  p.appendChild(h('span', null, 'Drag, or click it and use the arrow keys, to turn it.'));
  var r = btn('Reset view'); r.addEventListener('click', onReset); p.appendChild(r);
  return p;
}

RSVis.mods[ID] = function mount(el, data, ctx) {
  if (!document.getElementById('rs-vis-cad-css')) {
    var st = document.createElement('style'); st.id = 'rs-vis-cad-css'; st.textContent = CSS; document.head.appendChild(st);
  }
  var T = data.text;
  var uid = 'cd' + Math.random().toString(36).slice(2, 7);
  var root = h('div', 'cd');
  var live = h('div'); live.setAttribute('aria-live', 'polite');
  var count = h('p', 'cd-count'), title = h('h4', 'cd-title');
  live.appendChild(count); live.appendChild(title); root.appendChild(live);
  var body = h('div'); root.appendChild(body);
  var viewers = [];
  function redrawAll() { viewers.forEach(function (v) { v.draw(); }); }

  var STEPS = [];
  function step(t, build) { var p = h('div', 'cd-pane'); p.hidden = true; build(p); body.appendChild(p); STEPS.push({ t: t, el: p }); }

  step(T.titles[0], function (p) {
    var ol = h('ol', 'cd-loop');
    ol.setAttribute('aria-label', 'The repair loop');
    T.loop.forEach(function (s, i, a) {
      var li = h('li'); li.appendChild(h('span', 'cd-box' + (i === a.length - 1 ? ' again' : ''), s));
      if (i < a.length - 1) { var ar = h('span', 'cd-arr', '→'); ar.setAttribute('aria-hidden', 'true'); li.appendChild(ar); }
      ol.appendChild(li);
    });
    p.appendChild(ol);
    p.appendChild(h('p', null, T.frozen));
    p.appendChild(h('p', 'cd-note', T.gt));
  });

  var B = data.build, bview = { yaw: VIEW0.yaw, pitch: VIEW0.pitch };
  var bb = bounds(B.meshes);
  var bv;
  step(T.titles[1], function (p) {
    var grid = h('div', 'cd-v3'), left = h('div'), right = h('div', 'cd-read');
    var descId = uid + '-bd', desc = h('p', 'cd-sr'); desc.id = descId;
    var cv = canvas3d('3D view of the part at the chosen step', descId);
    bv = new Viewer(cv, bview); viewers.push(bv);
    bindRotate(cv, bview, function () { bv.draw(); });
    left.appendChild(cv); left.appendChild(desc);
    left.appendChild(hint(function () { bview.yaw = VIEW0.yaw; bview.pitch = VIEW0.pitch; bv.draw(); }));
    var op = h('p', 'op'); var rl = h('div'); rl.setAttribute('aria-live', 'polite'); rl.appendChild(op);
    var dl = h('dl', 'cd-ctr'), dd = {};
    [['faces', 'Faces'], ['edges', 'Edges'], ['vol', 'Volume']].forEach(function (k) {
      var d = h('div'); d.appendChild(h('dt', null, k[1])); dd[k[0]] = h('dd'); d.appendChild(dd[k[0]]); dl.appendChild(d);
    });
    rl.appendChild(dl);
    right.appendChild(rl);
    right.appendChild(h('p', 'cd-note', T.snap));
    right.appendChild(h('p', 'cd-note', B.label));
    var sl = h('div', 'cd-slide');
    var prev = btn('Earlier step'), nxt = btn('Later step');
    var rg = document.createElement('input'); rg.type = 'range'; rg.min = 0; rg.max = B.steps.length - 1; rg.value = 0;
    rg.setAttribute('aria-label', 'Build step');
    sl.appendChild(prev); sl.appendChild(rg); sl.appendChild(nxt);
    grid.appendChild(left); grid.appendChild(right);
    p.appendChild(grid); p.appendChild(sl);
    var last = null;
    function show(i) {
      var s = B.steps[i];
      rg.value = i; rg.setAttribute('aria-valuetext', 'Step ' + (i + 1) + ' of ' + B.steps.length);
      prev.disabled = i === 0; nxt.disabled = i === B.steps.length - 1;
      op.textContent = 'Step ' + (i + 1) + ' of ' + B.steps.length + ': ' + s.op;
      var n = s.faces != null ? s : (s.m != null ? last : null);
      if (s.faces != null) last = s;
      ['faces', 'edges', 'vol'].forEach(function (k) {
        dd[k].textContent = n ? String(n[k]) : 'not solid yet';
        dd[k].className = n ? '' : 'none';
      });
      desc.textContent = s.alt;
      bv.scene = s.m == null ? null : { mesh: B.meshes[s.m], ghost: s.ghost != null ? B.meshes[s.ghost] : null, b: bb };
      bv.draw();
    }
    var cur = 0;
    function go(i) { cur = Math.max(0, Math.min(B.steps.length - 1, i)); last = null; for (var k = 0; k < cur; k++) if (B.steps[k].faces != null) last = B.steps[k]; show(cur); }
    prev.addEventListener('click', function () { go(cur - 1); });
    nxt.addEventListener('click', function () { go(cur + 1); });
    rg.addEventListener('input', function () { go(+rg.value); });
    go(0);
  });

  var F = data.fault;
  step(T.titles[2], function (p) {
    p.appendChild(h('span', 'cd-ill', F.label));
    p.appendChild(h('p', null, F.intro));
    var ul = h('ul', 'cd-chain'), chips = [];
    ul.setAttribute('aria-label', 'Operations in order, with the verifier verdict for each');
    var det = h('div', 'cd-det'); det.setAttribute('aria-live', 'polite');
    var ctl = h('div', 'cd-nav'), pv = btn('Previous', 'cd-btn', 'Previous operation'), nx = btn('Next', 'cd-btn', 'Next operation');
    var cur = 0;
    function sel(i) {
      cur = i;
      chips.forEach(function (c, k) { c.setAttribute('aria-pressed', k === i ? 'true' : 'false'); });
      var o = F.ops[i], vd = F.verdicts[o.verdict];
      det.textContent = '';
      det.appendChild(h('strong', null, (i + 1) + '. ' + o.short + ': ' + vd.label + '. '));
      det.appendChild(document.createTextNode(o.state));
      pv.disabled = i === 0; nx.disabled = i === F.ops.length - 1;
    }
    F.ops.forEach(function (o, i) {
      var li = h('li'), b = btn(null, 'cd-chip ' + o.verdict);
      b.appendChild(h('span', 'nm', (i + 1) + '. ' + o.short));
      var vd = h('span', 'vd'); vd.appendChild(mark(o.verdict)); vd.appendChild(h('span', null, F.verdicts[o.verdict].label));
      b.appendChild(vd);
      b.addEventListener('click', function () { sel(i); });
      chips.push(b); li.appendChild(b); ul.appendChild(li);
    });
    pv.addEventListener('click', function () { sel(Math.max(0, cur - 1)); });
    nx.addEventListener('click', function () { sel(Math.min(F.ops.length - 1, cur + 1)); });
    ctl.appendChild(pv); ctl.appendChild(nx);
    p.appendChild(ul); p.appendChild(det); p.appendChild(ctl);
    var d = h('details'); d.appendChild(h('summary', null, 'What the verdicts mean'));
    var dl = h('ul', 'cd-parts');
    Object.keys(F.verdicts).forEach(function (k) { dl.appendChild(h('li', null, F.verdicts[k].label + ': ' + F.verdicts[k].meaning)); });
    d.appendChild(dl); p.appendChild(d);
    sel(4);
  });

  var R = data.repair, rview = { yaw: VIEW0.yaw, pitch: 0.45 };
  step(T.titles[3], function (p) {
    var trio = h('div', 'cd-trio');
    var ms = [R.target, R.attempt, R.repaired];
    var rad = 0;
    var bs = ms.map(function (m) { var b = bounds([m]); rad = Math.max(rad, b.r); return b; });
    var rv = [];
    function rdraw() { rv.forEach(function (v) { v.draw(); }); }
    T.views.forEach(function (t, i) {
      var fig = h('figure'), cv = canvas3d(t[1]);
      var v = new Viewer(cv, rview); v.scene = { mesh: ms[i], b: { c: bs[i].c, r: rad }, off: bs[i].c };
      rv.push(v); viewers.push(v);
      bindRotate(cv, rview, rdraw);
      fig.appendChild(cv); fig.appendChild(h('figcaption', null, t[0])); trio.appendChild(fig);
    });
    p.appendChild(trio);
    p.appendChild(hint(function () { rview.yaw = VIEW0.yaw; rview.pitch = 0.45; rdraw(); }));
    p.appendChild(h('p', 'cd-note', T.aligned));
    p.appendChild(iouChart(R.iou));
    p.appendChild(h('p', null, T.change));
    p.appendChild(h('p', 'cd-note', T.caption));
    var d = h('details'); d.appendChild(h('summary', null, 'Show the code change'));
    d.appendChild(h('p', 'cd-note', 'Before'));
    d.appendChild(h('pre', null, R.code_attempt.join('\n')));
    d.appendChild(h('p', 'cd-note', 'After'));
    d.appendChild(h('pre', null, R.code_repaired.join('\n')));
    p.appendChild(d);
  });

  function iouChart(vals) {
    var NS = 'http://www.w3.org/2000/svg';
    function s(tag, at, txt) { var e = document.createElementNS(NS, tag); for (var k in at) e.setAttribute(k, at[k]); if (txt != null) e.textContent = txt; return e; }
    var W = 360, H = 140, x0 = 30, dx = (W - 60) / (vals.length - 1);
    function X(i) { return x0 + i * dx; }
    function Y(v) { return 98 - v * 70; }
    var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'cd-iou', role: 'img',
      'aria-label': 'Overlap with the target (IoU) by turn: ' + vals.join(', ') + '.' });
    svg.appendChild(s('text', { x: 0, y: 14 }, 'overlap with the target (IoU)'));
    var d = vals.map(function (v, i) { return (i ? 'L' : 'M') + X(i) + ' ' + Y(v); }).join(' ');
    svg.appendChild(s('path', { d: d, fill: 'none', stroke: 'var(--mut)', 'stroke-width': 1.5 }));
    vals.forEach(function (v, i) {
      var lastP = i === vals.length - 1;
      svg.appendChild(s('circle', { cx: X(i), cy: Y(v), r: lastP ? 5 : 4, fill: lastP ? 'var(--ok)' : 'var(--bg-color)', stroke: lastP ? 'var(--ok)' : 'var(--text-color)', 'stroke-width': 1.5 }));
      svg.appendChild(s('text', { x: X(i), y: lastP ? Y(v) - 10 : Y(v) + 20, 'text-anchor': 'middle' }, String(v)));
    });
    svg.appendChild(s('text', { x: X(0), y: H - 8, 'text-anchor': 'middle', class: 'lb' }, 'first try'));
    svg.appendChild(s('text', { x: X(vals.length - 1), y: H - 8, 'text-anchor': 'middle', class: 'lb' }, 'fixed'));
    return svg;
  }

  step(T.titles[4], function (p) {
    var ul = h('ul', 'cd-stat');
    T.missing.forEach(function (r) {
      var li = h('li', r[0]); li.appendChild(mark(r[0]));
      var t = h('span'); t.appendChild(h('strong', null, r[2] + ': ' + r[1] + '. ')); t.appendChild(document.createTextNode(r[3]));
      li.appendChild(t); ul.appendChild(li);
    });
    p.appendChild(ul);
    p.appendChild(h('p', 'cd-note', T.cards));
    var d = h('details'); d.appendChild(h('summary', null, 'More detail on the parts'));
    var pl = h('ul', 'cd-parts');
    data.parts.forEach(function (x) {
      var li = h('li'); li.appendChild(mark(x.status)); li.appendChild(document.createTextNode(' '));
      li.appendChild(h('strong', null, x.name + ' (' + x.status_text + '): ')); li.appendChild(document.createTextNode(x.sentence));
      pl.appendChild(li);
    });
    d.appendChild(pl); p.appendChild(d);
  });

  var nav = h('div', 'cd-nav'), back = btn('Back'), next = btn('Next'), dots = h('div', 'cd-dots'), cur = 0;
  dots.setAttribute('role', 'group'); dots.setAttribute('aria-label', 'Steps');
  var dotBtns = STEPS.map(function (s, i) {
    var b = btn(String(i + 1), 'cd-dot', 'Step ' + (i + 1) + ': ' + s.t);
    b.addEventListener('click', function () { go(i); }); dots.appendChild(b); return b;
  });
  nav.appendChild(back); nav.appendChild(dots); nav.appendChild(next);
  root.appendChild(nav);
  function go(i) {
    cur = Math.max(0, Math.min(STEPS.length - 1, i));
    STEPS.forEach(function (s, k) { s.el.hidden = k !== cur; });
    dotBtns.forEach(function (b, k) { if (k === cur) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
    count.textContent = 'Step ' + (cur + 1) + ' of ' + STEPS.length;
    title.textContent = STEPS[cur].t;
    back.disabled = cur === 0; next.disabled = cur === STEPS.length - 1;
    redrawAll();
  }
  back.addEventListener('click', function () { go(cur - 1); });
  next.addEventListener('click', function () { go(cur + 1); });
  el.appendChild(root);
  go(0);

  var raf = 0;
  function later() { if (raf) return; raf = requestAnimationFrame(function () { raf = 0; redrawAll(); }); }
  window.addEventListener('resize', later);
  if (window.MutationObserver) {
    var mo = new MutationObserver(later);
    mo.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
  }
  if (ctx && ctx.reduceMotion) root.setAttribute('data-reduce-motion', 'true');
};
})();
