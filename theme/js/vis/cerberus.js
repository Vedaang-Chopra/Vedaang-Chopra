window.RSVis = window.RSVis || {mods:{}};
(function(){
  var NS = 'http://www.w3.org/2000/svg';

  function orderPatches(grid){
    var cells = [], sr = 0, sc = 0;
    grid.sign.forEach(function(p){ sr += p[0]; sc += p[1]; });
    sr /= grid.sign.length; sc /= grid.sign.length;
    for (var r = 0; r < grid.rows; r++) for (var c = 0; c < grid.cols; c++) {
      cells.push({r: r, c: c, d: Math.hypot(r - sr, c - sc)});
    }
    cells.sort(function(a, b){ return a.d - b.d || a.r - b.r || a.c - b.c; });
    return cells;
  }

  function lookAtLess(panel, data){
    var g = data.grid, cells = orderPatches(g), total = cells.length, S = 44;
    var W = g.cols * S, H = g.rows * S;
    panel.appendChild(RSCard.el('p', 'rc-line', 'Illustration only, not real model output. Example question: "What colour is the sign?" Patches nearest the sign are kept first.'));
    var v = RSCard.svg('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Grid of ' + total + ' image patches; kept patches are filled.'});
    v.style.maxWidth = '420px';
    var rects = cells.map(function(p){
      var r = RSCard.svg('rect', {x: p.c * S + 2, y: p.r * S + 2, width: S - 4, height: S - 4, rx: 4, stroke: 'var(--border-color)', 'stroke-width': 1});
      v.appendChild(r); return r;
    });
    panel.appendChild(v);

    var ctl = RSCard.el('div', 'rc-ctl');
    var lab = RSCard.el('label', '', 'Patches kept');
    var id = 'cb' + Math.random().toString(36).slice(2, 7);
    lab.setAttribute('for', id);
    var inp = RSCard.el('input'); inp.type = 'range'; inp.id = id; inp.min = 4; inp.max = total; inp.step = 1; inp.value = total;
    inp.style.maxWidth = '220px'; inp.style.flex = '1 1 140px';
    var out = RSCard.el('span', ''); out.setAttribute('aria-live', 'polite'); out.style.fontFamily = 'var(--font-mono)';
    ctl.appendChild(lab); ctl.appendChild(inp); ctl.appendChild(out);
    panel.appendChild(ctl);
    var cap = RSCard.el('p', 'rc-line', 'Status of the "pick parts to keep" step: Proposed. The language model would read only the filled patches; the selector is not built.');
    panel.appendChild(cap);

    function update(){
      var n = +inp.value;
      rects.forEach(function(r, i){
        r.setAttribute('fill', i < n ? 'var(--ok)' : 'var(--surface)');
        r.setAttribute('fill-opacity', i < n ? '0.55' : '1');
        r.setAttribute('stroke-dasharray', i < n ? '' : '3 3');
      });
      out.textContent = n + ' of ' + total + ' patches';
    }
    inp.addEventListener('input', update);
    update();
  }

  function realFigure(panel, data){
    var f = data.figure;
    var img = RSCard.el('img'); img.src = f.img; img.alt = f.alt; img.loading = 'lazy';
    img.style.cssText = 'display:block;width:100%;max-width:560px;height:auto;border:1px solid var(--border-color);border-radius:4px;background:#fff';
    panel.appendChild(img);
    panel.appendChild(RSCard.el('p', 'rc-line', 'Real saved results, early. Recall@5 vs vector length (128 to 4096, baseline 4096). Blue: attention pooling, kept. Orange: Perceiver, near zero, did not work. N not recorded.'));
  }

  window.RSVis.register('cerberus', function(el, data){
    RSCard.mount(el, data, {tabs: [
      {id: 'less', label: 'Look at less', render: lookAtLess},
      {id: 'real', label: 'Early real results', render: realFigure}
    ]});
  });
})();
