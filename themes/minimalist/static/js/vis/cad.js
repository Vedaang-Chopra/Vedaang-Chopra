window.RSVis = window.RSVis || { mods: {} };
(function () {
  var ID = 'cad';
  var P = '.rs-vis[data-vis="cad"] ';
  var CSS = [
    P + '{font-family:var(--font-sans);color:var(--text-color);font-size:15px;line-height:1.45}',
    P + 'button{font:inherit;color:var(--text-color);cursor:pointer}',
    P + 'button:focus-visible,' + P + 'summary:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    P + '.cv-tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .9rem}',
    P + '.cv-tab{padding:.45rem .95rem;border-radius:999px;border:1px solid var(--border-color);background:var(--btn-bg);font-weight:500}',
    P + '.cv-tab[aria-selected="true"]{background:var(--text-color);color:var(--bg-color);border-color:var(--text-color)}',
    P + '.cv-lead{margin:0 0 .6rem;color:var(--mut);max-width:46rem}',
    P + '.cv-ill{display:block;width:fit-content;margin:0 0 .7rem;padding:.1rem .5rem;border:1px dashed var(--mut);border-radius:4px;color:var(--mut);font-size:13px}',
    P + '.cv-seg{display:inline-grid;grid-template-columns:auto auto;max-width:100%;border:1px solid var(--border-color);border-radius:8px;overflow:hidden;margin:0 0 .8rem}',
    P + '.cv-seg button{border:0;background:var(--btn-bg);padding:.45rem .8rem;text-align:center}',
    P + '.cv-seg button+button{border-left:1px solid var(--border-color)}',
    P + '.cv-seg button[aria-pressed="true"]{background:var(--text-color);color:var(--bg-color);font-weight:600}',
    P + '.cv-stage{position:relative}',
    P + '.cv-chain{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;align-items:stretch;gap:.35rem}',
    P + '.cv-chain li{display:flex;align-items:center;gap:.35rem}',
    P + '.cv-arrow{color:var(--mut);font-size:14px}',
    P + '.cv-node{display:flex;flex-direction:column;align-items:flex-start;gap:.15rem;min-width:7.2rem;padding:.45rem .6rem;border:1px solid var(--border-color);border-radius:6px;background:var(--bg-color);text-align:left}',
    P + '.cv-node .nm{font-weight:600;font-size:14px}',
    P + '.cv-node .vd{font-size:13px;display:flex;align-items:center;gap:.3rem}',
    P + '.cv-node[aria-current="step"]{box-shadow:0 0 0 2px var(--text-color)}',
    P + '.cv-node.correct{border-color:var(--ok)}',
    P + '.cv-node.correct .vd{color:var(--ok)}',
    P + '.cv-node.cause{border:2px solid var(--bad);background:color-mix(in srgb,var(--bad) 9%,var(--bg-color))}',
    P + '.cv-node.cause .vd{color:var(--bad);font-weight:600}',
    P + '.cv-node.inherited{border:1px dashed var(--warn)}',
    P + '.cv-node.inherited .vd{color:var(--warn)}',
    P + '.mk{display:inline-block;width:10px;height:10px;flex:none;box-sizing:border-box}',
    P + '.mk.correct{background:var(--ok);border-radius:50%}',
    P + '.mk.cause{background:var(--bad);transform:rotate(45deg)}',
    P + '.mk.inherited{border:2px dashed var(--warn);border-radius:50%}',
    P + '.mk.built{background:var(--ok);border:2px solid var(--ok)}',
    P + '.mk.partial{border:2px solid var(--warn);background:linear-gradient(90deg,var(--warn) 50%,transparent 50%)}',
    P + '.cv-score .cv-chain{filter:blur(2px);opacity:.55;pointer-events:none}',
    P + '.cv-verdict{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none}',
    P + '.cv-verdict div{background:var(--bg-color);border:2px solid var(--text-color);border-radius:8px;padding:.5rem 1.2rem;font-weight:700;font-size:18px;filter:blur(.4px)}',
    P + '.cv-ctrl{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.8rem 0 .4rem}',
    P + '.cv-ctrl button,' + P + '.cv-step{border:1px solid var(--border-color);background:var(--btn-bg);border-radius:6px;padding:.35rem .8rem}',
    P + '.cv-ctrl button:disabled{opacity:.5;cursor:default}',
    P + '.cv-detail{border:1px solid var(--border-color);border-radius:6px;background:var(--bg-color);padding:.6rem .8rem;margin:.4rem 0 0;min-height:3.2rem}',
    P + '.cv-detail p{margin:.15rem 0}',
    P + '.cv-detail .hd{font-weight:600}',
    P + '.cv-mut{color:var(--mut)}',
    P + '.cv-key{display:flex;flex-wrap:wrap;gap:.3rem 1rem;margin:.6rem 0 0;padding:0;list-style:none;font-size:13px;color:var(--mut)}',
    P + '.cv-key li{display:flex;align-items:center;gap:.35rem}',
    P + '.cv-faults{margin:.9rem 0 0}',
    P + '.cv-faults summary{cursor:pointer;color:var(--mut);font-size:14px}',
    P + '.cv-chips{display:flex;flex-wrap:wrap;gap:.35rem;margin:.5rem 0 0;padding:0;list-style:none}',
    P + '.cv-chips li{border:1px solid var(--border-color);border-radius:999px;padding:.1rem .6rem;font-size:13px;background:var(--bg-color)}',
    P + '.cv-grid{display:grid;grid-template-columns:1fr;gap:1rem}',
    '@media (min-width:760px){' + P + '.cv-grid{grid-template-columns:3fr 2fr}}',
    P + '.cv-col h4{margin:0 0 .4rem;font-size:15px}',
    P + '.cv-steps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;align-items:stretch}',
    P + '.cv-steps li{display:flex;flex-direction:column;align-items:stretch}',
    P + '.cv-down{color:var(--mut);text-align:center;font-size:14px;line-height:1.1;margin:.1rem 0}',
    P + '.cv-step{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .6rem;text-align:left;background:var(--bg-color);padding:.45rem .7rem}',
    P + '.cv-step .nm{font-weight:600;flex:1 1 9rem}',
    P + '.cv-step .st{font-size:13px;display:flex;align-items:center;gap:.3rem}',
    P + '.cv-step.built{border-left:4px solid var(--ok)}',
    P + '.cv-step.partial{border-left:4px solid var(--warn)}',
    P + '.cv-step.built .st{color:var(--ok)}',
    P + '.cv-step.partial .st{color:var(--warn)}',
    P + '.cv-step[aria-pressed="true"]{box-shadow:0 0 0 2px var(--text-color)}',
    P + '.cv-parts{display:flex;gap:.3rem;flex-basis:100%;flex-wrap:wrap}',
    P + '.cv-parts span{font-size:13px;border:1px solid var(--border-color);border-radius:4px;padding:0 .4rem;color:var(--mut)}',
    P + '.cv-back{font-size:13px;color:var(--mut);border-left:2px dashed var(--mut);margin:.4rem 0 0 .5rem;padding:.1rem 0 .1rem .6rem}',
    P + '.cv-store{border:1px solid var(--border-color);border-radius:8px;padding:.6rem;background:var(--surface)}',
    P + '.cv-store .cv-steps{gap:.4rem}',
    P + '.cv-note{margin:.7rem 0 0;font-size:14px;color:var(--mut);border-left:3px solid var(--warn);padding:.1rem 0 .1rem .6rem}',
    P + '.cv-trans .cv-node,' + P + '.cv-trans .cv-step{transition:box-shadow .15s ease,background-color .15s ease}',
    '@media (max-width:560px){' + P + '.cv-seg{display:grid;grid-template-columns:1fr 1fr}' + P + '.cv-chain{flex-direction:column;align-items:stretch}' + P + '.cv-chain li{flex-direction:column;align-items:stretch}' + P + '.cv-node{min-width:0}' + P + '.cv-arrow{text-align:center;transform:rotate(90deg)}}'
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
  function mk(kind) { return h('span', { cls: 'mk ' + kind, 'aria-hidden': 'true' }); }

  var uid = 0;

  function buildFaultView(data, root) {
    var V = data.verdicts, ops = data.ops;
    var mode = 'verifier', cur = 0;
    root.appendChild(h('p', { cls: 'cv-lead', text: data.example_intro }));
    root.appendChild(h('p', { cls: 'cv-ill', text: data.fixture_label }));

    var bScore = h('button', { type: 'button', 'aria-pressed': 'false', text: 'What a final score says' });
    var bProc = h('button', { type: 'button', 'aria-pressed': 'true', text: 'What the process verifier says' });
    root.appendChild(h('div', { cls: 'cv-seg', role: 'group', 'aria-label': 'Choose which verdict to show' }, [bScore, bProc]));

    var stage = h('div', { cls: 'cv-stage' });
    var chain = h('ol', { cls: 'cv-chain', 'aria-label': 'Operations in the program, in order' });
    var nodes = ops.map(function (op, i) {
      var vd = h('span', { cls: 'vd' }, [mk(op.verdict), h('span', { text: V[op.verdict].label })]);
      var b = h('button', { type: 'button', cls: 'cv-node' }, [h('span', { cls: 'nm', text: op.short }), vd]);
      b._vd = vd;
      b.addEventListener('click', function () { cur = i; render(); });
      var li = h('li', null, [b]);
      if (i < ops.length - 1) li.appendChild(h('span', { cls: 'cv-arrow', 'aria-hidden': 'true', text: '→' }));
      chain.appendChild(li);
      return b;
    });
    var verdict = h('div', { cls: 'cv-verdict', 'aria-hidden': 'true' }, [h('div', { text: data.score_verdict })]);
    stage.appendChild(chain);
    stage.appendChild(verdict);
    root.appendChild(stage);

    var prev = h('button', { type: 'button', text: 'Previous operation' });
    var next = h('button', { type: 'button', text: 'Next operation' });
    var ctrl = h('div', { cls: 'cv-ctrl', role: 'group', 'aria-label': 'Step through the operations' }, [prev, next]);
    root.appendChild(ctrl);
    prev.addEventListener('click', function () { if (cur > 0) { cur--; render(); } });
    next.addEventListener('click', function () { if (cur < ops.length - 1) { cur++; render(); } });

    var detail = h('div', { cls: 'cv-detail', 'aria-live': 'polite' });
    root.appendChild(detail);

    var key = h('ul', { cls: 'cv-key', 'aria-label': 'Verdict key' });
    ['correct', 'cause', 'inherited'].forEach(function (k) {
      key.appendChild(h('li', null, [mk(k), h('span', { text: V[k].label + ': ' + V[k].meaning })]));
    });
    root.appendChild(key);

    var chips = h('ul', { cls: 'cv-chips' });
    data.fault_types.forEach(function (f) { chips.appendChild(h('li', { text: f })); });
    root.appendChild(h('details', { cls: 'cv-faults' }, [h('summary', { text: 'Faults the verifier is tested on' }), chips]));

    function setMode(m) { mode = m; render(); }
    bScore.addEventListener('click', function () { setMode('score'); });
    bProc.addEventListener('click', function () { setMode('verifier'); });

    function render() {
      var score = mode === 'score';
      bScore.setAttribute('aria-pressed', String(score));
      bProc.setAttribute('aria-pressed', String(!score));
      stage.className = 'cv-stage' + (score ? ' cv-score' : '');
      verdict.style.display = score ? '' : 'none';
      ctrl.style.display = score ? 'none' : '';
      key.style.display = score ? 'none' : '';
      nodes.forEach(function (b, i) {
        var op = ops[i];
        b.className = 'cv-node' + (score ? '' : ' ' + op.verdict);
        b._vd.style.display = score ? 'none' : '';
        b.tabIndex = score ? -1 : 0;
        if (!score && i === cur) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
        b.setAttribute('aria-label', score ? op.short : op.short + ', ' + V[op.verdict].label);
      });
      prev.disabled = cur === 0;
      next.disabled = cur === ops.length - 1;
      detail.textContent = '';
      if (score) {
        detail.appendChild(h('p', { cls: 'hd', text: 'A final score gives ' + data.score_verdict.toLowerCase() + '.' }));
        detail.appendChild(h('p', { cls: 'cv-mut', text: data.score_note }));
      } else {
        var op = ops[cur], v = V[op.verdict];
        detail.appendChild(h('p', { cls: 'hd' }, [mk(op.verdict), document.createTextNode(' ' + op.short + ': ' + v.label)]));
        detail.appendChild(h('p', { text: op.what + ' ' + v.meaning }));
        detail.appendChild(h('p', { cls: 'cv-mut', text: 'Part so far: ' + op.state }));
      }
    }
    render();
  }

  function buildLoopView(data, root) {
    root.appendChild(h('p', { cls: 'cv-lead', text: 'One repair cycle. Select a step to see what it is and how far along it is.' }));
    var grid = h('div', { cls: 'cv-grid' });
    var detail = h('div', { cls: 'cv-detail', 'aria-live': 'polite' });
    var all = [];

    function stepBtn(s) {
      var kids = [h('span', { cls: 'nm', text: s.short }), h('span', { cls: 'st' }, [mk(s.status), h('span', { text: s.status_text })])];
      if (s.parts) {
        kids.push(h('span', { cls: 'cv-parts', 'aria-hidden': 'true' }, s.parts.map(function (p) { return h('span', { text: p }); })));
      }
      var b = h('button', { type: 'button', cls: 'cv-step ' + s.status, 'aria-pressed': 'false' }, kids);
      b.setAttribute('aria-label', s.short + ', ' + s.status_text + (s.parts ? ', parts: ' + s.parts.join(', ') : ''));
      b.addEventListener('click', function () { select(s, b); });
      all.push(b);
      return b;
    }
    function list(steps, label, arrows) {
      var ol = h('ol', { cls: 'cv-steps', 'aria-label': label });
      steps.forEach(function (s, i) {
        var li = h('li', null, [stepBtn(s)]);
        if (arrows && i < steps.length - 1) li.appendChild(h('span', { cls: 'cv-down', 'aria-hidden': 'true', text: '↓' }));
        ol.appendChild(li);
      });
      return ol;
    }
    function select(s, b) {
      all.forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
      detail.textContent = '';
      detail.appendChild(h('p', { cls: 'hd', text: s.short }));
      detail.appendChild(h('p', { cls: 'cv-mut' }, [document.createTextNode('Component: ' + s.component + '. Status: '), mk(s.status), document.createTextNode(' ' + s.status_text + '.')]));
      detail.appendChild(h('p', { text: s.sentence }));
    }

    var loopCol = h('div', { cls: 'cv-col' }, [
      h('h4', { text: 'The repair loop' }),
      list(data.loop_steps, 'Repair loop steps, in order', true),
      h('p', { cls: 'cv-back', text: 'Not right yet? Go back to write code with the packet, up to ' + data.max_turns + ' turns, keeping the best attempt.' })
    ]);
    var storeCol = h('div', { cls: 'cv-col' }, [
      h('h4', { text: 'Repair cards store' }),
      h('div', { cls: 'cv-store' }, [
        h('p', { cls: 'cv-mut', style: 'margin:0 0 .5rem;font-size:14px', text: 'Short reusable lessons learned from ground-truth-labelled training samples.' }),
        list(data.card_steps, 'Repair card steps', true)
      ])
    ]);
    grid.appendChild(loopCol);
    grid.appendChild(storeCol);
    root.appendChild(grid);
    root.appendChild(detail);
    var key = h('ul', { cls: 'cv-key', 'aria-label': 'Status key' }, [
      h('li', null, [mk('built'), h('span', { text: 'Built' })]),
      h('li', null, [mk('partial'), h('span', { text: 'Partial' })])
    ]);
    root.appendChild(key);
    root.appendChild(h('p', { cls: 'cv-note', text: data.loop_note }));
    select(data.loop_steps[0], all[0]);
  }

  RSVis.mods[ID] = function mount(el, data, ctx) {
    ctx = ctx || {};
    if (!document.getElementById('rs-vis-cad-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-cad-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    var n = ++uid;
    var wrap = h('div', { cls: ctx.reduceMotion ? '' : 'cv-trans' });
    var tabs = h('div', { cls: 'cv-tabs', role: 'tablist', 'aria-label': 'CAD repair: two views' });
    var names = ['Find the broken step', 'The repair loop'];
    var builders = [buildFaultView, buildLoopView];
    var tabEls = [], panels = [];
    names.forEach(function (name, i) {
      var tid = 'cad-tab-' + n + '-' + i, pid = 'cad-panel-' + n + '-' + i;
      var t = h('button', { type: 'button', role: 'tab', id: tid, 'aria-controls': pid, text: name, cls: 'cv-tab' });
      var p = h('div', { role: 'tabpanel', id: pid, 'aria-labelledby': tid, tabindex: '0' });
      builders[i](data, p);
      t.addEventListener('click', function () { show(i); });
      t.addEventListener('keydown', function (e) {
        var j = null;
        if (e.key === 'ArrowRight') j = (i + 1) % names.length;
        else if (e.key === 'ArrowLeft') j = (i - 1 + names.length) % names.length;
        else if (e.key === 'Home') j = 0;
        else if (e.key === 'End') j = names.length - 1;
        if (j !== null) { e.preventDefault(); show(j); tabEls[j].focus(); }
      });
      tabEls.push(t); panels.push(p); tabs.appendChild(t);
    });
    function show(k) {
      tabEls.forEach(function (t, i) {
        t.setAttribute('aria-selected', String(i === k));
        t.tabIndex = i === k ? 0 : -1;
        panels[i].hidden = i !== k;
      });
    }
    wrap.appendChild(tabs);
    panels.forEach(function (p) { wrap.appendChild(p); });
    el.appendChild(wrap);
    show(0);
  };
})();
