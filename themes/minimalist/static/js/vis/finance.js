window.RSVis = window.RSVis || {mods:{}};
(function(){
  var ID = 'finance';
  var P = '.rs-vis[data-vis="finance"]';
  var CSS = [
    '@{font-family:var(--font-sans);color:var(--text-color);margin:1rem 0}',
    '@ *{box-sizing:border-box}',
    '@ button{font:inherit;font-size:14px;color:var(--text-color);background:var(--surface);border:1.5px solid var(--border-color);border-radius:6px;padding:.5rem .75rem;cursor:pointer;text-align:left;min-height:40px}',
    '@ button:hover{border-color:var(--text-color)}',
    '@ button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}',
    '@ button[aria-pressed="true"]{border-color:var(--text-color);background:color-mix(in srgb,var(--bg-color) 80%,var(--text-color));font-weight:600}',
    '@ .idea{font-size:15px;line-height:1.5;margin:0 0 .75rem}',
    '@ h4{margin:1rem 0 .5rem;font-size:15px}',
    '@ .cap{font-size:13px;line-height:1.5;margin:.5rem 0}',
    '@ .detail{font-size:14px;line-height:1.5;border-left:4px solid var(--text-color);background:var(--surface);padding:.6rem .8rem;margin:.75rem 0;min-height:3.2em}',
    '@ .st{display:inline-block;font-size:12px;line-height:1.2;padding:.15rem .45rem;border-radius:4px;border:1.5px solid var(--text-color);margin-left:.4rem;font-weight:600;vertical-align:middle}',
    '@ .st.built{background:color-mix(in srgb,var(--ok) 12%,var(--bg-color));border-color:var(--ok)}',
    '@ .st.prop{background:none;border-style:dashed}',
    '@ .st.tried{border-color:var(--bad);background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 14%,var(--bg-color)) 0 4px,var(--bg-color) 4px 8px)}',
    '@ ol{list-style:none;margin:0;padding:0}',
    '@ .ladder li{margin-bottom:.4rem}',
    '@ .ladder li button{width:calc(100% - var(--ind));margin-left:var(--ind)}',
    '@ .ladder li.nr button{border-style:dashed}',
    '@ .flow{display:flex;flex-direction:column;gap:.35rem}',
    '@ .flow li{position:relative}',
    '@ .flow li button{width:100%}',
    '@ .flow li+li::before{content:"";display:block;height:.3rem;margin-left:1.2rem;border-left:2px solid var(--border-color)}',
    '@ .flow li.side button{border-style:dashed}',
    '@ .chip{display:inline-block;margin:.6rem 0;padding:.4rem .7rem;font-size:13px;font-weight:600;border:1.5px solid var(--bad);border-radius:6px;background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--bad) 12%,var(--bg-color)) 0 4px,var(--bg-color) 4px 8px)}',
    '@media (min-width:760px){@ .flow{flex-direction:row;flex-wrap:wrap;align-items:stretch}@ .flow li{flex:1 1 150px}@ .flow li+li::before{display:none}}'
  ].join('\n').replace(/@(?!media)/g, P);

  function h(tag, cls, text){
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function badge(text, cls){ return h('span', 'st ' + cls, text); }

  RSVis.mods[ID] = function mount(el, data, ctx){
    if (!document.getElementById('rs-vis-finance-css')) {
      var st = document.createElement('style');
      st.id = 'rs-vis-finance-css';
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    el.textContent = '';
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', data.title + ': acquisition ladder and consensus flow');
    el.appendChild(h('p', 'idea', data.idea));

    var detail = h('div', 'detail', 'Select a rung or step to see what it does.');
    detail.setAttribute('aria-live', 'polite');
    var current = null;

    function item(parent, labelText, tag, tagCls, text, li){
      var b = h('button', null, labelText);
      b.type = 'button';
      if (tag) b.appendChild(badge(tag, tagCls));
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', function(){
        if (current) current.setAttribute('aria-pressed', 'false');
        current = b;
        b.setAttribute('aria-pressed', 'true');
        detail.textContent = labelText + ': ' + text;
      });
      li.appendChild(b);
      parent.appendChild(li);
    }

    el.appendChild(h('h4', null, data.ladderTitle));
    var ladder = h('ol', 'ladder');
    ladder.setAttribute('aria-label', 'Acquisition methods, tried in order');
    var n = data.rungs.length;
    data.rungs.forEach(function(r, i){
      var li = h('li', r.tag ? 'nr' : '');
      var pct = n > 1 ? Math.round(i * 10 / (n - 1)) : 0;
      li.style.setProperty('--ind', pct + '%');
      item(ladder, r.label, r.tag, 'prop', r.detail, li);
    });
    el.appendChild(ladder);
    el.appendChild(h('p', 'cap', data.ladderCaption));

    el.appendChild(h('h4', null, data.flowTitle));
    var flow = h('ol', 'flow');
    flow.setAttribute('aria-label', 'From raw disclosure to consensus');
    data.flow.forEach(function(s){
      var li = h('li', /quarantine/i.test(s.label) ? 'side' : '');
      item(flow, s.label, s.status, 'built', s.detail, li);
    });
    el.appendChild(flow);

    var after = h('ol', 'flow');
    after.setAttribute('aria-label', 'What sits on top of the consensus view');
    data.after.forEach(function(s){
      var li = h('li');
      item(after, s.label, s.status, s.status === 'Tried' ? 'tried' : 'built', s.detail, li);
    });
    el.appendChild(h('h4', null, 'On top of the consensus view'));
    el.appendChild(after);

    el.appendChild(detail);
    el.appendChild(h('div', 'chip', data.chip));
  };
})();
