/* Resume page enhancements: role lens, project and evidence drawers, skill filter, bar readouts,
   comparison slider, deep links. The page is complete without this script. */
(function () {
    'use strict';
    var root = document.getElementById('rs-root');
    if (!root) { return; }

    var LENSES = ['full', 'research', 'agentic'];
    var lensButtons = Array.prototype.slice.call(document.querySelectorAll('#rs-lens button'));
    var pdfLink = document.getElementById('rs-pdf');
    var pdfLabel = document.getElementById('rs-pdf-label');
    var lensNote = document.getElementById('rs-lens-note');
    var PDF_TEXT = {
        full: 'Master Extended, 2 pages',
        research: 'Research Engineer, 1 page',
        agentic: 'Agentic Engineer, 1 page'
    };

    /* ---- hash state: #lens=research&open=cad-iou ---- */
    function readHash() {
        var state = { lens: 'full', open: '' };
        var raw = location.hash.replace(/^#/, '');
        if (raw.indexOf('card-') === 0 || raw.indexOf('proj-') === 0) {
            state.open = raw.replace(/^card-/, '');
            return state;
        }
        raw.split('&').forEach(function (part) {
            var kv = part.split('=');
            if (kv[0] === 'lens' && LENSES.indexOf(kv[1]) !== -1) { state.lens = kv[1]; }
            if (kv[0] === 'open' && /^[a-z0-9-]+$/.test(kv[1] || '')) { state.open = kv[1]; }
        });
        return state;
    }
    function writeHash(state) {
        var parts = [];
        if (state.lens !== 'full') { parts.push('lens=' + state.lens); }
        if (state.open) { parts.push('open=' + state.open); }
        history.replaceState(null, '', parts.length ? '#' + parts.join('&') : location.pathname + location.search);
    }
    var state = readHash();

    /* ---- lens ---- */
    function countHidden() {
        var hidden = 0;
        document.querySelectorAll('.rs-proj').forEach(function (el) {
            if (el.offsetParent === null) { hidden += 1; }
        });
        return hidden;
    }
    function applyLens(lens) {
        LENSES.forEach(function (l) { root.classList.toggle('lens-' + l, l === lens); });
        lensButtons.forEach(function (b) {
            var on = b.getAttribute('data-lens') === lens;
            b.setAttribute('aria-checked', on ? 'true' : 'false');
            b.tabIndex = on ? 0 : -1;
            if (on) {
                pdfLink.href = pdfLink.getAttribute('data-base') + b.getAttribute('data-pdf');
            }
        });
        pdfLabel.textContent = PDF_TEXT[lens];
        var n = countHidden();
        lensNote.textContent = lens === 'full' ? '' : n + ' projects hidden by this view. Results that did not work stay visible.';
        state.lens = lens;
        writeHash(state);
    }
    lensButtons.forEach(function (b) {
        b.addEventListener('click', function () { applyLens(b.getAttribute('data-lens')); });
        b.addEventListener('keydown', function (e) {
            var i = lensButtons.indexOf(b), d = 0;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { d = 1; }
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { d = -1; }
            if (!d) { return; }
            e.preventDefault();
            var nb = lensButtons[(i + d + lensButtons.length) % lensButtons.length];
            nb.focus();
            applyLens(nb.getAttribute('data-lens'));
        });
    });

    /* ---- projects, evidence drawers, deep links ---- */
    var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    function openTarget(id, scroll) {
        var proj = document.getElementById(id.indexOf('proj-') === 0 ? id : '');
        var card = document.getElementById('card-' + id);
        var el = proj || card;
        if (!el) { return; }
        var parent = proj || card.closest('.rs-proj');
        if (parent) { parent.open = true; }
        if (card) {
            var drawer = card.querySelector('details');
            if (drawer) { drawer.open = true; }
        }
        state.open = id;
        writeHash(state);
        if (scroll) { el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' }); }
    }
    document.querySelectorAll('.rs-proj, .rs-drawer').forEach(function (d) {
        d.addEventListener('toggle', function () {
            var id = d.id.replace(/^drawer-/, '');
            if (d.open) { state.open = id; }
            else if (state.open === id) { state.open = ''; }
            writeHash(state);
        });
    });
    document.querySelectorAll('a[data-open]').forEach(function (a) {
        a.addEventListener('click', function (e) {
            e.preventDefault();
            openTarget(a.getAttribute('data-open'), true);
        });
    });

    /* ---- negatives: visible by default, the visitor may hide them ---- */
    var hideNeg = document.getElementById('rs-collapse-neg');
    hideNeg.addEventListener('change', function () { root.classList.toggle('hide-neg', hideNeg.checked); });

    /* ---- bar readouts ---- */
    function showBar(bar) {
        var card = bar.closest('.rs-card');
        card.querySelectorAll('.rs-bar.on').forEach(function (b) { b.classList.remove('on'); });
        bar.classList.add('on');
        card.querySelector('.rs-readout').textContent = bar.getAttribute('data-group') + ', ' + bar.getAttribute('data-label') + ': ' +
            bar.getAttribute('data-value') + ' ' + card.getAttribute('data-unit') + '. Sample size: ' + card.getAttribute('data-n') +
            '. Baseline: ' + card.getAttribute('data-baseline') + '.';
    }
    function clearBar(card) {
        card.querySelectorAll('.rs-bar.on').forEach(function (b) { b.classList.remove('on'); });
        var out = card.querySelector('.rs-readout');
        out.textContent = out.getAttribute('data-default');
    }
    document.querySelectorAll('.rs-card').forEach(function (card) {
        clearBar(card);
        card.addEventListener('mouseleave', function () { clearBar(card); });
    });
    document.querySelectorAll('.rs-bar').forEach(function (bar) {
        bar.addEventListener('mouseenter', function () { showBar(bar); });
        bar.addEventListener('focus', function () { showBar(bar); });
        bar.addEventListener('click', function () { showBar(bar); });
    });

    /* ---- before/after slider ---- */
    document.querySelectorAll('[data-cmp]').forEach(function (box) {
        var input = box.querySelector('input[type=range]');
        input.addEventListener('input', function () { box.style.setProperty('--pos', input.value + '%'); });
    });

    /* ---- skill filter ---- */
    var skillBtns = Array.prototype.slice.call(document.querySelectorAll('.rs-skill'));
    var status = document.getElementById('rs-skill-status');
    var activeSkill = '';
    function applySkill(skill) {
        document.querySelectorAll('.hit').forEach(function (el) { el.classList.remove('hit'); });
        skillBtns.forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-skill') === skill ? 'true' : 'false'); });
        if (!skill) { status.textContent = ''; return; }
        var sel = '[data-skills~="' + skill + '"]';
        var projs = document.querySelectorAll('.rs-proj' + sel);
        var cards = document.querySelectorAll('.rs-card' + sel);
        var buls = document.querySelectorAll('.rs-bul li' + sel);
        document.querySelectorAll(sel).forEach(function (el) { el.classList.add('hit'); });
        status.textContent = skill + ': ' + projs.length + ' projects, ' + cards.length + ' results, ' + buls.length + ' job bullets highlighted.';
    }
    skillBtns.forEach(function (b) {
        b.addEventListener('click', function () {
            var s = b.getAttribute('data-skill');
            activeSkill = activeSkill === s ? '' : s;
            applySkill(activeSkill);
        });
    });

    /* ---- init ---- */
    applyLens(state.lens);
    if (state.open) { openTarget(state.open, true); }
    window.addEventListener('hashchange', function () {
        var s = readHash();
        if (s.lens !== state.lens) { applyLens(s.lens); }
        if (s.open && s.open !== state.open) { openTarget(s.open, true); }
    });
}());
