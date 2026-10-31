/* Resume page enhancements: role lens, evidence drawers, skill filter, bar readouts, deep links.
   The page is complete without this script. */
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
        if (raw.indexOf('card-') === 0) { state.open = raw.slice(5); return state; }
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
        var next = parts.length ? '#' + parts.join('&') : location.pathname + location.search;
        history.replaceState(null, '', next);
    }
    var state = readHash();

    /* ---- lens ---- */
    function countHidden() {
        var hidden = 0;
        document.querySelectorAll('.rs [data-lenses]').forEach(function (el) {
            if (el.offsetParent === null && el.matches('.rs-card, .rs-proj')) { hidden += 1; }
        });
        return hidden;
    }
    function applyLens(lens) {
        LENSES.forEach(function (l) { root.classList.toggle('lens-' + l, l === lens); });
        lensButtons.forEach(function (b) {
            b.setAttribute('aria-checked', b.getAttribute('data-lens') === lens ? 'true' : 'false');
            b.tabIndex = b.getAttribute('data-lens') === lens ? 0 : -1;
        });
        pdfLink.href = pdfLink.getAttribute('data-base') + lensButtons.filter(function (b) {
            return b.getAttribute('data-lens') === lens;
        })[0].getAttribute('data-pdf');
        pdfLabel.textContent = PDF_TEXT[lens];
        var n = countHidden();
        lensNote.textContent = lens === 'full' ? '' : n + ' items hidden by this lens. Negative results are never hidden.';
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

    /* ---- drawers and deep links ---- */
    var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    function openCard(id, scroll) {
        var card = document.getElementById('card-' + id);
        if (!card) { return; }
        card.classList.remove('collapsed');
        var drawer = card.querySelector('details');
        if (drawer) { drawer.open = true; }
        state.open = id;
        writeHash(state);
        if (scroll) { card.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' }); }
    }
    document.querySelectorAll('.rs-drawer').forEach(function (d) {
        d.addEventListener('toggle', function () {
            var id = d.id.replace('drawer-', '');
            if (d.open) { state.open = id; }
            else if (state.open === id) { state.open = ''; }
            writeHash(state);
        });
    });
    document.querySelectorAll('a[data-open]').forEach(function (a) {
        a.addEventListener('click', function (e) {
            e.preventDefault();
            openCard(a.getAttribute('data-open'), true);
        });
    });

    /* ---- negatives: visible by default, visitor may collapse ---- */
    var collapse = document.getElementById('rs-collapse-neg');
    collapse.addEventListener('change', function () {
        document.querySelectorAll('.rs-card.is-neg').forEach(function (c) { c.classList.toggle('collapsed', collapse.checked); });
    });

    /* ---- bar readouts ---- */
    function showBar(bar) {
        var card = bar.closest('.rs-card');
        var out = card.querySelector('.rs-readout');
        card.querySelectorAll('.rs-bar.on').forEach(function (b) { b.classList.remove('on'); });
        bar.classList.add('on');
        out.textContent = bar.getAttribute('data-group') + ' / ' + bar.getAttribute('data-label') + ': ' +
            bar.getAttribute('data-value') + ' (' + card.getAttribute('data-unit') + '). n: ' + card.getAttribute('data-n') +
            '. Baseline: ' + card.getAttribute('data-baseline') + '. Caveat: ' + card.getAttribute('data-caveat');
    }
    function clearBar(card) {
        card.querySelectorAll('.rs-bar.on').forEach(function (b) { b.classList.remove('on'); });
        card.querySelector('.rs-readout').textContent = card.querySelector('.rs-readout').getAttribute('data-default');
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

    /* ---- skill filter ---- */
    var skillBtns = Array.prototype.slice.call(document.querySelectorAll('.rs-skill'));
    var status = document.getElementById('rs-skill-status');
    var activeSkill = '';
    function applySkill(skill) {
        document.querySelectorAll('.hit').forEach(function (el) { el.classList.remove('hit'); });
        skillBtns.forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-skill') === skill ? 'true' : 'false'); });
        if (!skill) { status.textContent = ''; return; }
        var sel = '[data-skills~="' + skill + '"]';
        var cards = document.querySelectorAll('.rs-card' + sel);
        var projs = document.querySelectorAll('.rs-proj' + sel);
        var buls = document.querySelectorAll('.rs-bul li' + sel);
        document.querySelectorAll(sel).forEach(function (el) { el.classList.add('hit'); });
        status.textContent = skill + ': ' + cards.length + ' results, ' + projs.length + ' projects, ' + buls.length + ' experience bullets highlighted.';
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
    if (state.open) { openCard(state.open, true); }
    window.addEventListener('hashchange', function () {
        var s = readHash();
        if (s.lens !== state.lens) { applyLens(s.lens); }
        if (s.open && s.open !== state.open) { openCard(s.open, true); }
    });
}());
