// Panels. Each section re-renders from state with innerHTML and uses one
// delegated listener -- simple, and fast enough for a few hundred rows.

import { TOOLS } from './tools.js';
import { shade, css } from './render.js';

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const STATUS_ICON = { todo: '○', in_progress: '◐', done: '●', rejected: '✕' };

function load(key, dflt) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : dflt; } catch { return dflt; } }
function save(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* private mode */ } }

export class UI {
  constructor(app) {
    this.app = app;
    this.layer = {};            // "cls" / "cls|sub" -> {vis, op, lock}
    this.refOn = {};            // "source|cls" -> {on, opacity}
    this.compare = { mode: 'off', dateB: null, opacity: 0.5, swipeX: 0.5, blinkOn: false };
    this.imageOpacity = 1;
    this.active = null;
    this.showConflicts = false;
    this.filter = load('annot.filter', { text: '', status: 'all' });
    this.bind();
  }

  get doc() { return this.app.doc; }
  get tools() { return this.app.tools; }

  // -- layer visibility model ---------------------------------------------------------

  L(key) { return (this.layer[key] ||= { vis: true, op: key.includes('|') ? 1 : 0.45, lock: false }); }
  classVisible(c) { return this.L(c).vis; }
  classOpacity(c) { return this.L(c).op; }
  subVisible(c, s) { return this.L(c).vis && this.L(c + '|' + s).vis; }
  subOpacity(c, s) { return this.L(c + '|' + s).op; }
  subLocked(c, s) { return this.L(c).lock || this.L(c + '|' + s).lock; }

  persist() {
    if (!this.doc) return;
    save('annot.view.' + this.doc.slug, { layer: this.layer, refOn: this.refOn, compare: { ...this.compare, blinkOn: false },
      active: this.active, imageOpacity: this.imageOpacity, viewDate: this.doc.viewDate });
  }

  restore() {
    const v = load('annot.view.' + this.doc.slug, null);
    this.layer = v?.layer || {}; this.refOn = v?.refOn || {};
    this.compare = { mode: 'off', dateB: null, opacity: 0.5, swipeX: 0.5, ...(v?.compare || {}), blinkOn: false };
    this.imageOpacity = v?.imageOpacity ?? 1;
    const d = this.doc;
    if (v?.viewDate && d.dates.includes(v.viewDate)) d.viewDate = v.viewDate;
    if (this.compare.dateB && !d.dates.includes(this.compare.dateB)) this.compare.dateB = null;
    const a = v?.active;
    if (a && d.sublayers(a.cls).some((s) => s.name === a.sub)) this.active = a;
    else { const c = d.classes[0]; this.active = { cls: c.name, sub: d.sublayers(c.name)[0]?.name }; }
  }

  setActive(cls, sub, render = true) {
    this.active = { cls, sub };
    this.persist();
    if (render) this.renderLayers(); else this.renderLayers();
    this.renderToolOptions();
  }

  // -- wiring ------------------------------------------------------------------------------

  bind() {
    $('#loc-filter').value = this.filter.text;
    $('#loc-status').value = this.filter.status;
    $('#loc-coll').value = this.filter.coll || 'all';
    $('#loc-coll').addEventListener('change', (e) => { this.filter.coll = e.target.value; save('annot.filter', this.filter); this.renderLocations(); });
    $('#loc-filter').addEventListener('input', (e) => { this.filter.text = e.target.value; save('annot.filter', this.filter); this.renderLocations(); });
    $('#loc-status').addEventListener('change', (e) => { this.filter.status = e.target.value; save('annot.filter', this.filter); this.renderLocations(); });
    $('#loc-list').addEventListener('click', (e) => { const li = e.target.closest('[data-slug]'); if (li) this.app.open(li.dataset.slug); });

    $('#toolbar').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.tool) this.tools.set(b.dataset.tool);
      if (b.dataset.act) this.app.action(b.dataset.act);
    });

    for (const id of ['#dates', '#target', '#layers', '#refs', '#toolopts', '#export', '#datebar']) {
      const el = $(id);
      el.addEventListener('click', (e) => this.onPanel(id, e, 'click'));
      el.addEventListener('input', (e) => this.onPanel(id, e, 'input'));
      el.addEventListener('change', (e) => this.onPanel(id, e, 'change'));
    }
    document.querySelectorAll('.panel > h3').forEach((h) => h.addEventListener('click', () => {
      h.parentElement.classList.toggle('collapsed');
      save('annot.collapsed', [...document.querySelectorAll('.panel.collapsed')].map((p) => p.id));
    }));
    for (const id of load('annot.collapsed', [])) document.getElementById(id)?.classList.add('collapsed');
  }

  onPanel(id, e, type) {
    const t = e.target, act = t.dataset.act;
    if (!act || !this.doc) return;
    // clicks: buttons, checkboxes, radios, rows. input: ranges (live).
    // change: selects and number fields.
    const isField = t.tagName === 'SELECT' || (t.tagName === 'INPUT' && !['checkbox', 'radio'].includes(t.type));
    if (type === 'click' && isField) return;
    if (type === 'input' && t.type !== 'range') return;
    if (type === 'change' && (!isField || t.type === 'range')) return;
    const d = this.doc, ds = t.dataset, r = this.app.renderer;
    const val = t.type === 'checkbox' ? t.checked : t.type === 'range' || t.type === 'number' ? +t.value : t.value;

    switch (act) {
      // dates & compare
      case 'view-date': d.viewDate = ds.date; this.afterDateChange(); break;
      case 'date-status': d.setDate(ds.date, { status: val }); this.renderDates(); break;
      case 'date-exclude': {
        const reason = val ? (prompt(`Why exclude ${ds.date}? (clouds, bad co-registration, …)`) ?? '') : '';
        d.setDate(ds.date, { excluded: !!val, reason }); this.renderDates(); break;
      }
      case 'cmp-mode': this.compare.mode = val; if (val !== 'off' && !this.compare.dateB) this.compare.dateB = d.dates.find((x) => x !== d.viewDate); this.renderDates(); r.request(); break;
      case 'cmp-b': this.compare.dateB = val || null; r.request(); break;
      case 'cmp-opacity': this.compare.opacity = val; r.request(); break;
      case 'img-opacity': this.imageOpacity = val; r.request(); break;
      case 'clear-override':
        if (confirm(`Delete every change made only on ${ds.date}? (Undo can bring it back this session.)`)) { d.deleteOverride(ds.date); r.invalidateClass(); this.renderDates(); }
        break;
      // edit target
      case 'target': this.setTarget(ds.target); break;
      // layers
      case 'cls-vis': this.L(ds.cls).vis = !this.L(ds.cls).vis; r.invalidateClass(ds.cls); this.renderLayers(); break;
      case 'cls-lock': this.L(ds.cls).lock = !this.L(ds.cls).lock; this.renderLayers(); break;
      case 'cls-op': this.L(ds.cls).op = val; r.request(); break;
      case 'sub-active': this.setActive(ds.cls, ds.sub); break;
      case 'sub-vis': { const l = this.L(ds.cls + '|' + ds.sub); l.vis = !l.vis; r.invalidateClass(ds.cls); this.renderLayers(); break; }
      case 'sub-lock': { const l = this.L(ds.cls + '|' + ds.sub); l.lock = !l.lock; this.renderLayers(); break; }
      case 'sub-solo': this.solo(ds.cls, ds.sub); break;
      case 'sub-op': this.L(ds.cls + '|' + ds.sub).op = val; clearTimeout(this._opT); this._opT = setTimeout(() => r.invalidateClass(ds.cls), 60); r.request(); break;
      case 'sub-menu': this.subMenu(ds.cls, ds.sub, t); break;
      case 'add-source': if (val) { d.addSourceSublayer(ds.cls, val).then((name) => { this.setActive(ds.cls, name); r.invalidateClass(ds.cls); }); } break;
      case 'add-manual': { const name = prompt('Name of the new sublayer', 'manual'); if (name) { const n = d.addManualSublayer(ds.cls, name.replace(/[^A-Za-z0-9_.-]/g, '-')); this.setActive(ds.cls, n); } break; }
      case 'conflicts': this.showConflicts = !this.showConflicts; if (this.showConflicts) { const n = r.computeConflicts(); this.app.toast(`${n.toLocaleString()} px claimed by 2+ classes`); } r.request(); this.renderLayers(); break;
      // refs
      case 'ref-on': { const k = ds.key; const o = (this.refOn[k] ||= { on: false, opacity: 0.6 }); o.on = !o.on; if (o.on) { const [s, c] = k.split('|'); d.refVectors(s).then(() => r.request()); d.refMaskData(s, c).then(() => r.request()); } r.request(); this.renderRefs(); break; }
      case 'ref-op': (this.refOn[ds.key] ||= { on: true, opacity: 0.6 }).opacity = val; r.request(); break;
      case 'ref-all-off': for (const o of Object.values(this.refOn)) o.on = false; r.request(); this.renderRefs(); break;
      // tool options
      case 'opt': this.tools.opt[ds.k] = t.type === 'range' || t.type === 'number' ? +t.value : t.value; this.renderToolOptionsValues(); r.request(); break;
      case 'move-to-active': this.tools.moveSelectionToActive(); break;
      case 'offset-reset': d.setImageOffset(d.target, 0, 0); this.renderToolOptions(); break;
      case 'translate-apply': {
        const u = $('#tr-unit').value, f = u === 'm' ? 1 / d.mpp : 1;
        this.tools.translateBy(+$('#tr-dx').value * f, +$('#tr-dy').value * f); break;
      }
      // export
      case 'export': this.app.runExport(ds.mode); break;
      case 'export-plan': this.renderExport(true); break;
      // datebar
      case 'mark-done': d.setDate(d.viewDate, { status: 'done' }); this.renderDates(); this.renderDatebar(); break;
      case 'mark-all-done': for (const x of d.dates) if (!d.state.dates[x].excluded) d.state.dates[x].status = 'done'; d.setDate(d.viewDate, {}); this.renderDates(); this.renderDatebar(); break;
      case 'reject': d.state.rejected = !d.state.rejected; d.dirty.add('state'); d.scheduleSave(); this.renderDatebar(); break;
      default: return;
    }
    this.persist();
  }

  afterDateChange() {
    const d = this.doc;
    if (this.compare.dateB === d.viewDate) this.compare.dateB = d.dates.find((x) => x !== d.viewDate);
    if (d.target !== 'base') d.target = d.viewDate;   // a date target follows the viewed date
    this.app.renderer.invalidateClass();
    this.renderDates(); this.renderTarget(); this.renderDatebar();
  }

  setTarget(t) {
    const d = this.doc;
    d.target = t === 'base' ? 'base' : d.viewDate;
    this.tools.selection.clear();
    document.body.classList.toggle('override-mode', d.target !== 'base');
    this.renderTarget(); this.renderToolOptions(); this.app.renderer.request();
  }

  solo(cls, sub) {
    const d = this.doc;
    const others = d.sublayers(cls).filter((s) => s.name !== sub);
    const soloed = others.every((s) => !this.L(cls + '|' + s.name).vis);
    for (const s of others) this.L(cls + '|' + s.name).vis = soloed;
    this.L(cls + '|' + sub).vis = true;
    this.app.renderer.invalidateClass(cls); this.renderLayers();
  }

  subMenu(cls, sub, anchor) {
    const d = this.doc, s = d.sublayers(cls).find((x) => x.name === sub);
    const others = d.sublayers(cls).filter((x) => x.name !== sub).map((x) => x.name);
    const menu = $('#menu');
    menu.innerHTML = [
      s?.source ? `<button data-m="reseed">Re-seed from ${esc(s.source)}</button>` : '',
      ...others.map((o) => `<button data-m="merge" data-into="${esc(o)}">Merge into ${esc(o)}</button>`),
      `<button data-m="delete" class="danger">Delete sublayer</button>`,
    ].join('');
    const rc = anchor.getBoundingClientRect();
    menu.style.left = Math.min(rc.left, window.innerWidth - 220) + 'px'; menu.style.top = rc.bottom + 'px';
    menu.hidden = false;
    const close = () => { menu.hidden = true; document.removeEventListener('mousedown', out); };
    const out = (e) => { if (!menu.contains(e.target)) close(); };
    setTimeout(() => document.addEventListener('mousedown', out));
    menu.onclick = async (e) => {
      const b = e.target.closest('button'); if (!b) return;
      close();
      const r = this.app.renderer;
      if (b.dataset.m === 'delete' && confirm(`Delete ${cls}/${sub} on every date? (Undo can bring it back this session.)`)) {
        d.deleteSublayer(cls, sub);
        if (this.active?.cls === cls && this.active?.sub === sub) this.active = { cls, sub: d.sublayers(cls)[0]?.name };
      }
      if (b.dataset.m === 'merge') { d.mergeSublayer(cls, sub, b.dataset.into); if (this.active?.sub === sub) this.active = { cls, sub: b.dataset.into }; }
      if (b.dataset.m === 'reseed' && confirm(`Replace ${cls}/${sub} with a fresh copy of ${s.source}?`)) await d.reseedSublayer(cls, sub);
      r.invalidateClass(cls); this.renderLayers(); this.persist();
    };
  }

  // -- rendering ---------------------------------------------------------------------------

  renderAll() {
    this.renderLocations(); this.renderToolbar(); this.renderDates(); this.renderTarget();
    this.renderLayers(); this.renderRefs(); this.renderToolOptions(); this.renderExport(); this.renderDatebar(); this.renderStatus();
  }

  renderLocations() {
    const list = this.app.locations || [];
    const f = this.filter.text.toLowerCase();
    const coll = this.filter.coll || 'all';
    const shown = list.filter((l) => (this.filter.status === 'all' || l.status === this.filter.status)
      && (coll === 'all' || l.collection === coll)
      && (!f || l.slug.includes(f) || (l.name || '').toLowerCase().includes(f)));
    const count = (s) => list.filter((l) => l.status === s).length;
    $('#loc-progress').textContent = `${count('done')} done · ${count('in_progress')} in progress · ${count('todo')} todo · ${count('rejected')} rejected`;
    const head = { sat: 'Satellite locations', drone: 'Drone frames (pixels only)', ortho: 'Test-area satellite (pixels only)' };
    let last = null;
    $('#loc-list').innerHTML = shown.map((l) => {
      const h = l.collection !== last ? `<li class="grp">${head[l.collection] || l.collection} · ${shown.filter((x) => x.collection === l.collection).length}</li>` : '';
      last = l.collection;
      const sub = l.collection === 'drone' ? `${esc(l.slug.replace('drone-', ''))}`
        : l.collection === 'ortho' ? esc(l.slug) : `${esc(l.slug.split('-').slice(1).join(' '))} · ${l.n_dates}d`;
      return `${h}<li data-slug="${esc(l.slug)}" class="${this.doc?.slug === l.slug ? 'sel' : ''}">
        <span class="st st-${l.status}" title="${l.status}">${STATUS_ICON[l.status]}</span>
        <span class="nm">${esc(l.name)}</span><span class="sub">${sub}</span>
      </li>`;
    }).join('');
  }

  renderToolbar() {
    const t = this.tools?.tool;
    const a = this.active && this.doc ? this.doc.classByName[this.active.cls] : null;
    $('#tools').innerHTML = Object.entries(TOOLS).map(([k, v]) => {
      const hint = a && ['rect', 'polygon', 'polyline', 'brush', 'eraser', 'wand'].includes(k) && !a.tools.includes(k) && k !== 'eraser' ? ' dim' : '';
      return `<button data-tool="${k}" class="${t === k ? 'on' : ''}${hint}" title="${esc(v.label)} [${v.key.toUpperCase()}]">${v.icon}<small>${v.key.toUpperCase()}</small></button>`;
    }).join('');
  }

  renderDates() {
    const d = this.doc; if (!d) return;
    const opts = ['todo', 'in_progress', 'done', 'rejected'];
    const rows = d.dates.map((x) => {
      const s = d.state.dates[x], ov = d.hasOverride(x);
      return `<tr class="${x === d.viewDate ? 'sel' : ''} ${s.excluded ? 'excl' : ''}">
        <td><label><input type="radio" name="vd" data-act="view-date" data-date="${x}" ${x === d.viewDate ? 'checked' : ''}> ${x}</label></td>
        <td><select data-act="date-status" data-date="${x}">${opts.map((o) => `<option ${s.status === o ? 'selected' : ''}>${o}</option>`).join('')}</select></td>
        <td title="${esc(s.reason)}"><label class="mini"><input type="checkbox" data-act="date-exclude" data-date="${x}" ${s.excluded ? 'checked' : ''}> excl</label></td>
        <td>${ov ? `<button class="mini" data-act="clear-override" data-date="${x}" title="This date has its own changes. Click to delete them.">Δ ✕</button>` : ''}${s.offset_px ? `<small title="image aligned by ${s.offset_px} px">↔${s.offset_px[0]},${s.offset_px[1]}</small>` : ''}</td>
      </tr>`;
    }).join('');
    const c = this.compare;
    $('#dates .body').innerHTML = `
      <table class="dates">${rows}</table>
      <div class="row"><label>Image</label><input type="range" min="0" max="1" step="0.05" value="${this.imageOpacity}" data-act="img-opacity"></div>
      <div class="row" ${d.dates.length < 2 ? 'hidden' : ''}><label>Compare</label>
        <select data-act="cmp-mode">${['off', 'opacity', 'swipe', 'blink'].map((m) => `<option ${c.mode === m ? 'selected' : ''}>${m}</option>`).join('')}</select>
        <select data-act="cmp-b">${d.dates.filter((x) => x !== d.viewDate).map((x) => `<option ${c.dateB === x ? 'selected' : ''}>${x}</option>`).join('')}</select>
      </div>
      ${c.mode === 'opacity' ? `<div class="row"><label>B opacity</label><input type="range" min="0" max="1" step="0.02" value="${c.opacity}" data-act="cmp-opacity"></div>` : ''}
      ${c.mode === 'blink' ? `<div class="hint">Hold <kbd>X</kbd> to show B instead of A. <kbd>,</kbd> <kbd>.</kbd> step the viewed date.</div>` : ''}
      ${c.mode === 'swipe' ? `<div class="hint">Drag the white divider. B is on the right.</div>` : ''}`;
    this.renderDatebar();
  }

  renderTarget() {
    const d = this.doc; if (!d) return;
    const base = d.target === 'base';
    if (d.dates.length < 2) {
      $('#target .body').innerHTML = '<div class="hint">One image here: every edit applies to it.</div>';
      $('#banner').hidden = true; document.body.classList.remove('override-mode');
      return;
    }
    $('#target .body').innerHTML = `
      <div class="seg">
        <button data-act="target" data-target="base" class="${base ? 'on' : ''}">BASE · all dates</button>
        <button data-act="target" data-target="date" class="${!base ? 'on warn' : ''}">only ${esc(d.viewDate)}</button>
      </div>
      <div class="hint">${base ? 'Edits change the label of every date.' : `Edits apply to <b>${esc(d.viewDate)}</b> only (new building, cut forest…). Base shapes you edit are copied into this date.`}</div>`;
    $('#banner').textContent = base ? '' : `EDITING ${d.viewDate} ONLY`;
    $('#banner').hidden = base;
    document.body.classList.toggle('override-mode', !base);
  }

  renderLayers() {
    const d = this.doc; if (!d) return;
    const srcs = Object.keys(d.refsMeta);
    const html = d.classes.map((c) => {
      const L = this.L(c.name), subs = d.sublayers(c.name);
      const feats = d.features();
      const counts = {};
      for (const f of feats) if (f.cls === c.name) counts[f.sub] = (counts[f.sub] || 0) + 1;
      const addable = srcs.filter((s) => d.refsMeta[s].classes?.[c.name]);
      return `<div class="cls ${L.vis ? '' : 'off'}">
        <div class="cls-h">
          <span class="sw" style="background:${c.color}"></span><b>${c.name}</b><small>${c.geometry}</small>
          <span class="grow"></span>
          <input type="range" min="0" max="1" step="0.05" value="${L.op}" data-act="cls-op" data-cls="${c.name}" title="class opacity">
          <button class="ic" data-act="cls-vis" data-cls="${c.name}" title="show/hide">${L.vis ? '👁' : '—'}</button>
          <button class="ic" data-act="cls-lock" data-cls="${c.name}" title="lock">${L.lock ? '🔒' : '🔓'}</button>
        </div>
        ${subs.map((s, k) => {
          const S = this.L(c.name + '|' + s.name), on = this.active?.cls === c.name && this.active?.sub === s.name;
          const col = css(shade(c.color, k, subs.length, s.name));
          return `<div class="sub ${on ? 'active' : ''} ${S.vis ? '' : 'off'}" data-act="sub-active" data-cls="${c.name}" data-sub="${esc(s.name)}">
            <span class="sw small" style="background:${col}" data-act="sub-active" data-cls="${c.name}" data-sub="${esc(s.name)}"></span>
            <span class="nm" data-act="sub-active" data-cls="${c.name}" data-sub="${esc(s.name)}">${esc(s.name)}</span>
            <small data-act="sub-active" data-cls="${c.name}" data-sub="${esc(s.name)}">${counts[s.name] ? counts[s.name] + ' shp' : ''}${d.hasRaster(c.name, s.name) ? ' ▦' : ''}</small>
            <span class="grow" data-act="sub-active" data-cls="${c.name}" data-sub="${esc(s.name)}"></span>
            <input type="range" min="0" max="1" step="0.05" value="${S.op}" data-act="sub-op" data-cls="${c.name}" data-sub="${esc(s.name)}">
            <button class="ic" data-act="sub-vis" data-cls="${c.name}" data-sub="${esc(s.name)}" title="show/hide">${S.vis ? '👁' : '—'}</button>
            <button class="ic" data-act="sub-solo" data-cls="${c.name}" data-sub="${esc(s.name)}" title="solo">S</button>
            <button class="ic" data-act="sub-lock" data-cls="${c.name}" data-sub="${esc(s.name)}" title="lock">${S.lock ? '🔒' : '🔓'}</button>
            <button class="ic" data-act="sub-menu" data-cls="${c.name}" data-sub="${esc(s.name)}" title="more">⋯</button>
          </div>`;
        }).join('')}
        <div class="sub add">
          <select data-act="add-source" data-cls="${c.name}" ${addable.length ? '' : 'hidden'}><option value="">+ source…</option>${addable.map((s) => `<option>${s}</option>`).join('')}</select>
          <button class="mini" data-act="add-manual" data-cls="${c.name}">+ empty</button>
        </div>
      </div>`;
    }).join('');
    $('#layers .body').innerHTML = html + `<div class="row"><button class="mini ${this.showConflicts ? 'on' : ''}" data-act="conflicts">${this.showConflicts ? 'Hide' : 'Show'} overlaps between classes</button></div>`;
    this.renderToolbar();
  }

  renderRefs() {
    const d = this.doc; if (!d) return;
    const rows = Object.entries(d.refsMeta).map(([src, m]) => {
      const cls = Object.keys(m.classes || {});
      return `<div class="ref"><div class="ref-h"><b>${src}</b> <small>v${m.version} · ${m.date}</small></div>
        ${cls.map((c) => {
          const k = src + '|' + c, o = this.refOn[k] || { on: false, opacity: 0.6 };
          const st = m.stats?.[c] || {};
          const n = st.features ? `${st.features} shp` : st.px ? `${(st.px / d.n * 100).toFixed(1)}%` : '';
          return `<div class="sub ${o.on ? '' : 'off'}">
            <label><input type="checkbox" data-act="ref-on" data-key="${k}" ${o.on ? 'checked' : ''}>
            <span class="sw small" style="background:${d.classByName[c]?.color || '#888'}"></span>${c}</label>
            <small>${n}</small><span class="grow"></span>
            <input type="range" min="0" max="1" step="0.05" value="${o.opacity}" data-act="ref-op" data-key="${k}">
          </div>`;
        }).join('')}</div>`;
    }).join('');
    $('#refs .body').innerHTML = (rows || (d.collection !== 'sat'
      ? '<div class="hint">No references for this image: draw every class in its <code>manual</code> sublayer.</div>'
      : '<div class="hint">No references for this location yet. Run <code>python -m tools.annotator.refs run</code>.</div>'))
      + `<div class="row"><button class="mini" data-act="ref-all-off">All overlays off</button></div>`;
  }

  renderToolOptions() {
    const t = this.tools?.tool, o = this.tools?.opt, d = this.doc;
    if (!o || !d) return;
    const m = (px) => (px * d.mpp).toFixed(1) + ' m';
    let h = '';
    if (t === 'brush' || t === 'eraser') h = `<div class="row"><label>Size</label><input type="range" min="1" max="200" value="${o.brushPx}" data-act="opt" data-k="brushPx"><span class="val" id="v-brush">${o.brushPx}px · ${m(o.brushPx)}</span></div><div class="hint"><kbd>[</kbd> <kbd>]</kbd> resize.</div>`;
    if (t === 'polyline' || t === 'polygon') h = `<div class="row"><label>Width</label><input type="number" min="0.5" max="40" step="0.5" value="${o.polylineWidthM}" data-act="opt" data-k="polylineWidthM"> m</div><div class="hint">Click points. <kbd>Enter</kbd>/double-click finishes, <kbd>Backspace</kbd> removes the last point, <kbd>Esc</kbd> cancels. On a raster class the shape fills pixels.</div>`;
    if (t === 'rect') h = `<div class="hint">Drag for an axis-aligned rectangle. Click, click, click for a rotated one: first edge, then depth.</div>`;
    if (t === 'wand') h = `
      <div class="row"><label>Tolerance</label><input type="range" min="0.01" max="0.5" step="0.01" value="${o.wandTol}" data-act="opt" data-k="wandTol"><span class="val" id="v-tol">${o.wandTol}</span></div>
      <div class="row"><label>Edges</label><input type="range" min="0" max="1" step="0.05" value="${o.wandEdge}" data-act="opt" data-k="wandEdge"><span class="val" id="v-edge">${o.wandEdge}</span></div>
      <div class="row"><label>Reach</label><input type="range" min="128" max="1536" step="64" value="${o.wandRoi}" data-act="opt" data-k="wandRoi"><span class="val" id="v-roi">${o.wandRoi}px · ${m(o.wandRoi)}</span></div>
      <div class="hint">Hover to preview, click to add to the active sublayer, <kbd>Alt</kbd>+click to erase (raster classes). On a polygon class the fill becomes a polygon.</div>
      <div class="hint" id="wand-info"></div>`;
    if (t === 'translate' && d.target !== 'base') {
      const o = d.state.dates[d.target].offset_px || [0, 0];
      h = `<div class="hint"><b>Align ${esc(d.target)}</b>: drag on the map to move this date's <b>image</b> under the layers -- the same as moving every layer on this date only. Other dates and the JPEG file are untouched.</div>
      <div class="row"><label>offset</label><span class="val">dx ${o[0]} px · dy ${o[1]} px (${(Math.hypot(o[0], o[1]) * d.mpp).toFixed(1)} m)</span><button class="mini" data-act="offset-reset">Reset</button></div>
      <div class="row"><label>nudge</label><input id="tr-dx" type="number" step="1" value="0" style="width:60px"><input id="tr-dy" type="number" step="1" value="0" style="width:60px">
        <select id="tr-unit"><option>px</option><option>m</option></select><button class="mini" data-act="translate-apply">Apply</button></div>
      <div class="hint">Tip: blink (<kbd>X</kbd>) against <code>current</code> while aligning.</div>`;
    } else if (t === 'translate') h = `
      <div class="row"><label>Scope</label><select data-act="opt" data-k="translateScope"><option value="sublayer" ${o.translateScope === 'sublayer' ? 'selected' : ''}>active sublayer</option><option value="class" ${o.translateScope === 'class' ? 'selected' : ''}>whole class</option></select></div>
      <div class="row"><label>dx</label><input id="tr-dx" type="number" step="0.5" value="0" style="width:60px"><label>dy</label><input id="tr-dy" type="number" step="0.5" value="0" style="width:60px">
        <select id="tr-unit"><option>px</option><option>m</option></select><button class="mini" data-act="translate-apply">Apply</button></div>
      <div class="hint">Drag on the map, or type an offset. Moves shapes and pixels on every date.</div>`;
    if (t === 'select') h = `<div class="hint">Click selects in the active class (<kbd>Alt</kbd>: any class). <kbd>Shift</kbd> adds. Drag handles to resize/rotate (<kbd>Shift</kbd> keeps ratio / snaps 15°). Double-click an edge adds a vertex, <kbd>Alt</kbd>+click a vertex deletes it. <kbd>Del</kbd> deletes, <kbd>⌘D</kbd> duplicates, <kbd>⌘A</kbd> selects the whole sublayer.</div>
      <div class="row"><button class="mini" data-act="move-to-active">Move selection → active sublayer</button></div>`;
    if (t === 'adopt') h = `<div class="hint">Turn a reference overlay on, then click one of its shapes or blobs: it is copied into the active sublayer (or <code>manual</code> of its class).</div>`;
    const a = this.active;
    $('#toolopts .body').innerHTML = `<div class="hint">Active: ${a ? `<b style="color:${d.classByName[a.cls].color}">${a.cls}</b> / ${esc(a.sub)}` : '—'}</div>` + h;
  }

  renderToolOptionsValues() {
    const o = this.tools.opt, d = this.doc, m = (px) => (px * d.mpp).toFixed(1) + ' m';
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('v-brush', `${o.brushPx}px · ${m(o.brushPx)}`); set('v-tol', o.wandTol); set('v-edge', o.wandEdge); set('v-roi', `${o.wandRoi}px · ${m(o.wandRoi)}`);
  }

  setWandInfo(area, touches, mpp) {
    const e = document.getElementById('wand-info');
    if (e) e.innerHTML = `${area.toLocaleString()} px · ${(area * mpp * mpp).toFixed(0)} m²${touches ? ' · <b>reaches the edge of Reach</b>, raise it or click again further along' : ''}`;
  }

  async renderExport(refresh) {
    const el = $('#export .body');
    if (refresh || !this._plan) {
      try { this._plan = await (await fetch('/api/export/plan')).json(); } catch { this._plan = null; }
    }
    const p = this._plan;
    el.innerHTML = p ? `
      <div class="hint">${p.done} done samples · <b>${p.new}</b> new · <b>${p.stale}</b> changed since export${p.needs_review.length ? ` · <b>${p.needs_review.length}</b> need review for new classes` : ''}${p.orphaned.length ? ` · ${p.orphaned.length} exported but no longer done` : ''}</div>
      <div class="row"><button class="mini" data-act="export" data-mode="new">Export new</button><button class="mini" data-act="export" data-mode="stale">Re-export changed</button><button class="mini" data-act="export" data-mode="all">Export all</button><button class="mini" data-act="export-plan">↻</button></div>`
      : '<div class="hint">export plan unavailable</div>';
  }

  renderDatebar() {
    const d = this.doc; if (!d) return;
    const s = d.state.dates[d.viewDate];
    $('#datebar').innerHTML = `
      <span class="loc">${esc(d.meta.name)} <small>${esc(d.slug)}${d.collection === 'drone' ? ` · ${d.meta.lat?.toFixed(5)}, ${d.meta.lon?.toFixed(5)} · yaw ${Math.round(d.meta.yaw_deg)}°` : d.collection === 'ortho' ? ` · ${d.meta.mpp.toFixed(2)} m/px · ${d.meta.source_shape?.[1]}×${d.meta.source_shape?.[0]} source` : ''}</small></span>
      <span class="date">${esc(d.viewDate)}</span>
      <span class="st st-${s.status}">${STATUS_ICON[s.status]} ${s.status}${s.excluded ? ' · excluded' : ''}</span>
      <button class="mini" data-act="mark-done" title="Approve this date [Ctrl+Enter]">✓ date done</button>
      ${d.dates.length > 1 ? '<button class="mini" data-act="mark-all-done" title="Approve every non-excluded date">✓ all dates done</button>' : ''}
      <button class="mini ${d.state.rejected ? 'on warn' : ''}" data-act="reject" title="Exclude the whole location from the dataset">${d.state.rejected ? 'rejected' : 'reject location'}</button>`;
  }

  renderStatus() {
    const t = this.tools;
    $('#sel-info').textContent = t?.selection.size ? `${t.selection.size} selected` : '';
  }

  setCursorInfo(x, y) {
    const d = this.doc; if (!d) return;
    const zoom = `zoom ${(this.app.renderer.view.s * 100).toFixed(0)}%`;
    if (d.grid.pixel) { $('#cursor').textContent = `${x.toFixed(0)}, ${y.toFixed(0)} px · ~${(d.mpp * 100).toFixed(1)} cm/px · ${zoom}`; return; }
    const [lat, lon] = d.grid.toLatLon(x, y);
    $('#cursor').textContent = `${x.toFixed(0)}, ${y.toFixed(0)} px · ${lat.toFixed(6)}, ${lon.toFixed(6)} · ${zoom}`;
  }

  setSaving(s) {
    const el = $('#save');
    el.textContent = s === 'saved' ? 'saved' : s === 'saving' ? 'saving…' : s === 'pending' ? 'unsaved' : s;
    el.className = s.startsWith('error') ? 'err' : s;
  }
}
