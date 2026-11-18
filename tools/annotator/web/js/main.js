// Boot, location switching, keyboard, export.

import * as api from './api.js';
import { Doc } from './doc.js';
import { Renderer } from './render.js';
import { Tools, TOOLS } from './tools.js';
import { UI } from './ui.js';

class App {
  constructor() {
    this.doc = null;
    this.renderer = new Renderer(document.getElementById('canvas'), this);
    this.ui = new UI(this);
    this.tools = new Tools(this);
    this.bindKeys();
    window.addEventListener('beforeunload', (e) => { if (this.doc?.dirty.size) { this.doc.save(); e.preventDefault(); e.returnValue = ''; } });
  }

  async start() {
    this.cfg = await api.getJSON('/api/config');
    await this.refreshLocations();
    this.ui.renderToolbar();
    const last = localStorage.getItem('annot.last');
    const slug = (location.hash.slice(1) || last);
    if (slug && this.locations.some((l) => l.slug === slug)) await this.open(slug);
    else if (this.locations.length) await this.open(this.locations[0].slug);
  }

  async refreshLocations() {
    this.locations = await api.getJSON('/api/locations');
    this.ui.renderLocations();
  }

  async open(slug) {
    if (this.doc?.slug === slug) return;
    if (this.doc) { await this.doc.save(); if (this.doc.dirty.size) { this.toast('Could not save — staying on this location'); return; } }
    this.busy(`Opening ${slug}…`);
    try {
      const loc = await api.getJSON(`/api/loc/${slug}`);
      const work = await api.getJSON(`/api/loc/${slug}/work`);
      const doc = new Doc(this.cfg, loc, work);
      await doc.loadMasks();
      this.doc = doc;
      this.tools.selection.clear(); this.tools.draft = null;
      doc.addEventListener('change', (e) => this.onChange(e.detail));
      doc.addEventListener('saving', (e) => { this.ui.setSaving(e.detail); if (e.detail === 'saved') this.refreshStatusOnly(); });
      doc.addEventListener('reopened', (e) => this.toast(`Edited → ${e.detail.join(', ')} back to in_progress`));
      this.ui.restore();
      this.renderer.classCanvas.clear(); this.renderer.refCanvas.clear(); this.renderer.conflicts = null;
      this.ui.showConflicts = false;
      await this.renderer.loadImages();
      this.renderer.fit();
      this.renderer.invalidateClass();
      for (const [k, o] of Object.entries(this.ui.refOn)) if (o.on) { const [s, c] = k.split('|'); doc.refVectors(s).then(() => this.renderer.request()); doc.refMaskData(s, c).then(() => this.renderer.request()); }
      localStorage.setItem('annot.last', slug);
      history.replaceState(null, '', '#' + slug);
      this.ui.renderAll();
      this.ui.setSaving('saved');
      const missing = Object.entries(work.state.seed_missing || {});
      if (missing.length) this.toast('Seed sources missing when this workspace was created: ' + missing.map(([c, s]) => `${c}: ${s.join(', ')}`).join('; ') + '. Add them with "+ source".', 8000);
    } catch (e) {
      console.error(e); this.toast('Failed to open: ' + e.message, 8000);
    } finally { this.busy(null); }
  }

  onChange(d) {
    if (d.image) { this.renderer.request(); this.ui.renderDates(); }
    if (d.vectors) this.renderer.request();
    if (d.cls && d.rect) this.renderer.invalidateClass(d.cls, d.rect);
    else if (d.structure) this.renderer.invalidateClass(d.cls || null);
    if (d.structure || d.vectors) { clearTimeout(this._lt); this._lt = setTimeout(() => this.ui.renderLayers(), 150); }
    if (d.status) { this.ui.renderDatebar(); clearTimeout(this._dt); this._dt = setTimeout(() => this.ui.renderDates(), 150); }
    if (this.ui.showConflicts) { this.ui.showConflicts = false; this.renderer.conflicts = null; }
    this.renderer.request();
  }

  async refreshStatusOnly() {
    clearTimeout(this._st);
    this._st = setTimeout(async () => {
      try { this.locations = await api.getJSON('/api/locations'); this.ui.renderLocations(); } catch { /* offline */ }
    }, 800);
  }

  action(a) {
    const d = this.doc; if (!d) return;
    if (a === 'undo') { const c = d.undo(); this.toast(c ? 'undo: ' + c.label : 'nothing to undo', 1200); this.renderer.invalidateClass(); }
    if (a === 'redo') { const c = d.redo(); this.toast(c ? 'redo: ' + c.label : 'nothing to redo', 1200); this.renderer.invalidateClass(); }
    if (a === 'save') d.save();
    if (a === 'fit') this.renderer.fit();
    if (a === 'help') document.getElementById('help').hidden = !document.getElementById('help').hidden;
  }

  async runExport(mode) {
    if (this.doc) await this.doc.save();
    this.busy(`Exporting (${mode})…`);
    try {
      const r = await api.postJSON(`/api/export?mode=${mode}`);
      this.toast(`Exported ${r.written} samples from ${r.locations} locations in ${r.seconds}s`, 6000);
      this.ui.renderExport(true);
    } catch (e) { this.toast('Export failed: ' + e.message, 8000); } finally { this.busy(null); }
  }

  bindKeys() {
    const byKey = Object.fromEntries(Object.entries(TOOLS).map(([k, v]) => [v.key, k]));
    window.addEventListener('keydown', (e) => {
      if (!this.doc) return;
      const tag = e.target.tagName;
      if (tag === 'INPUT' && !['range', 'checkbox', 'radio'].includes(e.target.type) || tag === 'SELECT' || tag === 'TEXTAREA') return;
      const k = e.key, mod = e.metaKey || e.ctrlKey;
      if (mod && k.toLowerCase() === 'z') { e.preventDefault(); this.action(e.shiftKey ? 'redo' : 'undo'); return; }
      if (mod && k.toLowerCase() === 'y') { e.preventDefault(); this.action('redo'); return; }
      if (mod && k.toLowerCase() === 's') { e.preventDefault(); this.action('save'); return; }
      if (mod && k === 'Enter') { e.preventDefault(); this.doc.setDate(this.doc.viewDate, { status: 'done' }); this.ui.renderDates(); this.toast(`${this.doc.viewDate} done`, 1200); return; }
      if (this.tools.key(e)) { e.preventDefault(); return; }
      if (mod) return;
      if (k === ' ') { this.tools.space = true; this.renderer.canvas.style.cursor = 'grab'; e.preventDefault(); return; }
      if (k === 'x' || k === 'X') { if (this.ui.compare.dateB) { this.ui.compare.blinkOn = true; this.renderer.request(); } return; }
      if (k === ',' || k === '.') { const ds = this.doc.dates, i = ds.indexOf(this.doc.viewDate); const j = Math.max(0, Math.min(ds.length - 1, i + (k === '.' ? 1 : -1))); this.doc.viewDate = ds[j]; this.ui.afterDateChange(); return; }
      if (k === 'f' || k === 'F') { this.renderer.fit(); return; }
      if (k === '?') { this.action('help'); return; }
      if (/^[1-9]$/.test(k)) { const c = this.doc.classes[+k - 1]; if (c) { const cur = this.ui.active?.cls === c.name ? this.ui.active.sub : this.doc.sublayers(c.name)[0]?.name; this.ui.setActive(c.name, cur); } return; }
      if (k === 'Tab') {
        e.preventDefault();
        const a = this.ui.active; if (!a) return;
        const subs = this.doc.sublayers(a.cls), i = subs.findIndex((s) => s.name === a.sub);
        this.ui.setActive(a.cls, subs[(i + (e.shiftKey ? subs.length - 1 : 1)) % subs.length].name); return;
      }
      if (k === 'g' || k === 'G') { this.ui.setTarget(this.doc.target === 'base' ? 'date' : 'base'); return; }
      const tool = byKey[k.toLowerCase()];
      if (tool) this.tools.set(tool);
    });
    window.addEventListener('keyup', (e) => {
      if (e.key === ' ') { this.tools.space = false; this.tools.set(this.tools.tool); }
      if (e.key === 'x' || e.key === 'X') { this.ui.compare.blinkOn = false; this.renderer.request(); }
    });
  }

  toast(msg, ms = 3000) {
    const el = document.getElementById('toast');
    el.textContent = msg; el.hidden = false;
    clearTimeout(this._tt); this._tt = setTimeout(() => { el.hidden = true; }, ms);
  }

  busy(msg) {
    const el = document.getElementById('busy');
    el.hidden = !msg; el.textContent = msg || '';
  }
}

const app = new App();
window.app = app;           // for the console and the headless checks
app.start();
