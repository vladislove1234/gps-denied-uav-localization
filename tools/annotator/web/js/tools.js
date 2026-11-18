// Editing tools. Every tool edits the ACTIVE sublayer (ui.active) at the
// current edit TARGET (doc.target: 'base' or a date). Polygon-shaped tools on a
// raster class fill pixels instead of creating a vector.

import * as R from './raster.js';
import { hexRgb, css } from './render.js';

export const TOOLS = {
  select: { key: 'v', label: 'Select / transform', icon: '⬚' },
  rect: { key: 'r', label: 'Rectangle (drag, or 3 clicks for rotated)', icon: '▭' },
  polygon: { key: 'p', label: 'Polygon (Enter / double-click closes)', icon: '⬠' },
  polyline: { key: 'l', label: 'Polyline with width (Enter finishes)', icon: '〰' },
  brush: { key: 'b', label: 'Brush', icon: '●' },
  eraser: { key: 'e', label: 'Eraser', icon: '◌' },
  wand: { key: 'w', label: 'Magic wand (hover previews, click fills, Alt+click erases)', icon: '✦' },
  adopt: { key: 'a', label: 'Adopt from reference (click an overlay feature/blob)', icon: '⇩' },
  translate: { key: 't', label: 'Move whole layer (BASE) / align this date\'s image (date target)', icon: '✥' },
  pan: { key: 'h', label: 'Pan', icon: '✋' },
};

const HANDLE = 7;          // screen px

export class Tools {
  constructor(app) {
    this.app = app;
    this.tool = 'select';
    this.selection = new Set();
    this.opt = { brushPx: 8, polylineWidthM: 4, wandTol: 0.12, wandEdge: 0.5, wandRoi: 512, translateScope: 'sublayer' };
    this.drag = null; this.draft = null; this.hover = null; this.mouse = null;
    this.wandPreview = null; this.wandBusy = false; this.wandQueued = null; this.adoptHover = null;
    this.worker = new Worker('js/wand-worker.js', { type: 'module' });
    this.worker.onmessage = (e) => this.onWand(e.data);
    this.wandSeq = 0;
    this.space = false;
    const c = app.renderer.canvas;
    c.addEventListener('pointerdown', (e) => this.down(e));
    c.addEventListener('pointermove', (e) => this.move(e));
    c.addEventListener('pointerup', (e) => this.up(e));
    c.addEventListener('pointerleave', () => { this.mouse = null; this.wandPreview = null; this.req(); });
    c.addEventListener('dblclick', (e) => this.dbl(e));
    c.addEventListener('wheel', (e) => { e.preventDefault(); this.app.renderer.zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015))); }, { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  get doc() { return this.app.doc; }
  get r() { return this.app.renderer; }
  get ui() { return this.app.ui; }
  req() { this.r.request(); }
  toast(m) { this.app.toast(m); }

  set(tool) {
    this.cancelDraft();
    this.tool = tool; this.wandPreview = null; this.adoptHover = null;
    this.app.renderer.canvas.style.cursor = { pan: 'grab', brush: 'none', eraser: 'none', translate: 'move' }[tool] || 'crosshair';
    this.app.ui.renderToolbar(); this.app.ui.renderToolOptions(); this.req();
  }

  active() {
    const a = this.ui.active;
    if (!a) return null;
    const cls = this.doc.classByName[a.cls];
    return { cls: a.cls, sub: a.sub, geometry: cls.geometry, tools: cls.tools };
  }

  canEdit() {
    const a = this.active();
    if (!a) { this.toast('Pick a sublayer in the Layers panel first'); return null; }
    if (this.ui.subLocked(a.cls, a.sub)) { this.toast(`${a.cls}/${a.sub} is locked`); return null; }
    if (!this.doc.sublayers(a.cls).some((s) => s.name === a.sub)) { this.toast('Active sublayer no longer exists'); return null; }
    return a;
  }

  pt(e) { return this.r.toImg(e.offsetX, e.offsetY); }

  // -- pointer --------------------------------------------------------------------------

  down(e) {
    if (!this.doc) return;
    this.app.renderer.canvas.setPointerCapture(e.pointerId);
    const [x, y] = this.pt(e);
    // swipe divider grab
    const cmp = this.ui.compare;
    if (cmp.dateB && cmp.mode === 'swipe' && Math.abs(e.offsetX - cmp.swipeX * this.r.cw) < 10 && e.button === 0) {
      this.drag = { kind: 'swipe' }; return;
    }
    if (e.button === 1 || e.button === 2 || this.space || this.tool === 'pan') {
      this.drag = { kind: 'pan', sx: e.offsetX, sy: e.offsetY, tx: this.r.view.tx, ty: this.r.view.ty };
      return;
    }
    const fn = this['down_' + this.tool];
    if (fn) fn.call(this, x, y, e);
  }

  move(e) {
    if (!this.doc) return;
    const [x, y] = this.pt(e);
    this.mouse = { x, y, sx: e.offsetX, sy: e.offsetY, alt: e.altKey, shift: e.shiftKey };
    this.app.ui.setCursorInfo(x, y);
    const d = this.drag;
    if (d?.kind === 'pan') {
      this.r.view.tx = d.tx + e.offsetX - d.sx; this.r.view.ty = d.ty + e.offsetY - d.sy; this.req(); return;
    }
    if (d?.kind === 'swipe') { this.ui.compare.swipeX = Math.max(0, Math.min(1, e.offsetX / this.r.cw)); this.req(); return; }
    const fn = this['move_' + this.tool];
    if (fn) fn.call(this, x, y, e);
    this.req();
  }

  up(e) {
    if (!this.doc) return;
    const d = this.drag;
    if (d?.kind === 'pan' || d?.kind === 'swipe') { this.drag = null; return; }
    const [x, y] = this.pt(e);
    const fn = this['up_' + this.tool];
    if (fn) fn.call(this, x, y, e);
    this.req();
  }

  dbl(e) {
    if (!this.doc) return;
    const [x, y] = this.pt(e);
    if (this.tool === 'polygon' || this.tool === 'polyline') { this.finishDraft(); return; }
    if (this.tool === 'select') this.insertVertex(x, y);
  }

  key(e) {
    const k = e.key;
    if (k === 'Escape') { this.cancelDraft(); this.selection.clear(); this.req(); return true; }
    if (k === 'Enter' && this.draft) { this.finishDraft(); return true; }
    if (k === 'Backspace' && this.draft?.pts?.length) { this.draft.pts.pop(); if (!this.draft.pts.length) this.draft = null; this.req(); return true; }
    if ((k === 'Delete' || k === 'Backspace') && this.selection.size) { this.deleteSelection(); return true; }
    if (k === '[') { this.opt.brushPx = Math.max(1, Math.round(this.opt.brushPx / 1.25)); this.ui.renderToolOptions(); this.req(); return true; }
    if (k === ']') { this.opt.brushPx = Math.min(300, Math.round(this.opt.brushPx * 1.25) + 1); this.ui.renderToolOptions(); this.req(); return true; }
    if ((e.metaKey || e.ctrlKey) && k.toLowerCase() === 'a') { this.selectAll(); return true; }
    if ((e.metaKey || e.ctrlKey) && k.toLowerCase() === 'd') { this.duplicate(); return true; }
    return false;
  }

  // -- hit testing ------------------------------------------------------------------------

  // Topmost editable feature at (x, y): the active sublayer first (OSM and MS
  // footprints overlap, and the one you are cleaning should win), then the
  // rest of the active class, then -- with `any` -- every class.
  hit(x, y, any = false) {
    const d = this.doc, s = this.r.view.s, a = this.ui.active;
    const feats = d.features();
    const tiers = [
      (f) => a && f.cls === a.cls && f.sub === a.sub,
      (f) => a && f.cls === a.cls,
      ...(any ? [() => true] : []),
    ];
    for (const tier of tiers) {
      for (let i = feats.length - 1; i >= 0; i--) {
        const f = feats[i];
        if (!tier(f)) continue;
        if (!this.ui.classVisible(f.cls) || !this.ui.subVisible(f.cls, f.sub) || this.ui.subLocked(f.cls, f.sub)) continue;
        const b = f.bbox, pad = 5 / s;
        if (x < b.x0 - pad || x > b.x1 + pad || y < b.y0 - pad || y > b.y1 + pad) continue;
        if (f.kind === 'polygon') {
          if (R.pointInPolygon(x, y, f.pts)) return f;
          if (R.distToPolyline(x, y, [...f.pts, f.pts[0]]) < 4 / s) return f;
        } else if (R.distToPolyline(x, y, f.pts) <= Math.max((f.width_m || 4) / d.mpp / 2, 4 / s)) return f;
      }
    }
    return null;
  }

  selected() {
    const all = new Map(this.doc.features().map((f) => [f.id, f]));
    const out = [];
    for (const id of [...this.selection]) { const f = all.get(id); if (f) out.push(f); else this.selection.delete(id); }
    return out;
  }

  selBox() {
    const fs = this.selected(); if (!fs.length) return null;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const f of fs) for (const [x, y] of f.pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    return { x0, y0, x1, y1 };
  }

  handles() {
    const b = this.selBox(); if (!b) return [];
    const s = this.r.view.s, mx = (b.x0 + b.x1) / 2, my = (b.y0 + b.y1) / 2;
    const hs = [
      ['nw', b.x0, b.y0], ['n', mx, b.y0], ['ne', b.x1, b.y0], ['e', b.x1, my],
      ['se', b.x1, b.y1], ['s', mx, b.y1], ['sw', b.x0, b.y1], ['w', b.x0, my],
      ['rot', mx, b.y0 - 28 / s],
    ].map(([id, x, y]) => ({ id, x, y }));
    const fs = this.selected();
    if (fs.length === 1 && s > 0.35) fs[0].pts.forEach(([x, y], k) => hs.push({ id: 'v' + k, x, y }));
    return hs;
  }

  handleAt(x, y) {
    const s = this.r.view.s;
    const hs = this.handles();
    for (let i = hs.length - 1; i >= 0; i--) if (Math.hypot(hs[i].x - x, hs[i].y - y) < HANDLE / s) return hs[i];
    return null;
  }

  // Make every selected feature editable at the current target (forks base
  // features into the date override) and return them.
  editSelection() {
    const out = [];
    const next = new Set();
    for (const f of this.selected()) {
      const e = this.doc.editable(f);
      if (e) { out.push(e); next.add(e.id); }
    }
    this.selection = next;
    return out;
  }

  // -- select -----------------------------------------------------------------------------

  down_select(x, y, e) {
    const h = this.handleAt(x, y);
    if (h && this.selection.size) {
      if (e.altKey && h.id.startsWith('v')) { this.deleteVertex(+h.id.slice(1)); return; }
      this.drag = { kind: 'handle', h, x0: x, y0: y, box: this.selBox(), started: false };
      return;
    }
    const f = this.hit(x, y, e.altKey);
    if (f) {
      if (e.shiftKey) { this.selection.has(f.id) ? this.selection.delete(f.id) : this.selection.add(f.id); this.req(); return; }
      if (!this.selection.has(f.id)) { this.selection = new Set([f.id]); }
      if (f.cls !== this.ui.active?.cls || f.sub !== this.ui.active?.sub) this.ui.setActive(f.cls, f.sub, false);
      this.drag = { kind: 'move', x0: x, y0: y, started: false };
      this.req();
      return;
    }
    this.drag = { kind: 'box', x0: x, y0: y, x1: x, y1: y, add: e.shiftKey };
    if (!e.shiftKey) this.selection.clear();
  }

  startTransform() {
    const d = this.drag;
    d.started = true;
    d.tx = this.doc.beginVectors();
    d.feats = this.editSelection();
    d.orig = d.feats.map((f) => f.pts.map((p) => p.slice()));
    if (!d.feats.length) { this.toast(this.doc.target === 'base' ? 'Date-only features are edited with that date as target' : 'Nothing editable'); d.tx.cancel(); this.drag = null; }
  }

  move_select(x, y, e) {
    const d = this.drag;
    if (!d) { this.hover = this.hit(x, y, e.altKey); return; }
    if (d.kind === 'box') { d.x1 = x; d.y1 = y; return; }
    if (!d.started) {
      if (Math.hypot(x - d.x0, y - d.y0) * this.r.view.s < 3) return;
      this.startTransform(); if (!this.drag) return;
    }
    if (d.kind === 'move') {
      const dx = x - d.x0, dy = y - d.y0;
      d.feats.forEach((f, i) => { f.pts = d.orig[i].map(([px, py]) => [px + dx, py + dy]); this.doc.touchFeature(f); });
    } else if (d.kind === 'handle') this.applyHandle(d, x, y, e);
  }

  applyHandle(d, x, y, e) {
    const b = d.box, id = d.h.id;
    if (id.startsWith('v')) {
      const k = +id.slice(1), f = d.feats[0];
      f.pts = d.orig[0].map((p) => p.slice()); f.pts[k] = [x, y]; this.doc.touchFeature(f);
      return;
    }
    if (id === 'rot') {
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
      let a = Math.atan2(y - cy, x - cx) - Math.atan2(d.y0 - cy, d.x0 - cx);
      if (e.shiftKey) a = Math.round(a / (Math.PI / 12)) * (Math.PI / 12);
      const c = Math.cos(a), s = Math.sin(a);
      d.feats.forEach((f, i) => { f.pts = d.orig[i].map(([px, py]) => [cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c]); this.doc.touchFeature(f); });
      d.angle = a;
      return;
    }
    // scale about the opposite handle
    const ax = id.includes('w') ? b.x1 : id.includes('e') ? b.x0 : null;
    const ay = id.includes('n') ? b.y1 : id.includes('s') ? b.y0 : null;
    const fx = ax === null ? 1 : (x - ax) / ((id.includes('w') ? b.x0 : b.x1) - ax || 1e-9);
    const fy = ay === null ? 1 : (y - ay) / ((id.includes('n') ? b.y0 : b.y1) - ay || 1e-9);
    let sx = fx, sy = fy;
    if (e.shiftKey && ax !== null && ay !== null) { const m = Math.max(Math.abs(fx), Math.abs(fy)); sx = Math.sign(fx) * m; sy = Math.sign(fy) * m; }
    d.feats.forEach((f, i) => {
      f.pts = d.orig[i].map(([px, py]) => [ax === null ? px : ax + (px - ax) * sx, ay === null ? py : ay + (py - ay) * sy]);
      this.doc.touchFeature(f);
    });
  }

  up_select() {
    const d = this.drag; this.drag = null;
    if (!d) return;
    if (d.kind === 'box') {
      const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1), y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
      if ((x1 - x0) * this.r.view.s < 3) return;
      const a = this.ui.active;
      for (const f of this.doc.features()) {
        if (a && f.cls !== a.cls) continue;
        if (!this.ui.subVisible(f.cls, f.sub) || this.ui.subLocked(f.cls, f.sub)) continue;
        const b = f.bbox;
        if (b.x0 >= x0 && b.x1 <= x1 && b.y0 >= y0 && b.y1 <= y1) this.selection.add(f.id);
      }
      this.ui.renderStatus();
      return;
    }
    if (d.started && d.tx) d.tx.commit(d.kind === 'move' ? 'move' : d.h.id === 'rot' ? 'rotate' : d.h.id.startsWith('v') ? 'move vertex' : 'resize', d.feats[0]?.cls);
  }

  insertVertex(x, y) {
    const fs = this.selected(); if (fs.length !== 1) return;
    const f0 = fs[0], s = this.r.view.s;
    const ring = f0.kind === 'polygon' ? [...f0.pts, f0.pts[0]] : f0.pts;
    let best = -1, bd = 8 / s;
    for (let i = 0; i < ring.length - 1; i++) { const dd = R.distToSegment(x, y, ...ring[i], ...ring[i + 1]); if (dd < bd) { bd = dd; best = i; } }
    if (best < 0) return;
    const tx = this.doc.beginVectors();
    const [f] = this.editSelection();
    if (!f) { tx.cancel(); return; }
    f.pts.splice(best + 1, 0, [x, y]); this.doc.touchFeature(f);
    tx.commit('add vertex', f.cls);
  }

  deleteVertex(k) {
    const tx = this.doc.beginVectors();
    const [f] = this.editSelection();
    if (!f || f.pts.length <= (f.kind === 'polygon' ? 3 : 2)) { tx.cancel(); return; }
    f.pts.splice(k, 1); this.doc.touchFeature(f);
    tx.commit('delete vertex', f.cls);
  }

  deleteSelection() {
    const fs = this.selected(); if (!fs.length) return;
    const cls = fs[0].cls;
    this.doc.vectorEdit(`delete ${fs.length}`, () => { for (const f of fs) this.doc.deleteFeature(f); }, cls);
    this.selection.clear(); this.ui.renderStatus();
  }

  selectAll() {
    const a = this.ui.active; if (!a) return;
    this.selection = new Set(this.doc.features().filter((f) => f.cls === a.cls && f.sub === a.sub).map((f) => f.id));
    this.ui.renderStatus(); this.req();
  }

  duplicate() {
    const fs = this.selected(); if (!fs.length) return;
    const off = 10 / this.r.view.s;
    const made = this.doc.vectorEdit('duplicate', () => fs.map((f) => this.doc.addFeature({
      id: null, cls: f.cls, sub: f.sub, kind: f.kind, width_m: f.width_m, props: { copy_of: f.id },
      pts: f.pts.map(([x, y]) => [x + off, y + off]) })), fs[0].cls);
    this.selection = new Set(made.map((f) => f.id)); this.ui.renderStatus();
  }

  // Move selected features into the active sublayer (same class).
  moveSelectionToActive() {
    const a = this.canEdit(); if (!a) return;
    const fs = this.selected().filter((f) => f.cls === a.cls && f.sub !== a.sub);
    if (!fs.length) { this.toast('Select features of the active class first'); return; }
    this.doc.vectorEdit(`move ${fs.length} to ${a.sub}`, () => {
      const next = new Set();
      for (const f of fs) { const e = this.doc.editable(f); if (e) { e.sub = a.sub; next.add(e.id); } }
      this.selection = next;
    }, a.cls);
  }

  // -- shapes: rect / polygon / polyline --------------------------------------------------------

  down_rect(x, y) {
    if (!this.canEdit()) return;
    if (this.draft?.kind === 'rect3') {
      const dr = this.draft;
      if (dr.pts.length === 1) { dr.pts.push([x, y]); return; }
      this.commitShape(rect3(dr.pts[0], dr.pts[1], [x, y]), 'polygon'); this.draft = null; return;
    }
    this.drag = { kind: 'rect', x0: x, y0: y, x1: x, y1: y };
  }
  move_rect(x, y) {
    if (this.drag?.kind === 'rect') { this.drag.x1 = x; this.drag.y1 = y; }
    if (this.draft?.kind === 'rect3') this.draft.cursor = [x, y];
  }
  up_rect(x, y) {
    const d = this.drag; this.drag = null;
    if (!d) return;
    if (Math.hypot(x - d.x0, y - d.y0) * this.r.view.s < 4) { this.draft = { kind: 'rect3', pts: [[d.x0, d.y0]], cursor: [x, y] }; return; }
    const pts = [[d.x0, d.y0], [x, d.y0], [x, y], [d.x0, y]];
    this.commitShape(pts, 'polygon');
  }

  down_polygon(x, y) {
    if (!this.canEdit()) return;
    const s = this.r.view.s;
    if (!this.draft) this.draft = { kind: 'polygon', pts: [] };
    const p = this.draft.pts;
    if (p.length >= 3 && Math.hypot(x - p[0][0], y - p[0][1]) < 8 / s) { this.finishDraft(); return; }
    p.push([x, y]);
  }
  move_polygon(x, y) { if (this.draft) this.draft.cursor = [x, y]; }

  down_polyline(x, y) {
    const a = this.canEdit(); if (!a) return;
    if (a.geometry === 'vector') { this.toast(`${a.cls} is a polygon class; polyline draws on raster classes`); return; }
    if (!this.draft) this.draft = { kind: 'polyline', pts: [] };
    this.draft.pts.push([x, y]);
  }
  move_polyline(x, y) { if (this.draft) this.draft.cursor = [x, y]; }

  finishDraft() {
    const dr = this.draft; if (!dr) return;
    this.draft = null;
    // a double-click adds the same point twice
    const pts = dr.pts.filter((p, i) => !i || Math.hypot(p[0] - dr.pts[i - 1][0], p[1] - dr.pts[i - 1][1]) > 0.5);
    if (dr.kind === 'polygon' && pts.length >= 3) this.commitShape(pts, 'polygon');
    else if (dr.kind === 'polyline' && pts.length >= 2) this.commitShape(pts, 'polyline');
    this.req();
  }
  cancelDraft() { this.draft = null; this.drag = null; this.req(); }

  // A finished shape becomes a vector feature, or pixels on a raster class
  // without a vector tool for it (polygon on woody = filled area).
  commitShape(pts, kind) {
    const a = this.canEdit(); if (!a) return;
    const asVector = a.geometry === 'vector' || (kind === 'polyline' && a.tools.includes('polyline'));
    if (kind === 'polygon' && R.polygonArea(pts) < 2) return;
    if (asVector) {
      const f = this.doc.vectorEdit(`draw ${kind}`, () => this.doc.addFeature({
        id: null, cls: a.cls, sub: a.sub, kind, pts, props: {},
        width_m: kind === 'polyline' ? this.opt.polylineWidthM : null }), a.cls);
      this.selection = new Set([f.id]);
      return;
    }
    const d = this.doc, ed = d.beginRaster(a.cls, a.sub, `fill ${kind}`);
    if (kind === 'polygon') ed.grow(R.fillPolygon(pts, d.w, d.h, (i) => ed.set(i, 1)));
    else {
      const r = this.opt.polylineWidthM / d.mpp / 2, scratch = null;
      for (let i = 0; i < pts.length - 1; i++) ed.grow(R.strokeSegment(scratch, d.w, d.h, ...pts[i], ...pts[i + 1], r, 1, (j) => ed.set(j, 1)));
    }
    const rect = ed.end();
    if (rect) this.r.invalidateClass(a.cls, rect);
  }

  // -- brush / eraser ---------------------------------------------------------------------------

  down_brush(x, y) { this.paintStart(x, y, 1); }
  down_eraser(x, y) { this.paintStart(x, y, 0); }
  paintStart(x, y, v) {
    const a = this.canEdit(); if (!a) return;
    const d = this.doc, ed = d.beginRaster(a.cls, a.sub, v ? 'brush' : 'erase');
    const r = this.opt.brushPx / 2;
    const box = R.stamp(null, d.w, d.h, x, y, r, v, (i) => ed.set(i, v));
    ed.grow(box);
    this.drag = { kind: 'paint', ed, v, x, y, cls: a.cls };
    this.r.invalidateClass(a.cls, box);
  }
  move_brush(x, y) { this.paintMove(x, y); }
  move_eraser(x, y) { this.paintMove(x, y); }
  paintMove(x, y) {
    const d = this.drag; if (d?.kind !== 'paint') return;
    const doc = this.doc, r = this.opt.brushPx / 2;
    const box = R.strokeSegment(null, doc.w, doc.h, d.x, d.y, x, y, r, d.v, (i) => d.ed.set(i, d.v));
    d.ed.grow(box); d.x = x; d.y = y;
    this.r.invalidateClass(d.cls, box);
  }
  up_brush() { this.paintEnd(); }
  up_eraser() { this.paintEnd(); }
  paintEnd() {
    const d = this.drag; this.drag = null;
    if (d?.kind === 'paint') { const rect = d.ed.end(); if (rect) this.r.invalidateClass(d.cls, rect); }
  }

  // -- wand -------------------------------------------------------------------------------------

  wandRequest(x, y, then) {
    const d = this.doc, size = this.opt.wandRoi;
    const ix = Math.floor(x), iy = Math.floor(y);
    if (ix < 0 || iy < 0 || ix >= d.w || iy >= d.h) return;
    const roi = R.clipRect({ x: ix - size / 2, y: iy - size / 2, w: size, h: size }, d.w, d.h);
    const date = this.ui.compare.blinkOn && this.ui.compare.dateB ? this.ui.compare.dateB : d.viewDate;
    const img = this.r.pixelsOf(date, roi);
    if (!img) return;
    const id = ++this.wandSeq;
    this.wandBusy = true;
    this.wandThen = then ? { id, fn: then } : this.wandThen;
    this.worker.postMessage({ id, rgba: img.data, w: roi.w, h: roi.h, seed: [ix - roi.x, iy - roi.y],
      tol: this.opt.wandTol, edge: this.opt.wandEdge, roi }, [img.data.buffer]);
  }

  onWand(res) {
    this.wandBusy = false;
    const a = this.active();
    const col = a ? hexRgb(this.doc.classByName[a.cls].color) : [255, 255, 255];
    const c = new OffscreenCanvas(res.roi.w, res.roi.h), g = c.getContext('2d');
    const img = g.createImageData(res.roi.w, res.roi.h);
    for (let i = 0; i < res.mask.length; i++) if (res.mask[i]) { img.data[i * 4] = col[0]; img.data[i * 4 + 1] = col[1]; img.data[i * 4 + 2] = col[2]; img.data[i * 4 + 3] = 150; }
    g.putImageData(img, 0, 0);
    this.wandPreview = { ...res, canvas: c };
    this.ui.setWandInfo(res.area, res.touchesBorder, this.doc.mpp);
    if (this.wandThen && this.wandThen.id === res.id) { const fn = this.wandThen.fn; this.wandThen = null; fn(res); }
    else if (this.wandQueued) { const q = this.wandQueued; this.wandQueued = null; this.wandRequest(q.x, q.y); }
    this.req();
  }

  move_wand(x, y) {
    if (this.wandBusy) { this.wandQueued = { x, y }; return; }
    this.wandRequest(x, y);
  }

  down_wand(x, y, e) {
    const a = this.canEdit(); if (!a) return;
    const erase = e.altKey;
    this.wandQueued = null;
    this.wandRequest(x, y, (res) => this.commitRegion(res.mask, res.roi, erase, 'wand'));
  }

  // Put a region (mask over roi) into the active sublayer: as a traced
  // polygon on a vector class, as pixels on a raster class.
  commitRegion(mask, roi, erase, label) {
    const a = this.canEdit(); if (!a) return;
    const d = this.doc;
    if (a.geometry === 'vector' && !erase) {
      // 0.6 m on satellite tiles (~1.6 px); capped at 3 px so fine drone frames keep their shape
      const ring = R.simplify(R.traceOuter(mask, roi.w, roi.h), Math.min(0.5 / d.mpp * 1.2, 3));
      if (ring.length < 3) return;
      const pts = ring.map(([x, y]) => [x + roi.x, y + roi.y]);
      const f = d.vectorEdit(label, () => d.addFeature({ id: null, cls: a.cls, sub: a.sub, kind: 'polygon', pts, props: { tool: label } }), a.cls);
      this.selection = new Set([f.id]);
      return;
    }
    if (a.geometry === 'vector' && erase) { this.toast('Alt+click erase works on raster classes; delete polygons with Select'); return; }
    const ed = d.beginRaster(a.cls, a.sub, erase ? `${label} erase` : label);
    for (let y = 0; y < roi.h; y++) for (let x = 0; x < roi.w; x++) if (mask[y * roi.w + x]) ed.set((roi.y + y) * d.w + roi.x + x, erase ? 0 : 1);
    ed.grow(roi);
    const rect = ed.end();
    if (rect) this.r.invalidateClass(a.cls, rect);
  }

  // -- adopt from reference -----------------------------------------------------------------------

  refHit(x, y) {
    const d = this.doc, s = this.r.view.s;
    const keys = Object.entries(this.ui.refOn).filter(([, o]) => o.on).map(([k]) => k).reverse();
    for (const key of keys) {
      const [source, cls] = key.split('|');
      const vec = d.refVec[source];
      if (vec) for (let i = vec.length - 1; i >= 0; i--) {
        const f = vec[i]; if (f.cls !== cls) continue;
        const b = f.bbox; if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
        if (f.kind === 'polygon' ? R.pointInPolygon(x, y, f.pts) : R.distToPolyline(x, y, f.pts) <= Math.max((f.width_m || 4) / d.mpp / 2, 4 / s)) return { type: 'vector', f, source, cls };
      }
      const m = d.refMask[source + '|' + cls];
      const ix = Math.floor(x), iy = Math.floor(y);
      if (m && ix >= 0 && iy >= 0 && ix < d.w && iy < d.h && m[iy * d.w + ix]) return { type: 'raster', source, cls, ix, iy, m };
    }
    return null;
  }

  move_adopt(x, y) { this.adoptHover = this.refHit(x, y); }

  down_adopt(x, y) {
    const h = this.refHit(x, y);
    if (!h) { this.toast('Nothing from a visible reference overlay here'); return; }
    const d = this.doc;
    const a = this.ui.active;
    let sub = a && a.cls === h.cls ? a.sub : 'manual';
    if (!d.sublayers(h.cls).some((s) => s.name === sub)) sub = d.addManualSublayer(h.cls, 'manual');
    if (this.ui.subLocked(h.cls, sub)) { this.toast(`${h.cls}/${sub} is locked`); return; }
    const geometry = d.classByName[h.cls].geometry;
    if (h.type === 'vector') {
      const f = h.f;
      if (geometry === 'vector' || (f.kind === 'polyline' && d.classByName[h.cls].tools.includes('polyline'))) {
        const made = d.vectorEdit(`adopt ${h.source}`, () => d.addFeature({ id: null, cls: h.cls, sub, kind: f.kind, width_m: f.width_m,
          pts: f.pts.map((p) => p.slice()), props: { ...f.props, ref: `${h.source}:${f.id}` } }), h.cls);
        this.selection = new Set([made.id]);
      } else {
        const ed = d.beginRaster(h.cls, sub, `adopt ${h.source}`);
        if (f.kind === 'polygon') ed.grow(R.fillPolygon(f.pts, d.w, d.h, (i) => ed.set(i, 1)));
        else for (let i = 0; i < f.pts.length - 1; i++) ed.grow(R.strokeSegment(null, d.w, d.h, ...f.pts[i], ...f.pts[i + 1], (f.width_m || 4) / d.mpp / 2, 1, (j) => ed.set(j, 1)));
        const rect = ed.end(); if (rect) this.r.invalidateClass(h.cls, rect);
      }
    } else {
      const comp = R.component(h.m, d.w, d.h, h.ix, h.iy);
      const roi = comp.box, sub_ = R.readRect(comp.mask, d.w, roi);
      const prev = this.ui.active;
      this.ui.active = { cls: h.cls, sub };
      this.commitRegion(sub_, roi, false, `adopt ${h.source}`);
      this.ui.active = prev;
    }
    this.toast(`adopted into ${h.cls}/${sub}`);
  }

  // -- translate whole layer ------------------------------------------------------------------------

  down_translate(x, y) {
    // Date target: align THAT DATE'S IMAGE under the layers instead. Same
    // effect for that date as moving every layer, without touching the others.
    if (this.doc.target !== 'base') {
      const date = this.doc.target, o = this.r.offsetOf(date);
      this.imageDrag = { date, x0: x, y0: y, start: o, off: o.slice() };
      return;
    }
    const a = this.canEdit(); if (!a) return;
    this.drag = { kind: 'translate', x0: x, y0: y, dx: 0, dy: 0, cls: a.cls,
      subs: this.opt.translateScope === 'class' ? this.doc.sublayers(a.cls).map((s) => s.name) : [a.sub] };
  }
  move_translate(x, y) {
    const im = this.imageDrag;
    if (im) { im.off = [Math.round(im.start[0] + x - im.x0), Math.round(im.start[1] + y - im.y0)]; return; }
    const d = this.drag; if (d?.kind === 'translate') { d.dx = x - d.x0; d.dy = y - d.y0; }
  }
  up_translate() {
    const im = this.imageDrag;
    if (im) { this.imageDrag = null; this.doc.setImageOffset(im.date, ...im.off); this.ui.renderToolOptions(); return; }
    const d = this.drag; this.drag = null;
    if (d?.kind !== 'translate') return;
    this.doc.translate(d.cls, d.subs, d.dx, d.dy);
    this.r.invalidateClass(d.cls);
  }
  translateBy(dx, dy) {
    if (this.doc.target !== 'base') {
      const date = this.doc.target, o = this.r.offsetOf(date);
      this.doc.setImageOffset(date, o[0] + dx, o[1] + dy); this.ui.renderToolOptions(); return;
    }
    const a = this.canEdit(); if (!a) return;
    const subs = this.opt.translateScope === 'class' ? this.doc.sublayers(a.cls).map((s) => s.name) : [a.sub];
    this.doc.translate(a.cls, subs, dx, dy);
    this.r.invalidateClass(a.cls);
  }

  // -- overlay drawing ---------------------------------------------------------------------------------

  drawOverlay(g, s) {
    const d = this.doc; if (!d) return;
    const a = this.active();
    const col = a ? hexRgb(d.classByName[a.cls].color) : [255, 255, 255];
    const m = this.mouse;

    // handles
    if (this.tool === 'select' && this.selection.size) {
      const b = this.selBox();
      if (b) {
        g.strokeStyle = 'rgba(255,255,255,.7)'; g.lineWidth = 1 / s; g.setLineDash([3 / s, 3 / s]);
        g.strokeRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0); g.setLineDash([]);
        g.beginPath(); g.moveTo((b.x0 + b.x1) / 2, b.y0); g.lineTo((b.x0 + b.x1) / 2, b.y0 - 28 / s); g.stroke();
      }
      for (const h of this.handles()) {
        const r = (h.id.startsWith('v') ? 4 : 5) / s;
        g.fillStyle = h.id === 'rot' ? '#ffd400' : h.id.startsWith('v') ? '#00e5ff' : '#fff';
        g.strokeStyle = '#000'; g.lineWidth = 1 / s;
        g.beginPath();
        if (h.id.startsWith('v') || h.id === 'rot') g.arc(h.x, h.y, r, 0, 7); else g.rect(h.x - r, h.y - r, 2 * r, 2 * r);
        g.fill(); g.stroke();
      }
    }
    if (this.tool === 'select' && this.hover && !this.drag && !this.selection.has(this.hover.id)) {
      const f = this.hover;
      g.strokeStyle = 'rgba(255,255,255,.8)'; g.lineWidth = 1.5 / s; g.beginPath();
      f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); if (f.kind === 'polygon') g.closePath(); g.stroke();
    }
    const dr = this.drag;
    if (dr?.kind === 'box') {
      g.strokeStyle = '#fff'; g.lineWidth = 1 / s; g.setLineDash([4 / s, 3 / s]);
      g.strokeRect(dr.x0, dr.y0, dr.x1 - dr.x0, dr.y1 - dr.y0); g.setLineDash([]);
    }
    if (dr?.kind === 'rect') {
      g.fillStyle = css(col, 0.3); g.strokeStyle = css(col, 1); g.lineWidth = 1.5 / s;
      g.fillRect(dr.x0, dr.y0, dr.x1 - dr.x0, dr.y1 - dr.y0); g.strokeRect(dr.x0, dr.y0, dr.x1 - dr.x0, dr.y1 - dr.y0);
    }
    const df = this.draft;
    if (df) {
      let pts = df.pts.slice();
      if (df.kind === 'rect3') pts = pts.length === 1 ? [pts[0], df.cursor || pts[0]] : rect3(pts[0], pts[1], df.cursor || pts[1]);
      else if (df.cursor) pts.push(df.cursor);
      g.strokeStyle = css(col, 1); g.fillStyle = css(col, 0.25);
      g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      if (df.kind === 'polyline') {
        g.lineCap = 'round'; g.lineJoin = 'round';
        g.lineWidth = this.opt.polylineWidthM / d.mpp; g.strokeStyle = css(col, 0.4); g.stroke();
        g.lineWidth = 1.5 / s; g.strokeStyle = css(col, 1); g.stroke();
      } else { if (df.kind !== 'rect3' || pts.length > 2) { g.closePath(); g.fill(); } g.lineWidth = 1.5 / s; g.stroke(); }
      g.fillStyle = '#fff';
      for (const [x, y] of df.pts) { g.beginPath(); g.arc(x, y, 3 / s, 0, 7); g.fill(); }
    }
    if ((this.tool === 'brush' || this.tool === 'eraser') && m) {
      g.strokeStyle = this.tool === 'eraser' ? '#fff' : css(col, 1); g.lineWidth = 1.5 / s;
      g.beginPath(); g.arc(m.x, m.y, Math.max(this.opt.brushPx / 2, 0.5), 0, 7); g.stroke();
      g.strokeStyle = 'rgba(0,0,0,.6)'; g.lineWidth = 0.75 / s; g.stroke();
    }
    if (this.tool === 'wand' && this.wandPreview && m) {
      const p = this.wandPreview;
      g.globalAlpha = m.alt ? 0.5 : 1;
      g.drawImage(p.canvas, p.roi.x, p.roi.y);
      g.globalAlpha = 1;
      g.strokeStyle = 'rgba(255,255,255,.35)'; g.lineWidth = 1 / s; g.setLineDash([2 / s, 4 / s]);
      g.strokeRect(p.roi.x, p.roi.y, p.roi.w, p.roi.h); g.setLineDash([]);
    }
    if (this.tool === 'adopt' && this.adoptHover) {
      const h = this.adoptHover;
      g.strokeStyle = '#00e5ff'; g.lineWidth = 2.5 / s;
      if (h.type === 'vector') {
        g.beginPath(); h.f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); if (h.f.kind === 'polygon') g.closePath(); g.stroke();
      } else { g.beginPath(); g.arc(h.ix, h.iy, 6 / s, 0, 7); g.stroke(); }
    }
    if (dr?.kind === 'translate') {
      g.save(); g.translate(dr.dx, dr.dy);
      const cc = this.r.classCanvas.get(dr.cls);
      if (cc && !cc.empty && this.opt.translateScope === 'class') { g.globalAlpha = 0.6; g.drawImage(cc.canvas, 0, 0); g.globalAlpha = 1; }
      g.strokeStyle = css(col, 1); g.lineWidth = 1.5 / s;
      for (const f of d.features()) {
        if (f.cls !== dr.cls || !dr.subs.includes(f.sub)) continue;
        g.beginPath(); f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); if (f.kind === 'polygon') g.closePath(); g.stroke();
      }
      g.restore();
    }
  }

  drawScreen(g) {
    const im = this.imageDrag;
    if (im && this.mouse) {
      g.fillStyle = 'rgba(0,0,0,.7)'; g.fillRect(this.mouse.sx + 14, this.mouse.sy + 10, 230, 22);
      g.fillStyle = '#ffb020'; g.font = '12px ui-monospace, monospace';
      g.fillText(`${im.date} image: dx ${im.off[0]} px  dy ${im.off[1]} px`, this.mouse.sx + 20, this.mouse.sy + 25);
    }
    const dr = this.drag;
    if (dr?.kind === 'translate' && this.mouse) {
      const mpp = this.doc.mpp;
      g.fillStyle = 'rgba(0,0,0,.7)'; g.fillRect(this.mouse.sx + 14, this.mouse.sy + 10, 170, 22);
      g.fillStyle = '#fff'; g.font = '12px ui-monospace, monospace';
      g.fillText(`dx ${Math.round(dr.dx)} px  dy ${Math.round(dr.dy)} px  (${(Math.hypot(dr.dx, dr.dy) * mpp).toFixed(1)} m)`, this.mouse.sx + 20, this.mouse.sy + 25);
    }
  }
}

// Rotated rectangle from edge a->b and a third point giving the depth.
export function rect3(a, b, c) {
  const vx = b[0] - a[0], vy = b[1] - a[1], L = Math.hypot(vx, vy) || 1e-9;
  const nx = -vy / L, ny = vx / L;
  const depth = (c[0] - a[0]) * nx + (c[1] - a[1]) * ny;
  return [a, b, [b[0] + nx * depth, b[1] + ny * depth], [a[0] + nx * depth, a[1] + ny * depth]];
}
