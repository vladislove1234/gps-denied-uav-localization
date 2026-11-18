// The open location: sublayers, vectors, masks, per-date overrides, undo, save.
//
// In memory a feature is {id, cls, sub, kind: 'polygon'|'polyline', width_m,
// pts: [[x, y], ...] in image px, props}. On disk it is GeoJSON in [lon, lat].
//
// Label for date D = base - remove(D) + add(D). `target` says where edits go:
// 'base' (every date) or a date string (that date only).

import * as api from './api.js';
import { gridFor } from './geo.js';
import * as R from './raster.js';

const clone = (o) => structuredClone(o);
let idSeq = 0;
export const newId = () => Date.now().toString(36) + (idSeq++).toString(36) + Math.random().toString(36).slice(2, 6);

export class Doc extends EventTarget {
  constructor(cfg, loc, work) {
    super();
    this.cfg = cfg;
    this.slug = loc.slug;
    this.meta = loc.meta;
    this.collection = loc.collection || 'sat';
    this.grid = gridFor(loc.meta);
    this.h = loc.meta.shape[0]; this.w = loc.meta.shape[1]; this.n = this.w * this.h;
    this.mpp = loc.meta.mpp;
    this.dates = loc.dates;
    this.refsMeta = loc.refs;
    this.state = work.state;
    this.classes = cfg.registry.classes.filter((c) => c.geometry !== 'none');
    this.classByName = Object.fromEntries(this.classes.map((c) => [c.name, c]));
    this.base = new Map();
    for (const f of work.base) this._put(this.base, this.fromGeo(f));
    this.over = {};
    for (const d of this.dates) {
      const o = work.overrides[d] || { add: [], remove: [] };
      const add = new Map();
      for (const f of o.add) this._put(add, this.fromGeo(f));
      this.over[d] = { add, removed: new Set(o.remove) };
    }
    this.masks = new Map();
    this.maskList = work.masks;
    this.dirty = new Set();
    this.target = 'base';
    this.viewDate = this.dates.includes('current') ? 'current' : this.dates[this.dates.length - 1];
    this.undoStack = []; this.redoStack = [];
    this.refVec = {}; this.refMask = {};
    this.saving = false; this.lastError = null;
  }

  async loadMasks() {
    await Promise.all(this.maskList.map(async (k) => {
      const data = await api.getMask(this.maskUrl(k), this.n);
      if (data) this.masks.set(k.join('|'), data);
    }));
  }

  // -- geometry conversion ---------------------------------------------------------

  fromGeo(f) {
    const p = f.properties, g = f.geometry;
    const ring = g.type === 'Polygon' ? g.coordinates[0] : g.coordinates;
    let pts = this.grid.ringToPx(ring);
    if (g.type === 'Polygon' && pts.length > 1) {
      const a = pts[0], b = pts[pts.length - 1];
      if (a[0] === b[0] && a[1] === b[1]) pts = pts.slice(0, -1);
    }
    const { class: cls, sublayer: sub, kind, width_m, ...rest } = p;
    const ft = { id: String(f.id ?? newId()), cls, sub, kind: kind || (g.type === 'Polygon' ? 'polygon' : 'polyline'),
      width_m: width_m ?? null, pts, props: rest };
    ft._geo = f.geometry; ft._pts0 = JSON.stringify(pts);    // untouched features keep exact lon/lat
    return ft;
  }

  toGeo(ft) {
    let geometry;
    if (ft._geo && ft._pts0 === JSON.stringify(ft.pts)) geometry = ft._geo;
    else {
      const ll = this.grid.ringToLonLat(ft.pts);
      geometry = ft.kind === 'polygon' ? { type: 'Polygon', coordinates: [[...ll, ll[0]]] } : { type: 'LineString', coordinates: ll };
    }
    const properties = { ...ft.props, class: ft.cls, sublayer: ft.sub, kind: ft.kind };
    if (ft.kind === 'polyline') properties.width_m = ft.width_m ?? 4;
    return { type: 'Feature', id: ft.id, properties, geometry };
  }

  _put(map, ft) { ft.bbox = bboxPts(ft.pts, ft.kind === 'polyline' ? (ft.width_m || 4) / this.mpp / 2 : 0); map.set(ft.id, ft); }
  touchFeature(ft) { ft.bbox = bboxPts(ft.pts, ft.kind === 'polyline' ? (ft.width_m || 4) / this.mpp / 2 : 0); }

  // -- sublayers --------------------------------------------------------------------

  sublayers(cls) { return this.state.sublayers[cls] || []; }

  // Effective features for a date, in draw order.
  features(date = this.viewDate) {
    const o = this.over[date];
    const out = [];
    for (const f of this.base.values()) if (!o || !o.removed.has(f.id)) out.push(f);
    if (o) for (const f of o.add.values()) { f._override = true; out.push(f); }
    return out;
  }

  isOverride(ft) { return !this.base.has(ft.id); }

  // -- masks ---------------------------------------------------------------------------

  maskUrl(k) {
    const [kind, ...rest] = k;
    if (kind === 'base') return `/api/loc/${this.slug}/mask?kind=base&cls=${rest[0]}&sub=${encodeURIComponent(rest[1])}`;
    return `/api/loc/${this.slug}/mask?kind=${kind}&date=${rest[0]}&cls=${rest[1]}&sub=${encodeURIComponent(rest[2])}`;
  }
  static key(kind, cls, sub, date) { return kind === 'base' ? `base|${cls}|${sub}` : `${kind}|${date}|${cls}|${sub}`; }
  mask(kind, cls, sub, date, create = false) {
    const k = Doc.key(kind, cls, sub, date);
    let m = this.masks.get(k);
    if (!m && create) { m = new Uint8Array(this.n); this.masks.set(k, m); }
    return m || null;
  }
  // The three arrays that decide a sublayer's pixels on `date`.
  maskTriple(cls, sub, date = this.viewDate) {
    return { B: this.mask('base', cls, sub), R: this.mask('remove', cls, sub, date), A: this.mask('add', cls, sub, date) };
  }
  hasRaster(cls, sub) {
    if (this.mask('base', cls, sub)) return true;
    return this.dates.some((d) => this.mask('add', cls, sub, d));
  }
  markMask(kind, cls, sub, date) { this.dirty.add('mask:' + Doc.key(kind, cls, sub, date)); }

  // Begin a raster edit on (cls, sub) at the current target. Returns an
  // object whose set(i, v) applies the override semantics, and end() which
  // records undo and returns the touched rect.
  beginRaster(cls, sub, label = 'paint') {
    const t = this.target;
    const arrs = t === 'base'
      ? { B: this.mask('base', cls, sub, null, true) }
      : { B: this.mask('base', cls, sub), A: this.mask('add', cls, sub, t, true), R: this.mask('remove', cls, sub, t, true) };
    const before = {};
    for (const [k, a] of Object.entries(arrs)) if (a) before[k] = a.slice();
    let box = null;
    const self = this;
    const setBase = (i, v) => { arrs.B[i] = v; };
    const setOver = (i, v) => {
      if (v) { arrs.A[i] = 1; arrs.R[i] = 0; } else { arrs.A[i] = 0; if (arrs.B && arrs.B[i]) arrs.R[i] = 1; }
    };
    return {
      set: t === 'base' ? setBase : setOver,
      grow(r) { box = R.union(box, r); },
      end() {
        const rect = box && R.clipRect(box, self.w, self.h);
        if (!rect) return null;
        const keys = t === 'base' ? { B: ['base', cls, sub, null] }
          : { A: ['add', cls, sub, t], R: ['remove', cls, sub, t] };
        const patches = [];
        for (const [k, key] of Object.entries(keys)) {
          patches.push({ key, before: R.readRect(before[k], self.w, rect), after: R.readRect(arrs[k], self.w, rect) });
          self.markMask(...key);
        }
        self.pushUndo(label, () => self._applyPatches(patches, rect, 'before'), () => self._applyPatches(patches, rect, 'after'));
        self.edited(cls, rect);
        return rect;
      },
    };
  }
  _applyPatches(patches, rect, which) {
    let cls;
    for (const p of patches) {
      const [kind, c, sub, date] = p.key; cls = c;
      R.writeRect(this.mask(kind, c, sub, date, true), this.w, rect, p[which]);
      this.markMask(kind, c, sub, date);
    }
    this.changed({ cls, rect });
  }

  // Effective 0/1 of a sublayer at pixel i on the view date.
  pixel(cls, sub, i, date = this.viewDate) {
    const { B, R: Rm, A } = this.maskTriple(cls, sub, date);
    return ((B && B[i] && !(Rm && Rm[i])) || (A && A[i])) ? 1 : 0;
  }

  // -- vector edits -----------------------------------------------------------------

  scopeMap(t = this.target) { return t === 'base' ? this.base : this.over[t].add; }

  // Snapshot-based undo for vector edits: cheap enough at ~2k features.
  snapVectors() {
    const s = { base: [...this.base.values()].map(clone), over: {} };
    for (const d of this.dates) s.over[d] = { add: [...this.over[d].add.values()].map(clone), removed: [...this.over[d].removed] };
    return s;
  }
  restoreVectors(s) {
    this.base = new Map(); for (const f of s.base) this.base.set(f.id, clone(f));
    for (const d of this.dates) {
      this.over[d] = { add: new Map(s.over[d].add.map((f) => [f.id, clone(f)])), removed: new Set(s.over[d].removed) };
      this.dirty.add('vec:' + d); this.dirty.add('rem:' + d);
    }
    this.dirty.add('vec:base');
  }
  // Wrap a vector mutation: fn() runs now, undo restores the snapshot.
  vectorEdit(label, fn, cls) {
    const before = this.snapVectors();
    const res = fn();
    const after = this.snapVectors();
    this.markVectors();
    this.pushUndo(label, () => { this.restoreVectors(before); this.changed({ vectors: true }); },
      () => { this.restoreVectors(after); this.changed({ vectors: true }); });
    this.edited(cls);
    this.changed({ vectors: true });
    return res;
  }
  // For drags: snapshot now, mutate freely, then commit() once at the end.
  beginVectors() {
    const before = this.snapVectors();
    return {
      commit: (label, cls) => {
        const after = this.snapVectors();
        this.markVectors();
        this.pushUndo(label, () => { this.restoreVectors(before); this.changed({ vectors: true }); },
          () => { this.restoreVectors(after); this.changed({ vectors: true }); });
        this.edited(cls);
        this.changed({ vectors: true });
      },
      cancel: () => { this.restoreVectors(before); this.changed({ vectors: true }); },
    };
  }

  markVectors() {
    if (this.target === 'base') this.dirty.add('vec:base');
    else { this.dirty.add('vec:' + this.target); this.dirty.add('rem:' + this.target); }
  }

  addFeature(ft) {
    ft.id = ft.id || newId();
    this._put(this.scopeMap(), ft);
    return ft;
  }

  // In a date override a base feature is never edited in place: it is hidden on
  // that date and an editable copy is added. Returns the feature to mutate.
  editable(ft) {
    if (this.target === 'base') {
      if (this.isOverride(ft)) return null;       // override features are edited in their date
      return ft;
    }
    const o = this.over[this.target];
    if (o.add.has(ft.id)) return ft;
    if (!this.base.has(ft.id)) return null;
    o.removed.add(ft.id);
    const copy = { ...clone(ft), id: newId(), props: { ...ft.props, fork_of: ft.id } };
    delete copy._geo; delete copy._pts0; delete copy._override;
    this._put(o.add, copy);
    return copy;
  }

  deleteFeature(ft) {
    if (this.target === 'base') { this.base.delete(ft.id); return; }
    const o = this.over[this.target];
    if (o.add.has(ft.id)) o.add.delete(ft.id); else o.removed.add(ft.id);
  }

  // -- undo ----------------------------------------------------------------------------

  pushUndo(label, undo, redo) {
    this.undoStack.push({ label, undo, redo });
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack = [];
  }
  undo() { const c = this.undoStack.pop(); if (c) { c.undo(); this.redoStack.push(c); this.scheduleSave(); } return c; }
  redo() { const c = this.redoStack.pop(); if (c) { c.redo(); this.undoStack.push(c); this.scheduleSave(); } return c; }

  // -- status bookkeeping ----------------------------------------------------------------

  // Any edit un-approves what it changes: a base edit every done date, an
  // override edit its own date. Nothing half-fixed gets exported.
  edited(cls, rect) {
    const dates = this.target === 'base' ? this.dates : [this.target];
    const reopened = [];
    for (const d of dates) {
      const s = this.state.dates[d];
      if (s.status === 'done') { s.status = 'in_progress'; reopened.push(d); }
      else if (s.status === 'todo') s.status = 'in_progress';
    }
    this.dirty.add('state');
    if (reopened.length) this.dispatchEvent(new CustomEvent('reopened', { detail: reopened }));
    this.scheduleSave();
    this.changed({ cls, rect, status: true });
  }

  // Align one date's image under the shared layers (px, integers). Undoable;
  // counts as an edit of that date, so a `done` date goes back to in_progress.
  setImageOffset(date, dx, dy) {
    const cur = (this.state.dates[date].offset_px || [0, 0]).slice();
    const next = [Math.round(dx), Math.round(dy)];
    if (cur[0] === next[0] && cur[1] === next[1]) return;
    const apply = (o) => {
      if (o[0] || o[1]) this.state.dates[date].offset_px = o.slice(); else delete this.state.dates[date].offset_px;
      this.dirty.add('state'); this.scheduleSave(); this.changed({ image: date, status: true });
    };
    apply(next);
    this.pushUndo(`align ${date} by ${next}`, () => apply(cur), () => apply(next));
    const s = this.state.dates[date];
    if (s.status === 'done') { s.status = 'in_progress'; this.dispatchEvent(new CustomEvent('reopened', { detail: [date] })); }
    else if (s.status === 'todo') s.status = 'in_progress';
  }

  setDate(date, patch) {
    Object.assign(this.state.dates[date], patch);
    this.dirty.add('state'); this.scheduleSave(); this.changed({ status: true });
  }

  changed(detail) { this.dispatchEvent(new CustomEvent('change', { detail })); }

  // -- references ---------------------------------------------------------------------

  async refVectors(source) {
    if (!this.refVec[source]) {
      const r = await fetch(`/api/refs/${source}/${this.slug}/vectors`);
      const fc = r.ok ? await r.json() : { features: [] };
      this.refVec[source] = fc.features.map((f) => { const ft = this.fromGeo(f); ft.bbox = bboxPts(ft.pts, ft.kind === 'polyline' ? (ft.width_m || 4) / this.mpp / 2 : 0); ft.source = source; return ft; });
    }
    return this.refVec[source];
  }
  async refMaskData(source, cls) {
    const k = source + '|' + cls;
    if (!(k in this.refMask)) this.refMask[k] = await api.getMask(`/api/refs/${source}/${this.slug}/mask/${cls}`, this.n);
    return this.refMask[k];
  }

  // -- sublayer operations (all undoable through a full snapshot) -------------------------

  snapAll(cls) {
    const masks = {};
    for (const [k, v] of this.masks) if (k.split('|').includes(cls)) masks[k] = v.slice();
    return { vec: this.snapVectors(), masks, subs: clone(this.state.sublayers) };
  }
  restoreAll(s, cls) {
    this.restoreVectors(s.vec);
    for (const k of [...this.masks.keys()]) if (k.split('|').includes(cls) && !(k in s.masks)) { this.masks.set(k, new Uint8Array(this.n)); this.dirty.add('mask:' + k); }
    for (const [k, v] of Object.entries(s.masks)) { this.masks.set(k, v.slice()); this.dirty.add('mask:' + k); }
    this.state.sublayers = clone(s.subs); this.dirty.add('state');
  }
  structural(label, cls, fn) {
    const before = this.snapAll(cls);
    fn();
    const after = this.snapAll(cls);
    for (const k of this.masks.keys()) if (k.split('|').includes(cls)) this.dirty.add('mask:' + k);
    this.dirty.add('vec:base'); this.dirty.add('state');
    this.pushUndo(label, () => { this.restoreAll(before, cls); this.changed({ cls, structure: true, vectors: true }); },
      () => { this.restoreAll(after, cls); this.changed({ cls, structure: true, vectors: true }); });
    this.edited(cls);
    this.changed({ cls, structure: true, vectors: true });
  }

  uniqueSubName(cls, name) {
    const names = new Set(this.sublayers(cls).map((s) => s.name));
    if (!names.has(name)) return name;
    for (let i = 2; ; i++) if (!names.has(`${name}-${i}`)) return `${name}-${i}`;
  }

  async addSourceSublayer(cls, source) {
    const vecs = (await this.refVectors(source)).filter((f) => f.cls === cls);
    const m = await this.refMaskData(source, cls);
    const name = this.uniqueSubName(cls, source);
    const version = this.refsMeta[source]?.version;
    this.structural(`add ${source} to ${cls}`, cls, () => {
      this.state.sublayers[cls].splice(Math.max(0, this.sublayers(cls).length - 1), 0, { name, source, source_version: version });
      for (const f of vecs) this._put(this.base, { ...clone(stripRef(f)), id: newId(), sub: name, props: { ...f.props, ref: `${source}:${f.id}` } });
      if (m) this.masks.set(Doc.key('base', cls, name), m.slice());
    });
    return name;
  }

  addManualSublayer(cls, name) {
    name = this.uniqueSubName(cls, name || 'manual');
    this.structural(`add sublayer ${name}`, cls, () => { this.state.sublayers[cls].push({ name, source: null }); });
    return name;
  }

  deleteSublayer(cls, sub) {
    this.structural(`delete ${cls}/${sub}`, cls, () => {
      this.state.sublayers[cls] = this.sublayers(cls).filter((s) => s.name !== sub);
      for (const [id, f] of this.base) if (f.cls === cls && f.sub === sub) this.base.delete(id);
      for (const d of this.dates) for (const [id, f] of this.over[d].add) if (f.cls === cls && f.sub === sub) this.over[d].add.delete(id);
      for (const k of this.masks.keys()) { const p = k.split('|'); if (p.at(-2) === cls && p.at(-1) === sub) this.masks.set(k, new Uint8Array(this.n)); }
    });
  }

  // Union `from` into `into`, preserving every date's effective pixels.
  mergeSublayer(cls, from, into) {
    this.structural(`merge ${from} into ${into}`, cls, () => {
      for (const f of this.base.values()) if (f.cls === cls && f.sub === from) f.sub = into;
      for (const d of this.dates) for (const f of this.over[d].add.values()) if (f.cls === cls && f.sub === from) f.sub = into;
      const Bi = this.mask('base', cls, into), Bf = this.mask('base', cls, from);
      for (const d of this.dates) {
        const Ri = this.mask('remove', cls, into, d), Rf = this.mask('remove', cls, from, d);
        const Ai = this.mask('add', cls, into, d), Af = this.mask('add', cls, from, d);
        if (Af) { const A = this.mask('add', cls, into, d, true); for (let i = 0; i < this.n; i++) if (Af[i]) A[i] = 1; }
        if (Ri || Rf) {
          const Rn = new Uint8Array(this.n);
          for (let i = 0; i < this.n; i++) {
            const bi = Bi ? Bi[i] : 0, bf = Bf ? Bf[i] : 0;
            if (!(bi || bf)) continue;
            const ei = bi && !(Ri && Ri[i]), ef = bf && !(Rf && Rf[i]);
            if (!ei && !ef) Rn[i] = 1;
          }
          this.masks.set(Doc.key('remove', cls, into, d), Rn);
        }
        if (Rf) this.masks.set(Doc.key('remove', cls, from, d), new Uint8Array(this.n));
        if (Af) this.masks.set(Doc.key('add', cls, from, d), new Uint8Array(this.n));
        void Ai;
      }
      if (Bf) { const B = this.mask('base', cls, into, null, true); for (let i = 0; i < this.n; i++) if (Bf[i]) B[i] = 1; this.masks.set(Doc.key('base', cls, from), new Uint8Array(this.n)); }
      this.state.sublayers[cls] = this.sublayers(cls).filter((s) => s.name !== from);
    });
  }

  async reseedSublayer(cls, sub) {
    const s = this.sublayers(cls).find((x) => x.name === sub);
    if (!s?.source) return;
    const vecs = (await this.refVectors(s.source)).filter((f) => f.cls === cls);
    const m = await this.refMaskData(s.source, cls);
    this.structural(`re-seed ${cls}/${sub}`, cls, () => {
      for (const [id, f] of this.base) if (f.cls === cls && f.sub === sub) this.base.delete(id);
      for (const f of vecs) this._put(this.base, { ...clone(stripRef(f)), id: newId(), sub, props: { ...f.props, ref: `${s.source}:${f.id}` } });
      this.masks.set(Doc.key('base', cls, sub), m ? m.slice() : new Uint8Array(this.n));
      s.source_version = this.refsMeta[s.source]?.version;
    });
  }

  // Move a sublayer (or every sublayer of a class) by (dx, dy) px, on every
  // date: base and override content together. Base target only.
  translate(cls, subs, dx, dy) {
    dx = Math.round(dx); dy = Math.round(dy);
    if (!dx && !dy) return;
    this.structural(`move ${cls} by ${dx},${dy}px`, cls, () => {
      const hit = (f) => f.cls === cls && subs.includes(f.sub);
      const move = (f) => { f.pts = f.pts.map(([x, y]) => [x + dx, y + dy]); this.touchFeature(f); };
      for (const f of this.base.values()) if (hit(f)) move(f);
      for (const d of this.dates) for (const f of this.over[d].add.values()) if (hit(f)) move(f);
      for (const [k, v] of [...this.masks]) {
        const p = k.split('|');
        if (p.at(-2) === cls && subs.includes(p.at(-1))) this.masks.set(k, R.shift(v, this.w, this.h, dx, dy));
      }
    });
  }

  deleteOverride(date) {
    const before = { vec: this.snapVectors(), masks: {} };
    for (const [k, v] of this.masks) if (k.split('|')[1] === date) before.masks[k] = v.slice();
    this.over[date] = { add: new Map(), removed: new Set() };
    for (const k of Object.keys(before.masks)) { this.masks.set(k, new Uint8Array(this.n)); this.dirty.add('mask:' + k); }
    this.dirty.add('vec:' + date); this.dirty.add('rem:' + date);
    this.pushUndo(`clear ${date} override`, () => {
      this.restoreVectors(before.vec);
      for (const [k, v] of Object.entries(before.masks)) { this.masks.set(k, v.slice()); this.dirty.add('mask:' + k); }
      this.changed({ vectors: true, structure: true });
    }, () => this.deleteOverride(date));
    this.edited();
    this.changed({ vectors: true, structure: true });
  }

  hasOverride(date) {
    const o = this.over[date];
    if (o.add.size || o.removed.size) return true;
    for (const [k, v] of this.masks) if (k.split('|')[1] === date && R.any(v)) return true;
    return false;
  }

  // -- saving -----------------------------------------------------------------------------

  scheduleSave(ms = 1500) {
    clearTimeout(this._t);
    this._t = setTimeout(() => this.save(), ms);
    this.dispatchEvent(new CustomEvent('saving', { detail: 'pending' }));
  }

  async save() {
    clearTimeout(this._t);
    if (this.saving) { this._again = true; return; }
    if (!this.dirty.size) { this.dispatchEvent(new CustomEvent('saving', { detail: 'saved' })); return; }
    this.saving = true;
    const items = [...this.dirty]; this.dirty.clear();
    this.dispatchEvent(new CustomEvent('saving', { detail: 'saving' }));
    try {
      // state last: it may reference sublayers whose files are written first
      for (const it of items.filter((x) => x !== 'state')) await this._saveItem(it);
      if (items.includes('state')) await this._saveItem('state');
      this.lastError = null;
      this.dispatchEvent(new CustomEvent('saving', { detail: 'saved' }));
    } catch (e) {
      for (const it of items) this.dirty.add(it);
      this.lastError = e;
      this.dispatchEvent(new CustomEvent('saving', { detail: 'error: ' + e.message }));
    } finally {
      this.saving = false;
      if (this._again) { this._again = false; this.save(); }
    }
  }

  async _saveItem(it) {
    const s = this.slug;
    if (it.startsWith('mask:')) {
      const k = it.slice(5), m = this.masks.get(k), parts = k.split('|');
      await api.putMask(this.maskUrl(parts), m || new Uint8Array(0), !m || !R.any(m));
    } else if (it === 'vec:base') {
      await api.putJSON(`/api/loc/${s}/vectors`, { features: [...this.base.values()].map((f) => this.toGeo(f)) });
    } else if (it.startsWith('vec:')) {
      const d = it.slice(4);
      await api.putJSON(`/api/loc/${s}/vectors?date=${d}`, { features: [...this.over[d].add.values()].map((f) => this.toGeo(f)) });
    } else if (it.startsWith('rem:')) {
      const d = it.slice(4);
      await api.putJSON(`/api/loc/${s}/removed/${d}`, { feature_ids: [...this.over[d].removed] });
    } else if (it === 'state') {
      // offset_px [0, 0] tells the server to drop a previous offset
      const dates = Object.fromEntries(Object.entries(this.state.dates).map(([d, v]) => [d, { ...v, offset_px: v.offset_px || [0, 0] }]));
      this.state = await api.putJSON(`/api/loc/${s}/state`, { sublayers: this.state.sublayers, dates,
        rejected: this.state.rejected, note: this.state.note });
      this.changed({ status: true });
    }
  }
}

function stripRef(f) { const { source, bbox, _geo, _pts0, ...rest } = f; return rest; }

export function bboxPts(pts, pad = 0) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
}
