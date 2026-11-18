// Canvas2D renderer. One visible canvas; image space -> screen by
// screen = img * s + (tx, ty). Raster sublayers are composited per class into
// a full-size offscreen canvas, repainted only in dirty rects. Vectors are
// drawn per frame with viewport culling.

export function hexRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

// Shade k of n around the class colour: sources get distinguishable tints,
// `manual` keeps the pure class colour.
export function shade(hex, k, n, name) {
  const [r, g, b] = hexRgb(hex);
  if (name === 'manual' || n <= 1) return [r, g, b];
  const t = (k / Math.max(1, n - 1) - 0.5) * 0.9;      // -0.45 .. 0.45
  const mix = (c) => Math.round(t > 0 ? c + (255 - c) * t : c * (1 + t));
  return [mix(r), mix(g), mix(b)];
}
export const css = ([r, g, b], a = 1) => `rgba(${r},${g},${b},${a})`;

const DASH = { osm: [], 'ms-footprints': [6, 3], landcoverai: [2, 3], flair: [8, 3, 2, 3], worldcover: [1, 4] };

export class Renderer {
  constructor(canvas, app) {
    this.canvas = canvas; this.app = app;
    this.ctx = canvas.getContext('2d');
    this.view = { s: 0.25, tx: 0, ty: 0 };
    this.images = new Map();
    this.classCanvas = new Map();
    this.refCanvas = new Map();
    this.conflicts = null;
    this.pending = false;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  get doc() { return this.app.doc; }
  get ui() { return this.app.ui; }

  resize() {
    const p = this.canvas.parentElement, dpr = window.devicePixelRatio || 1;
    this.cw = p.clientWidth; this.ch = p.clientHeight;
    this.canvas.width = this.cw * dpr; this.canvas.height = this.ch * dpr;
    this.canvas.style.width = this.cw + 'px'; this.canvas.style.height = this.ch + 'px';
    this.dpr = dpr;
    this.request();
  }

  fit() {
    const d = this.doc; if (!d) return;
    const s = Math.min(this.cw / d.w, this.ch / d.h) * 0.98;
    this.view = { s, tx: (this.cw - d.w * s) / 2, ty: (this.ch - d.h * s) / 2 };
    this.request();
  }

  toImg(sx, sy) { const v = this.view; return [(sx - v.tx) / v.s, (sy - v.ty) / v.s]; }
  toScreen(x, y) { const v = this.view; return [x * v.s + v.tx, y * v.s + v.ty]; }

  zoomAt(sx, sy, factor) {
    const v = this.view, s = Math.min(40, Math.max(0.03, v.s * factor));
    const [ix, iy] = this.toImg(sx, sy);
    this.view = { s, tx: sx - ix * s, ty: sy - iy * s };
    this.request();
  }

  // -- images -------------------------------------------------------------------------

  async loadImages() {
    this.images.clear(); this.pixels = new Map();
    const d = this.doc;
    const first = d.viewDate;
    const load = (date) => new Promise((res) => {
      const im = new Image();
      im.onload = () => { this.images.set(date, im); this.request(); res(); };
      im.onerror = () => res();
      im.src = `/img/${d.slug}/${date}.jpg`;
    });
    await load(first);
    Promise.all(d.dates.filter((x) => x !== first).map(load));
  }

  // Per-date image alignment (px). Only moves how that image is drawn and
  // sampled; the JPEG itself is never changed.
  offsetOf(date) { const o = this.doc?.state.dates[date]?.offset_px; return o ? [o[0], o[1]] : [0, 0]; }

  // RGBA of a region of a date's image in LAYER coordinates (for the wand):
  // the image is drawn at its offset, so the wand sees what you see.
  pixelsOf(date, rect) {
    const [dx, dy] = this.offsetOf(date), key = `${date}|${dx}|${dy}`;
    let c = this.pixels.get(key);
    if (!c) {
      const im = this.images.get(date); if (!im) return null;
      c = new OffscreenCanvas(im.width, im.height);
      const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(im, dx, dy);
      for (const k of [...this.pixels.keys()]) if (k.startsWith(date + '|')) this.pixels.delete(k);
      this.pixels.set(key, c);
    }
    return c.getContext('2d', { willReadFrequently: true }).getImageData(rect.x, rect.y, rect.w, rect.h);
  }

  // -- raster composite per class ------------------------------------------------------

  invalidateClass(cls, rect) {
    if (!cls) { for (const c of this.doc.classes) this.repaintClass(c.name); }
    else this.repaintClass(cls, rect);
    this.request();
  }

  repaintClass(cls, rect) {
    const d = this.doc, ui = this.ui;
    let entry = this.classCanvas.get(cls);
    const subs = d.sublayers(cls).filter((s) => ui.subVisible(cls, s.name) && d.hasRaster(cls, s.name));
    if (!subs.length) { if (entry) entry.empty = true; if (!rect && entry) entry.ctx.clearRect(0, 0, d.w, d.h); return; }
    if (!entry) {
      const c = new OffscreenCanvas(d.w, d.h);
      entry = { canvas: c, ctx: c.getContext('2d') };
      this.classCanvas.set(cls, entry);
      rect = null;
    }
    entry.empty = false;
    const r = rect ? clip(rect, d.w, d.h) : { x: 0, y: 0, w: d.w, h: d.h };
    if (!r) return;
    const all = d.sublayers(cls);
    const layers = subs.map((s) => {
      const k = all.findIndex((x) => x.name === s.name);
      const col = shade(d.classByName[cls].color, k, all.length, s.name);
      return { ...d.maskTriple(cls, s.name), col, a: ui.subOpacity(cls, s.name) };
    });
    const img = entry.ctx.createImageData(r.w, r.h), px = img.data;
    for (let y = 0; y < r.h; y++) {
      let i = (r.y + y) * d.w + r.x, o = y * r.w * 4;
      for (let x = 0; x < r.w; x++, i++, o += 4) {
        let cr = 0, cg = 0, cb = 0, ca = 0;
        for (const L of layers) {
          if (!((L.B && L.B[i] && !(L.R && L.R[i])) || (L.A && L.A[i]))) continue;
          const a = L.a, na = a + ca * (1 - a);                // "over"
          cr = (L.col[0] * a + cr * ca * (1 - a)) / na; cg = (L.col[1] * a + cg * ca * (1 - a)) / na;
          cb = (L.col[2] * a + cb * ca * (1 - a)) / na; ca = na;
        }
        if (ca) { px[o] = cr; px[o + 1] = cg; px[o + 2] = cb; px[o + 3] = ca * 255; }
      }
    }
    entry.ctx.putImageData(img, r.x, r.y);
  }

  // Reference raster overlay, tinted, cached per (source, class).
  async refLayer(source, cls) {
    const k = source + '|' + cls;
    if (this.refCanvas.has(k)) return this.refCanvas.get(k);
    this.refCanvas.set(k, null);
    const d = this.doc, m = await d.refMaskData(source, cls);
    if (!m) return null;
    const c = new OffscreenCanvas(d.w, d.h), g = c.getContext('2d');
    const img = g.createImageData(d.w, d.h), px = img.data;
    const [r, gg, b] = hexRgb(d.classByName[cls].color);
    for (let i = 0, o = 0; i < m.length; i++, o += 4) if (m[i]) { px[o] = r; px[o + 1] = gg; px[o + 2] = b; px[o + 3] = 255; }
    g.putImageData(img, 0, 0);
    // hatch-free outline cue: nothing else; opacity comes from the UI
    this.refCanvas.set(k, c);
    this.request();
    return c;
  }

  // -- frame ---------------------------------------------------------------------------

  request() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  draw() {
    const g = this.ctx, d = this.doc, ui = this.ui, v = this.view;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#15171a'; g.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (!d) return;
    g.setTransform(this.dpr * v.s, 0, 0, this.dpr * v.s, this.dpr * v.tx, this.dpr * v.ty);
    g.imageSmoothingEnabled = v.s < 2;

    // imagery: A, then B by compare mode
    const cmp = ui.compare;
    const dateA = cmp.blinkOn && cmp.dateB ? cmp.dateB : d.viewDate;
    const A = this.images.get(dateA);
    const live = this.app.tools.imageDrag;                      // offset being dragged right now
    const off = (date) => (live && live.date === date ? live.off : this.offsetOf(date));
    if (A) { const [ax, ay] = off(dateA); g.globalAlpha = ui.imageOpacity; g.drawImage(A, ax, ay); }
    const B = cmp.dateB && this.images.get(cmp.dateB);
    if (B && !cmp.blinkOn) {
      const [bx, by] = off(cmp.dateB);
      if (cmp.mode === 'opacity') { g.globalAlpha = cmp.opacity * ui.imageOpacity; g.drawImage(B, bx, by); }
      else if (cmp.mode === 'swipe') {
        const [sx] = this.toImg(cmp.swipeX * this.cw, 0);
        g.save(); g.beginPath(); g.rect(sx, -1e5, 2e5, 2e5); g.clip();
        g.globalAlpha = ui.imageOpacity; g.drawImage(B, bx, by);
        g.restore();
      }
    }
    g.globalAlpha = 1;

    const vp = this.viewport();

    // reference overlays under the edit layers
    for (const [key, o] of Object.entries(ui.refOn)) {
      if (!o.on) continue;
      const [source, cls] = key.split('|');
      const rc = this.refCanvas.get(key);
      if (rc) { g.globalAlpha = o.opacity; g.drawImage(rc, 0, 0); g.globalAlpha = 1; }
      else if (!this.refCanvas.has(key) && d.refsMeta[source]?.classes?.[cls]?.raster) this.refLayer(source, cls);
      const vec = d.refVec[source];
      if (vec) this.drawRefVectors(g, vec.filter((f) => f.cls === cls), d.classByName[cls].color, DASH[source] || [4, 2], o.opacity, vp);
      else if (d.refsMeta[source]?.classes?.[cls]?.vector) d.refVectors(source).then(() => this.request());
    }

    // edit layers: lowest priority first, so the winner draws on top
    const order = [...d.classes].sort((a, b) => b.priority - a.priority);
    const feats = d.features();
    for (const c of order) {
      if (!ui.classVisible(c.name)) continue;
      const cc = this.classCanvas.get(c.name);
      if (cc && !cc.empty) { g.globalAlpha = ui.classOpacity(c.name); g.drawImage(cc.canvas, 0, 0); g.globalAlpha = 1; }
      this.drawVectors(g, feats.filter((f) => f.cls === c.name), c, vp);
    }

    if (this.conflicts && ui.showConflicts) { g.globalAlpha = 0.85; g.drawImage(this.conflicts, 0, 0); g.globalAlpha = 1; }

    // tool overlay in image space
    this.app.tools.drawOverlay(g, v.s);

    // swipe divider in screen space
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    if (cmp.dateB && cmp.mode === 'swipe' && !cmp.blinkOn) {
      const x = cmp.swipeX * this.cw;
      g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, this.ch); g.stroke();
      g.fillStyle = '#fff'; g.beginPath(); g.arc(x, this.ch / 2, 7, 0, 7); g.fill();
    }
    this.app.tools.drawScreen(g);
  }

  viewport() {
    const [x0, y0] = this.toImg(0, 0), [x1, y1] = this.toImg(this.cw, this.ch);
    return { x0, y0, x1, y1 };
  }

  drawVectors(g, feats, c, vp) {
    const d = this.doc, ui = this.ui, s = this.view.s;
    const all = d.sublayers(c.name);
    const style = {};
    all.forEach((sub, k) => { style[sub.name] = { col: shade(c.color, k, all.length, sub.name), a: ui.subOpacity(c.name, sub.name) }; });
    const sel = this.app.tools.selection;
    for (const f of feats) {
      if (!ui.subVisible(c.name, f.sub)) continue;
      const b = f.bbox;
      if (b.x1 < vp.x0 || b.x0 > vp.x1 || b.y1 < vp.y0 || b.y0 > vp.y1) continue;
      const st = style[f.sub] || { col: hexRgb(c.color), a: 0.6 };
      g.beginPath();
      f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      if (f.kind === 'polygon') {
        g.closePath();
        g.fillStyle = css(st.col, 0.35 * st.a); g.fill();
        g.strokeStyle = css(st.col, Math.min(1, 0.95 * st.a + 0.05)); g.lineWidth = 1.5 / s;
        g.setLineDash(f._override ? [5 / s, 3 / s] : []); g.stroke(); g.setLineDash([]);
      } else {
        g.lineCap = 'round'; g.lineJoin = 'round';
        g.strokeStyle = css(st.col, 0.5 * st.a); g.lineWidth = (f.width_m || 4) / d.mpp; g.stroke();
        g.strokeStyle = css(st.col, Math.min(1, st.a)); g.lineWidth = 1 / s;
        g.setLineDash(f._override ? [5 / s, 3 / s] : []); g.stroke(); g.setLineDash([]);
      }
      if (sel.has(f.id)) {
        g.strokeStyle = '#fff'; g.lineWidth = 2.5 / s; g.setLineDash([4 / s, 3 / s]);
        g.beginPath(); f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
        if (f.kind === 'polygon') g.closePath();
        g.stroke(); g.setLineDash([]);
      }
    }
  }

  drawRefVectors(g, feats, color, dash, opacity, vp) {
    const s = this.view.s, col = hexRgb(color), d = this.doc;
    g.setLineDash(dash.map((x) => x / s));
    for (const f of feats) {
      const b = f.bbox;
      if (b.x1 < vp.x0 || b.x0 > vp.x1 || b.y1 < vp.y0 || b.y0 > vp.y1) continue;
      g.beginPath();
      f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      if (f.kind === 'polygon') { g.closePath(); g.fillStyle = css(col, 0.12 * opacity); g.fill(); g.lineWidth = 1.5 / s; }
      else g.lineWidth = Math.max(1.5 / s, (f.width_m || 4) / d.mpp * 0.4);
      g.strokeStyle = css([255, 255, 255], 0.9 * opacity); g.stroke();
    }
    g.setLineDash([]);
  }

  // Pixels claimed by two or more classes on the view date.
  computeConflicts() {
    const d = this.doc, n = d.n;
    const count = new Uint8Array(n);
    const c = new OffscreenCanvas(d.w, d.h), g = c.getContext('2d', { willReadFrequently: true });
    const feats = d.features();
    for (const cl of d.classes) {
      const hit = new Uint8Array(n);
      for (const s of d.sublayers(cl.name)) {
        const { B, R, A } = d.maskTriple(cl.name, s.name);
        if (B || A) for (let i = 0; i < n; i++) if ((B && B[i] && !(R && R[i])) || (A && A[i])) hit[i] = 1;
      }
      const fs = feats.filter((f) => f.cls === cl.name);
      if (fs.length) {
        g.clearRect(0, 0, d.w, d.h); g.fillStyle = '#fff'; g.strokeStyle = '#fff'; g.lineCap = 'round'; g.lineJoin = 'round';
        for (const f of fs) {
          g.beginPath(); f.pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
          if (f.kind === 'polygon') { g.closePath(); g.fill(); } else { g.lineWidth = (f.width_m || 4) / d.mpp; g.stroke(); }
        }
        const a = g.getImageData(0, 0, d.w, d.h).data;
        for (let i = 0; i < n; i++) if (a[i * 4 + 3] > 127) hit[i] = 1;
      }
      for (let i = 0; i < n; i++) count[i] += hit[i];
    }
    const img = g.createImageData(d.w, d.h), px = img.data;
    let conflicts = 0;
    for (let i = 0; i < n; i++) if (count[i] > 1) { conflicts++; px[i * 4] = 255; px[i * 4 + 1] = 0; px[i * 4 + 2] = 255; px[i * 4 + 3] = 200; }
    g.clearRect(0, 0, d.w, d.h); g.putImageData(img, 0, 0);
    this.conflicts = c;
    this.request();
    return conflicts;
  }
}

function clip(r, w, h) {
  const x = Math.max(0, Math.floor(r.x)), y = Math.max(0, Math.floor(r.y));
  const x1 = Math.min(w, Math.ceil(r.x + r.w)), y1 = Math.min(h, Math.ceil(r.y + r.h));
  return x1 <= x || y1 <= y ? null : { x, y, w: x1 - x, h: y1 - y };
}
