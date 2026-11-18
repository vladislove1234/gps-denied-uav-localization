// Pure raster helpers on Uint8Array(w*h) 0/1 masks. No DOM: also imported by
// the wand worker and by the node self-test (checks/web_selftest.mjs).

export function bboxOf(mask, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) if (mask[row + x]) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

export function any(mask) { for (let i = 0; i < mask.length; i++) if (mask[i]) return true; return false; }

// Round stamp of radius r at (cx, cy); value 1 = paint, 0 = erase.
// `fn(i)` (optional) is called for every touched index instead of writing.
export function stamp(mask, w, h, cx, cy, r, value, fn) {
  const r2 = r * r;
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(h - 1, Math.ceil(cy + r));
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(w - 1, Math.ceil(cx + r));
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= r2) { const i = y * w + x; if (fn) fn(i); else mask[i] = value; }
    }
  }
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// Stroke segment a->b with stamps spaced at r/3.
export function strokeSegment(mask, w, h, ax, ay, bx, by, r, value, fn) {
  const d = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.ceil(d / Math.max(0.5, r / 3)));
  let box = null;
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    box = union(box, stamp(mask, w, h, ax + (bx - ax) * t, ay + (by - ay) * t, r, value, fn));
  }
  return box;
}

export function union(a, b) {
  if (!a) return b; if (!b) return a;
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  return { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
}

export function clipRect(r, w, h) {
  const x = Math.max(0, r.x), y = Math.max(0, r.y);
  const x1 = Math.min(w, r.x + r.w), y1 = Math.min(h, r.y + r.h);
  return x1 <= x || y1 <= y ? null : { x, y, w: x1 - x, h: y1 - y };
}

export function readRect(mask, w, r) {
  const out = new Uint8Array(r.w * r.h);
  for (let y = 0; y < r.h; y++) out.set(mask.subarray((r.y + y) * w + r.x, (r.y + y) * w + r.x + r.w), y * r.w);
  return out;
}

export function writeRect(mask, w, r, patch) {
  for (let y = 0; y < r.h; y++) mask.set(patch.subarray(y * r.w, (y + 1) * r.w), (r.y + y) * w + r.x);
}

// Integer translation; pixels shifted off the edge are lost.
export function shift(mask, w, h, dx, dy) {
  const out = new Uint8Array(w * h);
  dx = Math.round(dx); dy = Math.round(dy);
  for (let y = 0; y < h; y++) {
    const sy = y - dy; if (sy < 0 || sy >= h) continue;
    for (let x = 0; x < w; x++) {
      const sx = x - dx; if (sx < 0 || sx >= w) continue;
      out[y * w + x] = mask[sy * w + sx];
    }
  }
  return out;
}

// 4-connected component of `mask` containing (sx, sy): returns {mask, box}.
export function component(mask, w, h, sx, sy) {
  const start = sy * w + sx;
  if (!mask[start]) return null;
  const out = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let top = 0; stack[top++] = start; out[start] = 1;
  let x0 = sx, x1 = sx, y0 = sy, y1 = sy;
  while (top) {
    const i = stack[--top], x = i % w, y = (i - x) / w;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    if (x > 0 && mask[i - 1] && !out[i - 1]) { out[i - 1] = 1; stack[top++] = i - 1; }
    if (x < w - 1 && mask[i + 1] && !out[i + 1]) { out[i + 1] = 1; stack[top++] = i + 1; }
    if (y > 0 && mask[i - w] && !out[i - w]) { out[i - w] = 1; stack[top++] = i - w; }
    if (y < h - 1 && mask[i + w] && !out[i + w]) { out[i + w] = 1; stack[top++] = i + w; }
  }
  return { mask: out, box: { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 } };
}

// Outer boundary of the component containing the first set pixel, as pixel-
// corner coordinates (a closed ring, no repeated last point). Walks the crack
// edges between inside and outside pixels, so the ring encloses exactly the
// pixels -- rasterizing it back reproduces the blob (minus holes).
export function traceOuter(mask, w, h) {
  let s = -1;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { s = i; break; }
  if (s < 0) return [];
  const at = (x, y) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] === 1;
  // start at the top-left corner of the first pixel, heading right (inside on the right... below)
  const sx = s % w, sy = (s - sx) / w;
  let x = sx, y = sy, dir = 0;            // 0 right, 1 down, 2 left, 3 up (corner lattice)
  const DX = [1, 0, -1, 0], DY = [0, 1, 0, -1];
  const ring = [[x, y]];
  const limit = 4 * (w + 1) * (h + 1);
  for (let step = 0; step < limit; step++) {
    // pixels to the front-left and front-right of the current corner, relative to dir
    // for heading d from corner (x,y): left pixel / right pixel of the next edge
    const lp = leftPix(x, y, dir), rp = rightPix(x, y, dir);
    const L = at(lp[0], lp[1]), R = at(rp[0], rp[1]);
    let nd;
    if (L) nd = (dir + 3) % 4;             // turn left
    else if (R) nd = dir;                  // straight
    else nd = (dir + 1) % 4;               // turn right
    if (nd !== dir) { dir = nd; continue; }
    x += DX[dir]; y += DY[dir];
    if (x === sx && y === sy) break;
    const last = ring[ring.length - 1], prev = ring[ring.length - 2];
    if (prev && (prev[0] === last[0] && last[0] === x || prev[1] === last[1] && last[1] === y)) ring[ring.length - 1] = [x, y];
    else ring.push([x, y]);
  }
  return ring;
}
// Edge from corner (x,y) heading d separates two pixels; "right" is the inside.
function leftPix(x, y, d) { return [[x, y - 1], [x, y], [x - 1, y], [x - 1, y - 1]][d]; }
function rightPix(x, y, d) { return [[x, y], [x - 1, y], [x - 1, y - 1], [x, y - 1]][d]; }

export function simplify(pts, eps) {
  if (pts.length < 4 || eps <= 0) return pts.slice();
  // closed ring: split at the farthest pair so DP has two open chains
  let far = 0, best = -1;
  for (let i = 1; i < pts.length; i++) {
    const d = (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2;
    if (d > best) { best = d; far = i; }
  }
  const a = dp(pts.slice(0, far + 1), eps), b = dp(pts.slice(far).concat([pts[0]]), eps);
  return a.slice(0, -1).concat(b.slice(0, -1));
}

function dp(pts, eps) {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0], [bx, by] = pts[pts.length - 1];
  const L = Math.hypot(bx - ax, by - ay) || 1e-9;
  let idx = 0, dmax = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs((bx - ax) * (ay - pts[i][1]) - (ax - pts[i][0]) * (by - ay)) / L;
    if (d > dmax) { dmax = d; idx = i; }
  }
  if (dmax <= eps) return [pts[0], pts[pts.length - 1]];
  return dp(pts.slice(0, idx + 1), eps).slice(0, -1).concat(dp(pts.slice(idx), eps));
}

export function polygonArea(pts) {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  return Math.abs(a / 2);
}

export function pointInPolygon(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function distToSegment(px, py, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay, L2 = vx * vx + vy * vy;
  let t = L2 ? ((px - ax) * vx + (py - ay) * vy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

export function distToPolyline(px, py, pts) {
  let d = Infinity;
  for (let i = 0; i < pts.length - 1; i++) d = Math.min(d, distToSegment(px, py, ...pts[i], ...pts[i + 1]));
  return d;
}

// Even-odd scanline fill of a polygon (image px, corner convention: pixel i
// covers [i, i+1]) sampling at pixel centres. fn(i) per covered pixel.
export function fillPolygon(pts, w, h, fn) {
  if (pts.length < 3) return null;
  let y0 = Infinity, y1 = -Infinity, x0 = Infinity, x1 = -Infinity;
  for (const [x, y] of pts) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
  const ya = Math.max(0, Math.floor(y0)), yb = Math.min(h - 1, Math.ceil(y1));
  const xs = [];
  for (let y = ya; y <= yb; y++) {
    const cy = y + 0.5;
    xs.length = 0;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > cy) !== (yj > cy)) xs.push(xi + (cy - yi) / (yj - yi) * (xj - xi));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const a = Math.max(0, Math.ceil(xs[k] - 0.5)), b = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = a; x <= b; x++) fn(y * w + x);
    }
  }
  return { x: Math.max(0, Math.floor(x0)), y: ya, w: Math.ceil(x1) - Math.floor(x0) + 1, h: yb - ya + 1 };
}
