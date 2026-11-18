// Self-test of the DOM-free frontend modules. Run from src/:
//   node tools/annotator/checks/web_selftest.mjs
// Pure bookkeeping/geometry -- the part that has no visible output to judge.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as R from '../web/js/raster.js';
import { Grid } from '../web/js/geo.js';
import { packBits, unpackBits } from '../web/js/api.js';
import { wand } from '../web/js/wand.js';

let n = 0;
const ok = (name) => { n++; console.log('ok', name); };

// 1) traceOuter -> fillPolygon reproduces a blob exactly (no holes)
function blobTest(seed) {
  const w = 64, h = 48, m = new Uint8Array(w * h);
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
  // random union of rectangles and a disc, one component guaranteed by a spine
  for (let x = 10; x < 50; x++) m[24 * w + x] = 1;
  for (let k = 0; k < 6; k++) {
    const x0 = 8 + (rnd() * 40) | 0, y0 = 6 + (rnd() * 30) | 0, ww = 2 + (rnd() * 12) | 0, hh = 2 + (rnd() * 12) | 0;
    for (let y = y0; y < Math.min(h, y0 + hh); y++) for (let x = x0; x < Math.min(w, x0 + ww); x++) m[y * w + x] = 1;
    for (let y = Math.min(24, y0); y <= Math.max(24, y0); y++) m[y * w + x0] = 1;   // connect to the spine
  }
  const comp = R.component(m, w, h, 10, 24).mask;
  // fill holes the way a traced outer ring would: flood outside from the border
  const outside = new Uint8Array(w * h), st = [];
  for (let x = 0; x < w; x++) { st.push(x, (h - 1) * w + x); }
  for (let y = 0; y < h; y++) { st.push(y * w, y * w + w - 1); }
  while (st.length) { const i = st.pop(); if (outside[i] || comp[i]) continue; outside[i] = 1; const x = i % w, y = (i - x) / w;
    if (x > 0) st.push(i - 1); if (x < w - 1) st.push(i + 1); if (y > 0) st.push(i - w); if (y < h - 1) st.push(i + w); }
  const want = outside.map((v) => (v ? 0 : 1));
  const ring = R.traceOuter(comp, w, h);
  const got = new Uint8Array(w * h);
  R.fillPolygon(ring, w, h, (i) => { got[i] = 1; });
  let diff = 0; for (let i = 0; i < w * h; i++) if (got[i] !== want[i]) diff++;
  return diff;
}
for (const seed of [1, 2, 3, 7, 42, 99]) assert.equal(blobTest(seed), 0, `blob ${seed}`);
ok('traceOuter -> fillPolygon round-trips 6 random blobs pixel-exactly');

// single pixel and diagonal-touching pixels
{
  const w = 5, h = 5, m = new Uint8Array(25); m[2 * 5 + 2] = 1;
  assert.deepEqual(R.traceOuter(m, w, h), [[2, 2], [3, 2], [3, 3], [2, 3]]);
  ok('single pixel ring');
}

// 2) packbits round trip, MSB first like numpy
{
  const a = new Uint8Array(1000).map((_, i) => (i * 7919 % 13 < 5 ? 1 : 0));
  assert.deepEqual(unpackBits(packBits(a), a.length), a);
  assert.equal(packBits(Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 1]))[0], 0b10000001);
  ok('packBits/unpackBits');
}

// 3) geo.js equals geo.py on a real location
{
  const meta = JSON.parse(readFileSync('data/derived/ua-village-multidate/birky-49p9105-23p9192/meta.json', 'utf8'));
  const g = new Grid(meta.grid);
  const [top, left] = g.toLatLon(0, 0), [bottom, right] = g.toLatLon(g.w, g.h);
  for (const [k, v] of Object.entries({ top, left, bottom, right })) assert.ok(Math.abs(v - meta.bounds[k]) < 1e-12, k);
  let worst = 0;
  for (let i = 0; i < 1000; i++) {
    const x = Math.random() * g.w, y = Math.random() * g.h;
    const [la, lo] = g.toLatLon(x, y), [x2, y2] = g.toPx(la, lo);
    worst = Math.max(worst, Math.abs(x - x2), Math.abs(y - y2));
  }
  assert.ok(worst < 1e-6, `roundtrip ${worst}`);
  ok(`geo.js matches meta.json bounds; roundtrip ${worst.toExponential(1)} px`);
}

// 4) wand fills one flat region and stops at the colour edge
{
  const w = 60, h = 40, rgba = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4, road = Math.abs(y - 20) < 4;           // grey road across green
    const noise = ((x * 31 + y * 17) % 7) - 3;
    rgba[o] = (road ? 150 : 60) + noise; rgba[o + 1] = (road ? 150 : 120) + noise; rgba[o + 2] = (road ? 150 : 50) + noise; rgba[o + 3] = 255;
  }
  const r = wand(rgba, w, h, [30, 20], 0.12, 0.5);
  let inRoad = 0, outRoad = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (r.mask[y * w + x]) (Math.abs(y - 20) < 4 ? inRoad++ : outRoad++);
  assert.equal(inRoad, w * 7, `road px ${inRoad}`);
  assert.equal(outRoad, 0, `leaked ${outRoad}`);
  ok('wand fills the road exactly, no leak');
}

// 5) simplify keeps a rectangle's 4 corners
{
  const ring = [];
  for (let x = 0; x < 10; x++) ring.push([x, 0]); for (let y = 0; y < 6; y++) ring.push([10, y]);
  for (let x = 10; x > 0; x--) ring.push([x, 6]); for (let y = 6; y > 0; y--) ring.push([0, y]);
  const s = R.simplify(ring, 0.5);
  assert.equal(s.length, 4, JSON.stringify(s));
  ok('simplify rectangle -> 4 points');
}

console.log(`\n${n} checks passed`);
