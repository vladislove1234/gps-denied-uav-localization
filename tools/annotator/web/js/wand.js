// Magic wand: flood fill from a seed pixel within a region of interest,
// accepting pixels whose colour is within `tol` of the seed's neighbourhood
// mean and stopping at image edges (Sobel magnitude above a percentile set by
// `edge`). Pure function -- run in wand-worker.js, and by the node self-test.
//
//   rgba   Uint8ClampedArray of the ROI (w*h*4)
//   seed   [sx, sy] inside the ROI
//   tol    0..1   colour distance (fraction of max RGB distance)
//   edge   0..1   0 = ignore edges, 1 = the strongest 30% of gradients are walls
//   returns {mask: Uint8Array(w*h), area, touchesBorder}

export function wand(rgba, w, h, seed, tol, edge) {
  const n = w * h;
  // 1) 3x3 box blur of RGB -- tames JPEG block noise before comparing colours
  const L = new Float32Array(n * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let r = 0, g = 0, b = 0, c = 0;
    for (let dy = -1; dy <= 1; dy++) {
      const yy = y + dy; if (yy < 0 || yy >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx; if (xx < 0 || xx >= w) continue;
        const o = (yy * w + xx) * 4; r += rgba[o]; g += rgba[o + 1]; b += rgba[o + 2]; c++;
      }
    }
    const i = (y * w + x) * 3; L[i] = r / c; L[i + 1] = g / c; L[i + 2] = b / c;
  }

  // 2) edge walls from Sobel magnitude on luminance
  const wall = new Uint8Array(n);
  if (edge > 0) {
    const Y = new Float32Array(n);
    for (let i = 0; i < n; i++) Y[i] = 0.299 * L[i * 3] + 0.587 * L[i * 3 + 1] + 0.114 * L[i * 3 + 2];
    const G = new Float32Array(n);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = -Y[i - w - 1] - 2 * Y[i - 1] - Y[i + w - 1] + Y[i - w + 1] + 2 * Y[i + 1] + Y[i + w + 1];
      const gy = -Y[i - w - 1] - 2 * Y[i - w] - Y[i - w + 1] + Y[i + w - 1] + 2 * Y[i + w] + Y[i + w + 1];
      G[i] = Math.hypot(gx, gy);
    }
    const thr = percentile(G, 1 - 0.3 * edge);
    for (let i = 0; i < n; i++) if (G[i] > thr && thr > 0) wall[i] = 1;
  }

  // 3) seed colour = mean of the 5x5 neighbourhood
  const [sx, sy] = seed;
  let mr = 0, mg = 0, mb = 0, mc = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const x = sx + dx, y = sy + dy; if (x < 0 || y < 0 || x >= w || y >= h) continue;
    const i = (y * w + x) * 3; mr += L[i]; mg += L[i + 1]; mb += L[i + 2]; mc++;
  }
  mr /= mc; mg /= mc; mb /= mc;
  const lim = (tol * 441.7) ** 2;          // 441.7 = |(255,255,255)|
  const ok = (i) => {
    const dr = L[i * 3] - mr, dg = L[i * 3 + 1] - mg, db = L[i * 3 + 2] - mb;
    return dr * dr + dg * dg + db * db <= lim;
  };
  // unblurred colour: the blur smears the last row of a region into its
  // neighbour, so the edge band is judged on raw pixels
  const okRaw = (i) => {
    const dr = rgba[i * 4] - mr, dg = rgba[i * 4 + 1] - mg, db = rgba[i * 4 + 2] - mb;
    return dr * dr + dg * dg + db * db <= lim;
  };

  // 4) 4-connected flood; walls stop it, but the seed may sit on one
  const mask = new Uint8Array(n), stack = new Int32Array(n);
  let top = 0, area = 0, touches = false;
  const s = sy * w + sx;
  mask[s] = 1; stack[top++] = s;
  while (top) {
    const i = stack[--top], x = i % w, y = (i - x) / w;
    area++;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) touches = true;
    const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
    for (const j of nb) if (j >= 0 && !mask[j] && !wall[j] && ok(j)) { mask[j] = 1; stack[top++] = j; }
  }

  // 5) the Sobel band is ~2 px wide, so the flood stops short of the visible
  //    edge: grow twice into wall pixels whose raw colour still fits, then
  //    close 1-px pinholes
  let grown = mask;
  for (let pass = 0; pass < 3; pass++) {
    const next = grown.slice();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (grown[i]) continue;
      let k = 0;
      if (x > 0 && grown[i - 1]) k++; if (x < w - 1 && grown[i + 1]) k++;
      if (y > 0 && grown[i - w]) k++; if (y < h - 1 && grown[i + w]) k++;
      if (pass < 2 ? (k >= 1 && wall[i] && okRaw(i)) : k >= 3) next[i] = 1;
    }
    grown = next;
  }
  let a2 = 0; for (let i = 0; i < n; i++) a2 += grown[i];
  return { mask: grown, area: a2, touchesBorder: touches };
}

function percentile(arr, q) {
  // histogram percentile: fast and good enough for a threshold
  let max = 0; for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  if (!max) return 0;
  const bins = 1024, hist = new Uint32Array(bins);
  for (let i = 0; i < arr.length; i++) hist[Math.min(bins - 1, (arr[i] / max * bins) | 0)]++;
  const target = q * arr.length;
  let acc = 0;
  for (let b = 0; b < bins; b++) { acc += hist[b]; if (acc >= target) return (b + 1) / bins * max; }
  return max;
}
