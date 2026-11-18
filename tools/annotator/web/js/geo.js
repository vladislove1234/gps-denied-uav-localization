// Web Mercator grid of one location -- same formulas as tools/annotator/geo.py.
// Features live in image pixels in memory and in [lon, lat] on disk.

const TILE = 256;

export class Grid {
  constructor({ z, x0, y0, nx, ny }) {
    this.z = z; this.x0 = x0; this.y0 = y0; this.nx = nx; this.ny = ny;
    this.n = TILE * 2 ** z;
    this.ox = x0 * TILE; this.oy = y0 * TILE;
    this.w = nx * TILE; this.h = ny * TILE;
  }

  toPx(lat, lon) {
    const x = (lon + 180) / 360 * this.n;
    const y = (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * this.n;
    return [x - this.ox, y - this.oy];
  }

  toLatLon(x, y) {
    const gx = x + this.ox, gy = y + this.oy;
    const lon = gx / this.n * 360 - 180;
    const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * gy / this.n))) * 180 / Math.PI;
    return [lat, lon];
  }

  // GeoJSON [lon, lat] ring <-> [[x, y], ...]
  ringToPx(ring) { return ring.map(([lon, lat]) => this.toPx(lat, lon)); }
  ringToLonLat(pts) { return pts.map(([x, y]) => { const [lat, lon] = this.toLatLon(x, y); return [lon, lat]; }); }
}

// Pixel-only images (drone frames): stored coordinates ARE frame pixels,
// [x, y] where a map location would store [lon, lat]. Same interface as Grid.
export class PixelGrid {
  constructor(w, h) { this.w = w; this.h = h; this.pixel = true; }
  toPx(y, x) { return [x, y]; }
  toLatLon(x, y) { return [y, x]; }
  ringToPx(ring) { return ring.map(([x, y]) => [x, y]); }
  ringToLonLat(pts) { return pts.map(([x, y]) => [x, y]); }
}

export const gridFor = (meta) => (meta.crs === 'pixel' || !meta.grid
  ? new PixelGrid(meta.shape[1], meta.shape[0]) : new Grid(meta.grid));
