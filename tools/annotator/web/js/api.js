// Thin client for server/app.py. Masks travel as gzip(packbits), MSB first
// (numpy's default bit order), row-major over the full location shape.

async function check(r) {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r;
}

export const getJSON = async (url) => (await check(await fetch(url))).json();

export async function putJSON(url, body) {
  const r = await fetch(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return (await check(r)).json();
}

export async function postJSON(url, body = {}) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return (await check(r)).json();
}

export async function del(url) { return (await check(await fetch(url, { method: 'DELETE' }))).json(); }

async function gunzip(buf) {
  const s = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

async function gzip(bytes) {
  const s = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

export function unpackBits(bits, n) {
  const out = new Uint8Array(n);
  for (let i = 0, j = 0; i < bits.length && j < n; i++) {
    const b = bits[i];
    if (b === 0) { j += 8; continue; }
    for (let k = 7; k >= 0 && j < n; k--, j++) out[j] = (b >> k) & 1;
  }
  return out;
}

export function packBits(data) {
  const out = new Uint8Array(Math.ceil(data.length / 8));
  for (let j = 0; j < data.length; j++) if (data[j]) out[j >> 3] |= 0x80 >> (j & 7);
  return out;
}

// Returns Uint8Array(w*h) of 0/1, or null when the mask does not exist.
export async function getMask(url, n) {
  const r = await check(await fetch(url));
  if (r.status === 204) return null;
  return unpackBits(await gunzip(await r.arrayBuffer()), n);
}

export async function putMask(url, data, empty) {
  const body = empty ? new Uint8Array(0) : await gzip(packBits(data));
  const r = await fetch(url + (empty ? '&empty=1' : ''), {
    method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body });
  return (await check(r)).json();
}
