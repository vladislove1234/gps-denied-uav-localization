// Runs the wand off the main thread so hover previews never stall panning.
import { wand } from './wand.js';

self.onmessage = (e) => {
  const { id, rgba, w, h, seed, tol, edge, roi } = e.data;
  const r = wand(rgba, w, h, seed, tol, edge);
  self.postMessage({ id, roi, ...r }, [r.mask.buffer]);
};
