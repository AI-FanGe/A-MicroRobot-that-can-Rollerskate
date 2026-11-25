// LED dot-matrix face rendering (same look as the robot's screen) and expression animation.
// A frame is one byte per LED: colour index << 5 | brightness 0-31 (src/face_engine.py), sent as
// base64 of its zero-run-length encoding.

import * as api from "../api.js";

const DEFAULT_FRAME_MS = 100;
const LEVEL_MAX = 31;
const GAMMA = 1.0;
// face_engine.PALETTE
const PALETTE = [
  [255, 236, 210], [255, 105, 155], [80, 170, 255], [255, 72, 48],
  [255, 196, 64], [120, 230, 110], [190, 120, 255], [150, 235, 255],
];
const loops = new Map();

export function decodeLevels(b64, cells) {
  const raw = atob(b64);
  const out = new Uint8Array(cells);
  let pos = 0;
  for (let i = 0; i < raw.length && pos < cells; i += 1) {
    const byte = raw.charCodeAt(i);
    if (byte !== 0) {
      out[pos] = byte;
      pos += 1;
    } else {
      i += 1;
      pos += raw.charCodeAt(i);
    }
  }
  return out;
}

function layer(cache, key, width, height, paint) {
  if (!cache.has(key)) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    paint(canvas.getContext("2d"));
    cache.set(key, canvas);
  }
  return cache.get(key);
}

function dots(ctx, cols, rows, pitch, ox, oy, radius) {
  ctx.beginPath();
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const cx = ox + (x + 0.5) * pitch;
      const cy = oy + (y + 0.5) * pitch;
      ctx.moveTo(cx + radius, cy);
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    }
  }
  ctx.fill();
}

// Unlit grid and the round-dot mask only depend on the canvas and matrix size.
const backgrounds = new Map();
const masks = new Map();
const scratch = new Map();

function geometry(canvas, cols, rows) {
  const { width, height } = canvas;
  const pitch = Math.min(width / cols, height / rows);
  return { width, height, pitch, ox: (width - cols * pitch) / 2, oy: (height - rows * pitch) / 2 };
}

export function drawLevels(canvas, cols, rows, levels, { glow = true } = {}) {
  const ctx = canvas.getContext("2d");
  const { width, height, pitch, ox, oy } = geometry(canvas, cols, rows);
  const key = `${width}x${height}:${cols}x${rows}`;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.filter = "none";
  ctx.drawImage(layer(backgrounds, key, width, height, (bg) => {
    bg.fillStyle = "#000";
    bg.fillRect(0, 0, width, height);
    bg.fillStyle = "#15161a";
    dots(bg, cols, rows, pitch, ox, oy, Math.max(0.5, pitch * 0.29));
  }), 0, 0);
  if (!levels) return;

  // One pixel per LED, then scaled up: blurred for the glow, masked by round dots for the LEDs.
  const small = layer(scratch, `small:${cols}x${rows}`, cols, rows, () => {});
  const sctx = small.getContext("2d");
  const image = sctx.createImageData(cols, rows);
  let any = false;
  for (let i = 0; i < cols * rows; i += 1) {
    const value = levels[i];
    const level = value & 31;
    if (level) any = true;
    const b = (level / LEVEL_MAX) ** GAMMA;
    const color = PALETTE[value >> 5];
    image.data[i * 4] = color[0] * b;
    image.data[i * 4 + 1] = color[1] * b;
    image.data[i * 4 + 2] = color[2] * b;
    image.data[i * 4 + 3] = 255;
  }
  if (!any) return;
  sctx.putImageData(image, 0, 0);

  const big = layer(scratch, `big:${key}`, width, height, () => {});
  const bctx = big.getContext("2d");
  bctx.globalCompositeOperation = "source-over";
  bctx.imageSmoothingEnabled = false;
  bctx.clearRect(0, 0, width, height);
  bctx.drawImage(small, ox, oy, cols * pitch, rows * pitch);
  bctx.globalCompositeOperation = "destination-in";
  bctx.drawImage(layer(masks, key, width, height, (mask) => {
    mask.fillStyle = "#fff";
    dots(mask, cols, rows, pitch, ox, oy, Math.max(0.7, pitch * 0.5));
  }), 0, 0);

  ctx.globalCompositeOperation = "lighter";
  if (glow) {
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.imageSmoothingEnabled = true;
    ctx.filter = `blur(${Math.max(1, pitch * 0.7)}px)`;
    ctx.drawImage(small, ox, oy, cols * pitch, rows * pitch);
    ctx.restore();
  }
  ctx.drawImage(big, 0, 0);
  ctx.globalCompositeOperation = "source-over";
}

// Older callers: a list of lit [x, y] LEDs.
export function drawMatrix(canvas, cols, rows, pixels, options = {}) {
  const levels = new Uint8Array(cols * rows);
  for (const [x, y] of pixels) levels[y * cols + x] = LEVEL_MAX;
  drawLevels(canvas, cols, rows, levels, options);
}

export async function loadLoop(name) {
  if (!loops.has(name)) {
    loops.set(name, api.get(`/interaction/emotion/${encodeURIComponent(name)}/frames`).then((data) => ({
      ...data,
      frames: (data.levels || []).map((b64) => decodeLevels(b64, data.cols * data.rows)),
    })).catch((error) => {
      loops.delete(name);
      throw error;
    }));
  }
  return loops.get(name);
}

// Plays an expression's animation loop on a canvas; onFrame lets a 3D texture refresh.
export class FaceAnimator {
  constructor(canvas, { glow = true, onFrame } = {}) {
    this.canvas = canvas;
    this.glow = glow;
    this.onFrame = onFrame;
    this.name = null;
    this.loop = null;
    this.frame = 0;
    this.frameMs = DEFAULT_FRAME_MS;
    this.timer = setInterval(() => this.tick(), this.frameMs);
  }

  async show(name) {
    if (!name || name === this.name) return;
    this.name = name;
    try {
      const loop = await loadLoop(name);
      if (this.name !== name) return;
      this.loop = loop;
      this.frame = 0;
      const frameMs = loop.frame_ms || DEFAULT_FRAME_MS;
      if (frameMs !== this.frameMs) {
        this.frameMs = frameMs;
        clearInterval(this.timer);
        this.timer = setInterval(() => this.tick(), frameMs);
      }
      this.draw();
    } catch {
      /* keep the previous face */
    }
  }

  tick() {
    if (!this.loop || this.loop.frames.length < 2 || document.hidden) return;
    this.frame = (this.frame + 1) % this.loop.frames.length;
    this.draw();
  }

  draw() {
    if (!this.loop) return;
    drawLevels(this.canvas, this.loop.cols, this.loop.rows, this.loop.frames[this.frame], { glow: this.glow });
    this.onFrame?.();
  }

  dispose() {
    clearInterval(this.timer);
  }
}
