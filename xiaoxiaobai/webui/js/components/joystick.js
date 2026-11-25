// Touch/mouse joystick. Reports x (right +) and y (up +) in -1..1; snaps back to 0 on release.

import { h } from "../ui.js";

export function joystick({ small = false, onChange } = {}) {
  const knob = h("div.knob");
  const el = h(`div.joystick${small ? ".sm" : ""}`, { role: "slider", "aria-label": "摇杆" }, knob);
  let enabled = true;
  let active = null;

  const set = (x, y) => {
    const r = el.clientWidth / 2 - knob.clientWidth / 2 - 4;
    knob.style.transform = `translate(${x * r}px, ${-y * r}px)`;
    onChange?.({ x, y });
  };

  const fromEvent = (event) => {
    const rect = el.getBoundingClientRect();
    const r = rect.width / 2 - knob.clientWidth / 2 - 4;
    let dx = (event.clientX - rect.left - rect.width / 2) / r;
    let dy = -(event.clientY - rect.top - rect.height / 2) / r;
    const mag = Math.hypot(dx, dy);
    if (mag > 1) {
      dx /= mag;
      dy /= mag;
    }
    const dead = (v) => (Math.abs(v) < 0.08 ? 0 : v);
    set(dead(dx), dead(dy));
  };

  el.addEventListener("pointerdown", (event) => {
    if (!enabled) return;
    active = event.pointerId;
    el.setPointerCapture(event.pointerId);
    el.classList.add("dragging");
    fromEvent(event);
  });
  el.addEventListener("pointermove", (event) => {
    if (event.pointerId === active) fromEvent(event);
  });
  const end = (event) => {
    if (event.pointerId !== active) return;
    active = null;
    el.classList.remove("dragging");
    set(0, 0);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener("lostpointercapture", end);

  el.setEnabled = (value) => {
    enabled = value;
    el.classList.toggle("disabled", !value);
    if (!value && active !== null) {
      active = null;
      el.classList.remove("dragging");
      set(0, 0);
    }
  };
  return el;
}
