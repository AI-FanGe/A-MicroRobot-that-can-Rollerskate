// Virtual robot: the real robot's geometry (from the CAD model) animated from the
// commands the user sends. It previews intent; it does not mirror live joint readings.
// Every motor, board and sensor stays a separate mesh so the UI can label and pick them.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { FaceAnimator } from "./face.js";

const MODEL_URL = "model/xiaoxiaobai.json";
const WALK_SPEED = { vx: 0.22, vy: 0.12, vtheta: 1.0 }; // m/s, m/s, rad/s at full stick
const FOCUS_Y = 0.19; // camera looks at the robot's chest
let modelPromise = null;

function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const manifest = await (await fetch(MODEL_URL)).json();
      const buffer = await (await fetch(`model/${manifest.bin}`)).arrayBuffer();
      return { manifest, buffer };
    })().catch((error) => {
      modelPromise = null;
      throw error;
    });
  }
  return modelPromise;
}

function decodeMesh(mesh, buffer) {
  const count = mesh.vertices * 3;
  const q = new Int16Array(buffer, mesh.position.offset, count);
  const pos = new Float32Array(count);
  const { origin, scale } = mesh.position;
  for (let i = 0; i < count; i += 3) {
    pos[i] = q[i] * scale[0] + origin[0];
    pos[i + 1] = q[i + 1] * scale[1] + origin[1];
    pos[i + 2] = q[i + 2] * scale[2] + origin[2];
  }
  const Index = mesh.index.type === "uint16" ? Uint16Array : Uint32Array;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  // Crease-angle normals are baked by the build script (int8, normalized).
  geometry.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(buffer, mesh.normal.offset, count), 3, true));
  geometry.setIndex(new THREE.BufferAttribute(new Index(buffer, mesh.index.offset, mesh.triangles * 3), 1));
  geometry.computeBoundingSphere();
  return geometry;
}

const MATERIALS = {
  shell: { color: 0xe7eaf0, roughness: 0.5, metalness: 0.02 },
  motor: { color: 0x2b2f37, roughness: 0.38, metalness: 0.3 },
  metal: { color: 0xb8bfc9, roughness: 0.3, metalness: 0.75 },
  pi: { color: 0x1f6b47, roughness: 0.55, metalness: 0.1 },
  pcb: { color: 0x243661, roughness: 0.55, metalness: 0.1 },
  imu: { color: 0x8b5cf6, roughness: 0.4, metalness: 0.1 },
  battery: { color: 0x2f6fd6, roughness: 0.5, metalness: 0.1 },
  esp32: { color: 0x1c2027, roughness: 0.5, metalness: 0.25 },
  speaker: { color: 0x3a3f4a, roughness: 0.6, metalness: 0.1 },
  screen: { color: 0x07080a, roughness: 0.15, metalness: 0.0 },
};
// Opacity of everything but the focused part while one is hovered or selected.
const FADED_OPACITY = { shell: 0.1, default: 0.2 };
// Keeps the fade up while the pointer crosses the gap between two parts.
const FOCUS_RELEASE_S = 0.25;
// 撑起/躺下 (BIPED_PROPUP): the robot's reference (src/agents/propup_ref.npz) every PROPUP_KEY_S:
// right shoulder pitch, shoulder roll, elbow, hip pitch (the left side mirrors them), trunk
// pitch [rad], trunk height [m]. Lying until 0.5 s, up from 3.5 s, down from 5.4 s, lying from 8.5 s.
const PROPUP_KEY_S = 0.2;
const PROPUP_KEYS = [
  [0, 0.09, 0, 0.05, -1.57, 0.042], [0, 0.09, 0, 0.05, -1.57, 0.042], [0, 0.09, 0, 0.05, -1.57, 0.042],
  [0, 0.33, -0.4, 0.05, -1.57, 0.042], [-0.3, 1.1, -1.63, 0.05, -1.57, 0.042], [-1.21, 1.13, -1.69, 0.05, -1.57, 0.042],
  [-1.34, 1.11, -1.69, -0.02, -1.53, 0.039], [-1.45, 1, -1.69, -0.12, -1.45, 0.037], [-1.46, 0.69, -1.69, -0.22, -1.35, 0.040],
  [-1.4, 0.54, -1.57, -0.35, -1.21, 0.043], [-1.23, 0.43, -1.3, -0.55, -1.01, 0.047], [-1.03, 0.36, -0.98, -0.71, -0.83, 0.052],
  [-0.86, 0.33, -0.71, -0.83, -0.71, 0.054], [-1.08, 0.31, -1.04, -0.89, -0.64, 0.055], [-1.12, 0.13, -0.89, -0.93, -0.6, 0.055],
  [-0.96, 0.13, -0.6, -1, -0.54, 0.055], [-0.67, 0.13, -0.09, -1.06, -0.47, 0.056], [-0.65, 0.13, -0.05, -1.11, -0.42, 0.056],
  ...Array(10).fill([-0.65, 0.13, -0.05, -1.11, -0.42, 0.056]),
  [-0.65, 0.13, -0.05, -1.08, -0.45, 0.056], [-0.9, 0.13, -0.51, -1.02, -0.51, 0.056],
  [-1.05, 0.13, -0.76, -0.95, -0.59, 0.055], [-1.2, 0.24, -1.18, -0.91, -0.63, 0.055], [-0.83, 0.32, -0.66, -0.86, -0.68, 0.054],
  [-0.98, 0.35, -0.9, -0.75, -0.79, 0.052], [-1.17, 0.4, -1.2, -0.6, -0.95, 0.049], [-1.36, 0.5, -1.51, -0.41, -1.16, 0.044],
  [-1.46, 0.62, -1.69, -0.25, -1.32, 0.041], [-1.47, 0.92, -1.69, -0.15, -1.42, 0.038], [-1.36, 1.1, -1.69, -0.05, -1.51, 0.038],
  [-1.29, 1.13, -1.69, 0.04, -1.57, 0.041], [-0.6, 1.13, -1.69, 0.05, -1.57, 0.042], [-0.01, 0.61, -0.85, 0.05, -1.57, 0.042],
  [0, 0.09, -0.01, 0.05, -1.57, 0.042], [0, 0.09, 0, 0.05, -1.57, 0.042], [0, 0.09, 0, 0.05, -1.57, 0.042],
];
// 偷看 (BIPED_PEEK): trunk roll [rad, - leans left] and head yaw [rad] of the robot's reference
// (src/agents/peek_ref.npz) every PEEK_KEY_S from the start of the mode; the limbs are a stand-in.
const PEEK_KEY_S = 0.25;
const PEEK_KEYS = [
  [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.01, -0.01], [0.03, -0.04], [0.03, -0.09], [0.03, -0.14],
  [0.03, -0.18], [0.02, -0.21], [0, -0.01], [0, 0], [-0.02, -0.01], [-0.27, -0.11], [-0.42, -0.28], [-0.44, -0.58],
  [-0.45, -0.68], [-0.45, -0.44], [-0.45, -0.17], [-0.24, -0.07], [0, 0.02], [0, 0.29], [0, 0.34], [0, -0.38],
  [0, -0.14], [0, 0], [-0.05, -0.01], [-0.13, -0.02], [-0.23, -0.04], [-0.33, -0.07], [-0.41, -0.11], [-0.45, -0.16],
  [-0.45, -0.33], [-0.45, -0.68], [-0.45, -0.73], [-0.45, -0.73], [-0.45, -0.44], [-0.45, -0.6], [-0.23, -0.37],
  [0, 0], [0, 0], [0, 0], [0, 0], [-0.28, -0.23], [-0.38, -0.31], [-0.29, -0.24], [-0.06, -0.07], [-0.45, -0.49],
  [-0.45, -0.44], [-0.43, -0.32], [-0.01, 0], [0, 0], [0.01, 0], [0.03, 0], [0.03, 0], [0.03, 0], [0.03, 0],
  [0.03, 0], [0.03, 0], [0.01, 0], [0, 0],
];

// 躺着/坐地/坐凳休闲, 起床伸懒腰: trunk pitch [rad] and height [m] of the robot's clips (src/agents/chill_*.npz,
// wake_stretch.npz: the base pose of mjlab_xiaoxiaobai scripts/chill_motion.py); the limbs are a stand-in around it.
const CHILL_BODY = {
  BIPED_CHILL_LIE: [-1.57, 0.042],
  BIPED_WAKE_STRETCH: [-1.57, 0.042],
  BIPED_CHILL_SIT: [-0.42, 0.056],
  BIPED_CHILL_CHAIR: [-0.49, 0.164],
};
// The stool of 坐凳休闲 (chill_chair.npz's sim scene): [x, z, half x, half y, half z] of its seat and leg block.
const STOOL_BOXES = [[-0.044, 0.098, 0.076, 0.14, 0.01], [-0.095, 0.044, 0.025, 0.12, 0.044]];

const quat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class Twin {
  constructor(container) {
    this.container = container;
    this.state = { mode: "PASSIVE_DEFAULT", velocity: { vx: 0, vy: 0, vtheta: 0 }, head: false, upper: {}, expression: "smile" };
    this.modeSince = performance.now() / 1000;
    this.joints = new Map();
    this.angles = new Map();
    this.base = { x: 0, y: 0, yaw: 0 };
    this.phase = 0;
    this.disposed = false;
    this.components = new Map();
    this.highlighted = new Set();
    this.focus = 0;
    this.focusReleaseAt = 0;
    this.preview = null;
    this.onFrame = null;
    this.inset = { right: 0, bottom: 0 };
    this.insetTarget = { right: 0, bottom: 0 };
  }

  async init() {
    const { manifest, buffer } = await loadModel();
    if (this.disposed) return;
    this.manifest = manifest;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    this.renderer = renderer;
    this.container.append(renderer.domElement);

    const scene = new THREE.Scene();
    this.scene = scene;
    const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 30);
    this.camera = camera;
    this.controls = new OrbitControls(camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 3;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.ROTATE, RIGHT: THREE.MOUSE.PAN };
    // Stop the browser's middle-click autoscroll from taking over the drag.
    renderer.domElement.addEventListener("mousedown", (e) => e.button === 1 && e.preventDefault());
    this.resetView();

    scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4255, 1.25));
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(0.8, 1.6, 0.6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -0.5, right: 0.5, top: 0.5, bottom: -0.5, near: 0.1, far: 4 });
    sun.shadow.bias = -0.0005;
    this.sun = sun;
    scene.add(sun, sun.target);
    const fill = new THREE.DirectionalLight(0x9db8ff, 0.6);
    fill.position.set(-1, 0.6, -0.8);
    scene.add(fill);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: 0.28 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    this.floor = floor;
    this.buildGrid();

    // MuJoCo is z-up; three.js is y-up.
    const root = new THREE.Group();
    root.rotation.x = -Math.PI / 2;
    scene.add(root);
    this.root = root;

    const stool = new THREE.Group();
    const wood = new THREE.MeshStandardMaterial({ color: 0x8c6a4a, roughness: 0.8 });
    for (const [x, z, hx, hy, hz] of STOOL_BOXES) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz), wood);
      box.position.set(x, 0, z);
      box.castShadow = box.receiveShadow = true;
      stool.add(box);
    }
    stool.visible = false;
    root.add(stool);
    this.stool = stool;

    this.materials = {};
    for (const [kind, def] of Object.entries(MATERIALS)) {
      this.materials[kind] = new THREE.MeshStandardMaterial({ ...def, side: THREE.DoubleSide });
    }
    this.baseColors = Object.fromEntries(Object.entries(this.materials).map(([k, m]) => [k, m.color.clone()]));
    this.accent = new THREE.Color(cssVar("--accent") || "#5b8cff");

    const bodies = manifest.bodies.map((b) => {
      const outer = new THREE.Group();
      outer.name = b.name;
      outer.position.fromArray(b.pos);
      outer.quaternion.copy(quat(b.quat));
      return { outer, inner: outer, def: b };
    });
    for (const j of manifest.joints) {
      const body = bodies[j.body];
      const group = new THREE.Group();
      group.name = j.name;
      group.position.fromArray(j.pos);
      body.inner.add(group);
      body.inner = group;
      this.joints.set(j.name, { group, axis: new THREE.Vector3().fromArray(j.axis).normalize(), def: j });
      this.angles.set(j.name, j.neutral);
    }
    for (const body of bodies) (body.def.parent < 0 ? root : bodies[body.def.parent].inner).add(body.outer);
    this.bodies = bodies;
    this.trunk = bodies[0].outer;

    const geometries = manifest.meshes.map((m) => decodeMesh(m, buffer));
    this.parts = manifest.parts.map((p, index) => {
      const mesh = new THREE.Mesh(geometries[p.mesh], this.materials[p.kind] || this.materials.shell);
      if (p.matrix) {
        mesh.matrixAutoUpdate = false;
        mesh.matrix.fromArray(p.matrix);
      }
      mesh.castShadow = true;
      mesh.userData.part = index;
      bodies[p.body].inner.add(mesh);
      return { def: p, mesh, body: bodies[p.body].inner, center: new THREE.Vector3().fromArray(p.center) };
    });
    for (const c of manifest.components) this.components.set(c.id, { ...c, part: this.parts[c.part] });
    this.componentOfPart = new Map([...this.components.values()].map((c) => [c.part, c.id]));
    this.raycaster = new THREE.Raycaster();
    this.feet = bodies.filter((b) => /^foot/.test(b.def.name)).map((b) => b.outer);
    this.buildFace(bodies[manifest.face.body]);

    this.applyPose(this.neutralPose());
    this.trunk.position.set(0, 0, manifest.ground_z);
    root.updateMatrixWorld(true);
    this.soleZ = this.lowestFoot();
    this.applyPose(this.poseFor(this.state.mode, 0, 0));
    this.settleOnGround();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.clock = new THREE.Clock();
    this.onTheme = () => this.buildGrid();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  buildGrid() {
    if (!this.scene) return;
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
    }
    const color = new THREE.Color(cssVar("--scene-grid") || "#232a38");
    this.grid = new THREE.GridHelper(6, 60, color, color);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.75;
    this.grid.position.y = 0.0005;
    this.scene.add(this.grid);
  }

  buildFace(head) {
    if (!head) return;
    const face = this.manifest.face;
    this.faceCanvas = document.createElement("canvas");
    this.faceCanvas.width = 256;
    this.faceCanvas.height = 192;
    this.faceTexture = new THREE.CanvasTexture(this.faceCanvas);
    this.faceTexture.colorSpace = THREE.SRGBColorSpace;
    this.faceTexture.anisotropy = 4;
    // The face canvas is 4:3; fit it inside the screen part's front.
    const height = Math.min(face.size[1] * 0.92, (face.size[0] * 0.92 * 3) / 4);
    this.faceMaterial = new THREE.MeshBasicMaterial({ map: this.faceTexture, toneMapped: false });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry((height * 4) / 3, height), this.faceMaterial);
    // Plane +x/+y/+z → robot left (+y), up (+z), forward (+x), so it reads correctly from the front.
    plane.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0)),
    );
    plane.position.set(face.front_x + 0.0006, face.center[0], face.center[1]);
    head.inner.add(plane);
    this.faceAnimator = new FaceAnimator(this.faceCanvas, {
      glow: true,
      onFrame: () => {
        this.faceTexture.needsUpdate = true;
      },
    });
    this.faceAnimator.show(this.state.expression);
  }

  // ------------------------------------------------------------ public API

  update(patch) {
    if (patch.mode && patch.mode !== this.state.mode) this.modeSince = performance.now() / 1000;
    Object.assign(this.state, patch);
    if (patch.expression) this.faceAnimator?.show(patch.expression);
  }

  // ------------------------------------------------------------ components

  componentList() {
    return [...this.components.values()];
  }

  /** Screen position (CSS px inside the container) of a component's center. */
  project(id, out = {}) {
    const c = this.components.get(id);
    if (!c || !this.camera) return null;
    const v = c.part.body.localToWorld(c.part.center.clone());
    v.project(this.camera);
    const { clientWidth: w, clientHeight: h } = this.container;
    out.x = (v.x * 0.5 + 0.5) * w;
    out.y = (-v.y * 0.5 + 0.5) * h;
    out.visible = v.z < 1;
    return out;
  }

  /** Component under a pointer position (CSS px inside the container); the ray passes through the shell. */
  pick(x, y) {
    if (!this.parts) return null;
    const { clientWidth: w, clientHeight: h } = this.container;
    this.raycaster.setFromCamera(new THREE.Vector2((x / w) * 2 - 1, -(y / h) * 2 + 1), this.camera);
    this.pickable ??= this.parts.filter((p) => this.componentOfPart.has(p)).map((p) => p.mesh);
    const hit = this.raycaster.intersectObjects(this.pickable, false)[0];
    if (!hit) return null;
    return this.componentOfPart.get(this.parts[hit.object.userData.part]) || null;
  }

  /** Tint the given components with the accent colour and fade the rest of the robot around them. */
  highlight(ids) {
    const next = new Set(ids.filter(Boolean));
    if (!next.size && this.highlighted.size) this.focusReleaseAt = performance.now() / 1000 + FOCUS_RELEASE_S;
    for (const id of this.highlighted) {
      if (next.has(id)) continue;
      const part = this.components.get(id)?.part;
      if (part) {
        part.mesh.material.dispose();
        part.mesh.material = this.materials[part.def.kind] || this.materials.shell;
      }
    }
    for (const id of next) {
      if (this.highlighted.has(id)) continue;
      const part = this.components.get(id)?.part;
      if (!part) continue;
      const material = (this.materials[part.def.kind] || this.materials.shell).clone();
      material.emissive = this.accent.clone();
      material.emissiveIntensity = 0.55;
      material.transparent = false;
      material.opacity = 1;
      material.depthWrite = true;
      part.mesh.material = material;
    }
    this.highlighted = next;
    this.applyFocus(this.focus);
  }

  /** 0 = whole robot solid, 1 = everything except the highlighted parts faded. */
  applyFocus(f) {
    this.focus = f;
    const see = f > 0.001;
    const fade = (m, faded) => {
      m.opacity = 1 - f * (1 - faded);
      if (m.transparent !== see) {
        m.transparent = see;
        m.depthWrite = !see;
        m.needsUpdate = true;
      }
    };
    for (const [kind, m] of Object.entries(this.materials)) fade(m, FADED_OPACITY[kind] ?? FADED_OPACITY.default);
    if (this.faceMaterial) {
      const screenOn = [...this.highlighted].some((id) => this.components.get(id)?.kind === "screen");
      fade(this.faceMaterial, screenOn ? 1 : FADED_OPACITY.default);
    }
  }

  /** Swing one joint back and forth for a few seconds to show its axis. */
  previewJoint(joint) {
    this.preview = this.joints.has(joint) ? { joint, start: performance.now() / 1000 } : null;
  }

  resetView() {
    const target = this.trunk ? this.trunkWorld() : new THREE.Vector3(0, FOCUS_Y, 0);
    this.camera.position.set(target.x + 0.78, 0.42, target.z + 0.58);
    this.controls.target.copy(target);
    this.controls.update();
  }

  resetPosition() {
    this.base = { x: 0, y: 0, yaw: 0 };
    this.resetView();
  }

  dispose() {
    this.disposed = true;
    this.renderer?.setAnimationLoop(null);
    this.resizeObserver?.disconnect();
    this.faceAnimator?.dispose();
    this.controls?.dispose();
    this.scene?.traverse((obj) => {
      obj.geometry?.dispose();
      if (obj.material) [].concat(obj.material).forEach((m) => {
        m.map?.dispose();
        m.dispose();
      });
    });
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
  }

  // ------------------------------------------------------------ poses

  neutralPose() {
    const pose = {};
    for (const [name, j] of this.joints) pose[name] = j.def.neutral;
    pose.right_elbow = -0.25;
    pose.left_elbow = 0.25;
    return pose;
  }

  // flex > 0 bends the leg (hip forward, knee back, ankle keeps the sole flat);
  // swing > 0 moves the foot backward. Left-side pitch joints turn the other way.
  leg(pose, side, flex, swing = 0, roll = 0, yaw = 0) {
    const s = side === "right" ? 1 : -1;
    pose[`${side}_hip_pitch`] = s * (-flex + swing);
    pose[`${side}_knee`] = s * 2 * flex;
    pose[`${side}_ankle_pitch`] = s * (-flex - swing);
    pose[`${side}_hip_roll`] = (pose[`${side}_hip_roll`] || 0) + roll;
    pose[`${side}_ankle_roll`] = (pose[`${side}_ankle_roll`] || 0) - roll;
    pose[`${side}_hip_yaw`] = (pose[`${side}_hip_yaw`] || 0) + yaw;
  }

  crouchPose(amount) {
    const pose = this.neutralPose();
    this.leg(pose, "right", 0.78 * amount);
    this.leg(pose, "left", 0.78 * amount);
    pose.right_shoulder_pitch = 0.35 * amount;
    pose.left_shoulder_pitch = -0.35 * amount;
    pose.right_elbow = -0.25 - 0.5 * amount;
    pose.left_elbow = 0.25 + 0.5 * amount;
    return pose;
  }

  poseFor(mode, t, speed) {
    const since = t - this.modeSince;
    const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
    switch (mode) {
      case "PASSIVE_DEFAULT":
      case "STOPPING":
        return this.crouchPose(1);
      case "STARTING":
        return this.crouchPose(1 - ease(since / 2.5));
      case "BIPED_WALK_RUN":
      case "BIPED_SKATE":
        return this.walkPose(speed);
      case "BIPED_WALK_UPPER": {
        // Legs as in the walk; arms and head from the robot's joints (13-20, polled).
        const pose = this.walkPose(speed);
        for (const [name, value] of Object.entries(this.state.liveUpper || {})) pose[name] = value;
        return pose;
      }
      case "BIPED_GYMNASTICS": {
        // A stand-in groove, not the robot's choreography.
        const pose = this.neutralPose();
        const beat = t * 2 * Math.PI * 1.1;
        this.leg(pose, "right", 0.12 + 0.18 * Math.max(0, Math.sin(beat)));
        this.leg(pose, "left", 0.12 + 0.18 * Math.max(0, -Math.sin(beat)));
        const wave = Math.sin(beat);
        pose.right_shoulder_pitch = -0.7 - 0.5 * wave;
        pose.left_shoulder_pitch = 0.7 - 0.5 * wave;
        pose.right_shoulder_roll = 0.5 + 0.3 * wave;
        pose.left_shoulder_roll = -0.5 + 0.3 * wave;
        pose.right_elbow = -0.6;
        pose.left_elbow = 0.6;
        pose.head_wobble_roll = 0.08 * wave;
        return pose;
      }
      case "BIPED_PEEK": {
        // Crouched with the hands at the chest from 1.2 s to 14 s; the trunk lean is set in frame().
        const [lean, head] = this.peekKey(since);
        const hide = clamp(Math.min((since - 1.2) / 0.8, (14 - since) / 0.6), 0, 1);
        const out = clamp(-lean / 0.45, 0, 1);
        const pose = this.neutralPose();
        this.leg(pose, "right", hide * (0.35 + 0.1 * out));
        this.leg(pose, "left", hide * (0.35 + 0.15 * out));
        Object.assign(pose, {
          right_shoulder_pitch: 0.8 * hide, right_shoulder_roll: 0.09 + 0.22 * hide, right_elbow: -1.19 * hide,
          left_shoulder_pitch: (-0.8 + 0.52 * out) * hide, left_shoulder_roll: -0.09 - (0.22 + 0.6 * out) * hide,
          left_elbow: (1.19 - 0.73 * out) * hide,
          head,
        });
        return pose;
      }
      case "BIPED_PROPUP": {
        const [sp, sr, el, hp] = this.propupKey();
        const pose = {};
        for (const name of this.joints.keys()) pose[name] = 0;
        Object.assign(pose, {
          right_shoulder_pitch: sp, right_shoulder_roll: sr, right_elbow: el, right_hip_pitch: hp,
          left_shoulder_pitch: -sp, left_shoulder_roll: -sr, left_elbow: -el, left_hip_pitch: -hp,
          right_hip_roll: 0.06, left_hip_roll: -0.06, right_ankle_roll: -0.06, left_ankle_roll: 0.06,
        });
        return pose;
      }
      case "BIPED_CHILL_LIE":
      case "BIPED_CHILL_SIT":
      case "BIPED_CHILL_CHAIR":
      case "BIPED_WAKE_STRETCH":
        return this.chillPose(mode, t);
      case "BIPED_CUSTOM_UPPER": {
        const pose = this.neutralPose();
        for (const [name, j] of this.joints) {
          const deg = this.state.upper?.[j.def.motor_id];
          if (deg != null && j.def.motor_id >= 13) pose[name] = (deg * Math.PI) / 180;
        }
        return pose;
      }
      default: {
        const pose = this.neutralPose();
        const breathe = Math.sin(t * 1.6) * 0.025;
        this.leg(pose, "right", 0.06 + breathe);
        this.leg(pose, "left", 0.06 + breathe);
        return pose;
      }
    }
  }

  /** Right-side angles of a chill mode (the left mirrors them), feet and head drifting. */
  chillPose(mode, t) {
    const pose = {};
    for (const name of this.joints.keys()) pose[name] = 0;
    const r = { hip_roll: 0.06, ankle_roll: -0.06, shoulder_roll: 0.1 };
    const l = {};
    const swing = (f, ph) => Math.sin(t * 2 * Math.PI * f + ph) * (0.6 + 0.4 * Math.sin(t * 0.37 + ph));
    if (mode === "BIPED_CHILL_LIE") {
      const lift = Math.max(0, Math.sin(t * 0.21)) ** 2;
      Object.assign(r, { hip_pitch: -0.7 * lift, knee: 1.3 * lift, ankle_pitch: 0.3 * swing(0.6, 0) });
      Object.assign(l, { ankle_pitch: 0.3 * swing(0.6, 2.0) });
      r.shoulder_pitch = 0.25 * Math.max(0, Math.sin(t * 0.17 + 1));
      r.elbow = -0.3;
    } else if (mode === "BIPED_WAKE_STRETCH") {
      // wake_stretch.npz after its 1.5 s ease in: a knee shuffle, arms up 1.6–3.0 s with the toes pointed,
      // shivering until 5.1 s, all back down by 6.3 s.
      const s = t - this.modeSince - 1.5;
      const ramp = (a, b) => {
        const x = clamp((s - a) / (b - a), 0, 1);
        return x * x * (3 - 2 * x);
      };
      const up = ramp(1.6, 3.0) * (1 - ramp(5.1, 6.3));
      const shuffle = ramp(0.4, 1.0) * (1 - ramp(1.0, 1.6));
      const shake = s > 3.1 && s < 5.1 ? 0.06 * Math.sin(s * 2 * Math.PI * 4.2) : 0;
      Object.assign(r, {
        shoulder_pitch: 2.15 * up + shake, shoulder_roll: 0.09 + 0.31 * up, elbow: -0.17 * up,
        hip_pitch: 0.05 - 0.19 * shuffle, knee: 0.38 * shuffle + 0.09 * up + shake, ankle_pitch: 0.55 * up + shake,
      });
      Object.assign(l, { hip_pitch: 0.05, knee: 0.09 * up - shake, ankle_pitch: 0.55 * up - shake });
    } else {
      const chair = mode === "BIPED_CHILL_CHAIR";
      Object.assign(r, chair ? { hip_pitch: -0.7, knee: 0.75 } : { hip_pitch: -1.15, knee: 0.15 },
        { shoulder_pitch: chair ? -0.61 : -0.65, shoulder_roll: chair ? 0.1 : 0.13 });
      l.knee = (chair ? 0.75 : 0.15) + 0.2 * swing(0.7, Math.PI);
      r.knee += 0.2 * swing(0.7, 0);
      r.ankle_pitch = 0.25 * swing(0.7, 0.8);
      l.ankle_pitch = 0.25 * swing(0.7, Math.PI + 0.8);
    }
    for (const [joint, v] of Object.entries(r)) {
      pose[`right_${joint}`] = v;
      pose[`left_${joint}`] = -(joint in l ? l[joint] : v);
    }
    pose.head = 0.45 * Math.sin(t * 0.33) + 0.15 * Math.sin(t * 0.9);
    return pose;
  }

  /** The 撑起/躺下 reference at the time the robot reports (state.propup), interpolated. */
  propupKey() {
    const i = clamp((this.propupT || 0) / PROPUP_KEY_S, 0, PROPUP_KEYS.length - 1);
    const a = PROPUP_KEYS[Math.floor(i)];
    const b = PROPUP_KEYS[Math.min(PROPUP_KEYS.length - 1, Math.floor(i) + 1)];
    const f = i - Math.floor(i);
    return a.map((v, k) => v + (b[k] - v) * f);
  }

  /** The 偷看 reference [trunk roll, head] `since` seconds into the mode, interpolated. */
  peekKey(since) {
    const i = clamp(since / PEEK_KEY_S, 0, PEEK_KEYS.length - 1);
    const a = PEEK_KEYS[Math.floor(i)];
    const b = PEEK_KEYS[Math.min(PEEK_KEYS.length - 1, Math.floor(i) + 1)];
    const f = i - Math.floor(i);
    return a.map((v, k) => v + (b[k] - v) * f);
  }

  walkPose(speed) {
    const v = this.velocity();
    const pose = this.neutralPose();
    const lift = 0.16 + 0.22 * Math.min(1, speed);
    const stride = 0.32 * v.vx;
    const p = this.phase;
    for (const [side, offset] of [["right", 0], ["left", Math.PI]]) {
      const ph = p + offset;
      const up = Math.max(0, Math.sin(ph));
      const side_s = side === "right" ? -1 : 1;
      this.leg(pose, side, 0.22 + lift * up, stride * Math.cos(ph), side_s * 0.12 * v.vy * up, 0.18 * v.vtheta * up);
    }
    const arm = (0.18 + 0.4 * Math.abs(v.vx)) * Math.cos(p);
    pose.right_shoulder_pitch = arm;
    pose.left_shoulder_pitch = arm;
    pose.head_wobble_roll = 0.05 * Math.sin(p);
    return pose;
  }

  velocity() {
    const cmd = this.state.velocity || {};
    return { vx: cmd.vx || 0, vy: cmd.vy || 0, vtheta: cmd.vtheta || 0 };
  }

  applyPose(pose) {
    for (const [name, value] of Object.entries(pose)) {
      const j = this.joints.get(name);
      if (!j) continue;
      const [lo, hi] = j.def.range;
      const angle = hi > lo ? clamp(value, lo, hi) : value;
      this.angles.set(name, angle);
      j.group.quaternion.setFromAxisAngle(j.axis, angle);
    }
  }

  lowestFoot() {
    const v = new THREE.Vector3();
    let low = Infinity;
    for (const foot of this.feet) {
      foot.getWorldPosition(v);
      this.root.worldToLocal(v);
      low = Math.min(low, v.z);
    }
    return low;
  }

  settleOnGround() {
    this.root.updateMatrixWorld(true);
    this.trunk.position.z += this.soleZ - this.lowestFoot();
  }

  trunkWorld() {
    const v = new THREE.Vector3();
    this.trunk.getWorldPosition(v);
    v.y = FOCUS_Y;
    return v;
  }

  // ------------------------------------------------------------ frame loop

  frame() {
    if (document.hidden || !this.renderer || !this.container.clientWidth) return;
    const dt = Math.min(0.05, this.clock.getDelta());
    const t = performance.now() / 1000;
    const mode = this.state.mode;
    const v = this.velocity();
    const walking = mode === "BIPED_WALK_RUN" || mode === "BIPED_WALK_UPPER" || mode === "BIPED_SKATE";
    const speed = Math.min(1, Math.hypot(v.vx, v.vy) + 0.5 * Math.abs(v.vtheta));

    if (walking) {
      this.phase = (this.phase + dt * 2 * Math.PI * (1.6 + 0.6 * speed)) % (2 * Math.PI * 1000);
      this.base.yaw += v.vtheta * WALK_SPEED.vtheta * dt;
      const c = Math.cos(this.base.yaw);
      const s = Math.sin(this.base.yaw);
      const fx = v.vx * WALK_SPEED.vx;
      const fy = v.vy * WALK_SPEED.vy;
      this.base.x += (c * fx - s * fy) * dt;
      this.base.y += (s * fx + c * fy) * dt;
    }

    const propup = mode === "BIPED_PROPUP" ? this.state.propup || {} : null;
    if (propup) {
      // Runs on between the 5 Hz status reports, caught up when it drifts.
      const reported = propup.t_s ?? 0;
      const moving = propup.phase === "rising" || propup.phase === "lowering";
      this.propupT = moving ? (this.propupT ?? reported) + dt : reported;
      if (Math.abs(this.propupT - reported) > 0.3) this.propupT = reported;
    } else this.propupT = null;

    const target = this.poseFor(mode, t, speed);
    if (this.state.head && mode !== "PASSIVE_DEFAULT" && mode !== "BIPED_WALK_UPPER") target.head = 0.55 * Math.sin(t * 1.2);
    else if (mode === "BIPED_STAND_DEFAULT") target.head = 0.12 * Math.sin(t * 0.35);
    let previewing = false;
    if (this.preview) {
      const age = t - this.preview.start;
      if (age > 3.2) this.preview = null;
      else {
        const { joint } = this.preview;
        const [lo, hi] = this.joints.get(joint).def.range;
        const mid = clamp(target[joint] ?? this.joints.get(joint).def.neutral, lo, hi);
        const amp = Math.min(0.45, (hi - lo) / 2);
        target[joint] = clamp(mid + amp * Math.sin(age * 4.2) * Math.sin(Math.min(1, age / 3.2) * Math.PI), lo, hi);
        previewing = true;
      }
    }
    const rate = walking || previewing || propup || mode in CHILL_BODY || mode === "BIPED_PEEK" ? 22 : mode === "BIPED_CUSTOM_UPPER" ? 10 : 5;
    const k = 1 - Math.exp(-dt * rate);
    // Polled joints arrive at 5 Hz: follow them more softly than the generated walk.
    const live = mode === "BIPED_WALK_UPPER" ? this.state.liveUpper || {} : {};
    const kLive = 1 - Math.exp(-dt * 9);
    const blended = {};
    for (const [name, value] of Object.entries(target)) {
      const current = this.angles.get(name) ?? value;
      blended[name] = current + (value - current) * (name in live ? kLive : k);
    }
    this.applyPose(blended);

    const lying = propup ? this.propupKey().slice(4) : CHILL_BODY[mode] || null;
    const roll = walking ? 0.035 * Math.sin(this.phase) : mode === "BIPED_PEEK" ? this.peekKey(t - this.modeSince)[0] : 0;
    const pitch = lying ? lying[0] : walking ? 0.06 * v.vx : 0;
    this.trunk.quaternion.setFromEuler(new THREE.Euler(roll, pitch, this.base.yaw, "ZYX"));
    this.trunk.position.x = this.base.x;
    this.trunk.position.y = this.base.y;
    if (lying) this.trunk.position.z = lying[1];
    else this.settleOnGround();
    this.stool.visible = mode === "BIPED_CHILL_CHAIR";
    if (this.stool.visible) {
      this.stool.position.set(this.base.x, this.base.y, 0);
      this.stool.rotation.z = this.base.yaw;
    }

    const dim = mode === "PASSIVE_DEFAULT" || mode === "STOPPING" ? 0.7 : 1;
    if (dim !== this.dim) {
      this.dim = dim;
      for (const [kind, m] of Object.entries(this.materials)) m.color.copy(this.baseColors[kind]).multiplyScalar(dim);
    }
    const focusTarget = this.highlighted.size || t < this.focusReleaseAt ? 1 : 0;
    if (focusTarget !== this.focus) {
      const df = focusTarget - this.focus;
      this.applyFocus(Math.abs(df) < 0.01 ? focusTarget : this.focus + df * Math.min(1, dt * 10));
    }

    // Follow the robot with the camera and keep the grid under it (an endless floor).
    const focus = this.trunkWorld();
    const delta = focus.clone().sub(this.controls.target).multiplyScalar(Math.min(1, dt * 4));
    this.controls.target.add(delta);
    this.camera.position.add(delta);
    this.grid.position.x = Math.round(focus.x * 10) / 10;
    this.grid.position.z = Math.round(focus.z * 10) / 10;
    this.floor.position.x = focus.x;
    this.floor.position.z = focus.z;
    this.sun.position.set(focus.x + 0.8, 1.6, focus.z + 0.6);
    this.sun.target.position.set(focus.x, 0, focus.z);

    const ease = Math.min(1, dt * 8);
    const dr = this.insetTarget.right - this.inset.right;
    const db = this.insetTarget.bottom - this.inset.bottom;
    if (Math.abs(dr) > 0.5 || Math.abs(db) > 0.5) {
      this.inset.right += Math.abs(dr) < 1 ? dr : dr * ease;
      this.inset.bottom += Math.abs(db) < 1 ? db : db * ease;
      this.applyProjection();
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.onFrame?.();
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.applyProjection();
  }

  /** Keep the robot centred in the part of the stage not covered by a panel (CSS px). */
  setInset({ right = 0, bottom = 0 } = {}) {
    this.insetTarget = { right, bottom };
  }

  applyProjection() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    const { right: r, bottom: b } = this.inset;
    // Render a window of a larger virtual image whose centre is the free area's centre.
    this.camera.aspect = (w + r) / (h + b);
    if (r > 0.5 || b > 0.5) this.camera.setViewOffset(w + r, h + b, r, b, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }
}
