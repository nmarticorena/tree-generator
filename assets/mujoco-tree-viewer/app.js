import loadMujoco from "../../node_modules/@mujoco/mujoco/mujoco.js";

const MODEL_URL = "../../generated/tree/gen/pliable04/ta/raw_train/ta_0000/ternary_a.mjcf";
const MAX_SCENE_GEOMS = 5000;

const $ = (selector) => document.querySelector(selector);
const ui = {
  canvas: $("#scene"),
  viewport: $("#viewport"),
  loadingCard: $("#loading-card"),
  loadingTitle: $("#loading-title"),
  loadingDetail: $("#loading-detail"),
  statusDot: $("#status-dot"),
  statusText: $("#status-text"),
  errorToast: $("#error-toast"),
  errorMessage: $("#error-message"),
  compatibilityNote: $("#compatibility-note"),
  simTime: $("#sim-time"),
  geomCount: $("#geom-count"),
  bodyCount: $("#body-count"),
  dofCount: $("#dof-count"),
  jointCount: $("#joint-count"),
  timestep: $("#timestep"),
  contactCount: $("#contact-count"),
  runState: $("#run-state"),
  play: $("#play-button"),
  reset: $("#reset-button"),
  step: $("#step-button"),
  speed: $("#speed-select"),
  gravity: $("#gravity-toggle"),
  geometry: $("#geometry-select"),
  sites: $("#sites-toggle"),
  wireframe: $("#wireframe-toggle"),
  rotate: $("#rotate-toggle"),
  fit: $("#fit-button"),
};

const state = {
  mj: null,
  model: null,
  data: null,
  scene: null,
  option: null,
  perturb: null,
  camera: null,
  renderer: null,
  playing: false,
  speed: 1,
  accumulator: 0,
  lastFrame: performance.now(),
  needsSceneUpdate: true,
  needsRender: true,
  timestep: 0.002,
  originalGravity: [0, 0, -9.81],
  bounds: { center: [0, 0, 0.8], radius: 1.4 },
  orbit: { azimuth: 48, elevation: 24, distance: 3.2, target: [0, 0, 0.8] },
  drag: {
    active: false,
    bodyId: -1,
    geomId: -1,
    distance: 0,
    localPoint: [0, 0, 0],
    anchor: [0, 0, 0],
    target: [0, 0, 0],
  },
  geomBodyIds: null,
  bodyParentIds: null,
  bodyDofCounts: null,
  bodyMasses: null,
  bodyPositions: null,
  bodyMatrices: null,
  qfrcApplied: null,
};

function setLoading(title, detail) {
  ui.loadingTitle.textContent = title;
  ui.loadingDetail.textContent = detail;
  ui.statusText.textContent = detail;
}

function setReady() {
  ui.loadingCard.classList.add("hidden");
  ui.statusDot.classList.remove("loading");
  ui.statusText.textContent = "MuJoCo 3.11 · ready";
  for (const control of [ui.play, ui.reset, ui.step, ui.speed, ui.gravity, ui.geometry, ui.sites, ui.wireframe, ui.rotate, ui.fit]) {
    control.disabled = false;
  }
}

function showError(error) {
  const message = error instanceof Error ? error.message : String(error);
  ui.statusDot.className = "status-dot error";
  ui.statusText.textContent = "Viewer error";
  ui.errorMessage.textContent = location.protocol === "file:"
    ? "This page must be served over HTTP. Run `npm run viewer` from the repository root."
    : message;
  ui.errorToast.hidden = false;
  ui.loadingCard.classList.add("hidden");
  console.error(error);
}

function normalizeGeneratedModel(xml) {
  const reservedWorldBody = /<body\s+name=(['"])world\1/;
  const normalized = reservedWorldBody.test(xml);
  if (normalized) {
    xml = xml.replace(reservedWorldBody, '<body name="tree-root"');
  }

  return {
    xml,
    normalized,
  };
}

async function boot() {
  if (location.protocol === "file:") throw new Error("HTTP server required");

  setLoading("Starting MuJoCo", "Loading the WebAssembly runtime");
  const [mj, response] = await Promise.all([
    loadMujoco(),
    fetch(MODEL_URL),
  ]);
  if (!response.ok) throw new Error(`Model request failed: ${response.status} ${response.statusText}`);

  setLoading("Compiling model", "Fetching and parsing ternary_a.mjcf");
  const source = normalizeGeneratedModel(await response.text());
  ui.compatibilityNote.hidden = !source.normalized;

  state.mj = mj;
  state.model = mj.MjModel.from_xml_string(source.xml);
  state.data = new mj.MjData(state.model);
  state.option = new mj.MjvOption();
  state.perturb = new mj.MjvPerturb();
  state.camera = new mj.MjvCamera();
  state.scene = new mj.MjvScene(state.model, MAX_SCENE_GEOMS);

  mj.mjv_defaultOption(state.option);
  mj.mjv_defaultPerturb(state.perturb);
  mj.mjv_defaultFreeCamera(state.model, state.camera);
  mj.mj_forward(state.model, state.data);

  const modelOptions = state.model.opt;
  state.timestep = modelOptions.timestep;
  state.originalGravity = Array.from(modelOptions.gravity);
  modelOptions.delete();
  state.geomBodyIds = state.model.geom_bodyid;
  state.bodyParentIds = state.model.body_parentid;
  state.bodyDofCounts = state.model.body_dofnum;
  state.bodyMasses = state.model.body_mass;
  state.bodyPositions = state.data.xpos;
  state.bodyMatrices = state.data.xmat;
  state.qfrcApplied = state.data.qfrc_applied;
  configureSceneGroups();
  updateMjvScene();

  setLoading("Preparing renderer", "Building WebGL geometry");
  state.renderer = new Renderer(ui.canvas);
  state.bounds = computeSceneBounds();
  fitCamera();
  populateModelInfo();
  bindControls();
  setReady();
  requestAnimationFrame(frame);
}

function configureSceneGroups() {
  const mode = ui.geometry.value;
  state.option.geomgroup.fill(0);
  state.option.geomgroup[1] = mode === "collision" || mode === "both" ? 1 : 0;
  state.option.geomgroup[2] = mode === "visual" || mode === "both" ? 1 : 0;
  state.option.sitegroup.fill(ui.sites.checked ? 1 : 0);
  state.needsSceneUpdate = true;
  state.needsRender = true;
}

function updateMjvScene() {
  state.mj.mjv_updateScene(
    state.model,
    state.data,
    state.option,
    state.perturb,
    state.camera,
    state.mj.mjtCatBit.mjCAT_ALL.value,
    state.scene,
  );
  state.needsSceneUpdate = false;
  state.needsRender = true;
  ui.geomCount.textContent = state.scene.ngeom.toLocaleString();
}

function populateModelInfo() {
  ui.bodyCount.textContent = state.model.nbody.toLocaleString();
  ui.dofCount.textContent = state.model.nv.toLocaleString();
  ui.jointCount.textContent = state.model.njnt.toLocaleString();
  ui.timestep.textContent = `${(state.timestep * 1000).toFixed(1)} ms`;
  updateStats();
}

function updateStats() {
  ui.simTime.textContent = `${state.data.time.toFixed(3)} s`;
  ui.contactCount.textContent = state.data.ncon.toLocaleString();
}

function setPlaying(playing) {
  state.playing = playing;
  state.accumulator = 0;
  ui.play.classList.toggle("is-playing", playing);
  ui.play.querySelector("span").textContent = playing ? "Pause" : "Play";
  ui.runState.textContent = playing ? "RUNNING" : "PAUSED";
  ui.runState.classList.toggle("running", playing);
}

function resetSimulation() {
  endObjectDrag();
  state.mj.mj_resetData(state.model, state.data);
  applyGravitySetting();
  state.mj.mj_forward(state.model, state.data);
  state.accumulator = 0;
  state.needsSceneUpdate = true;
  state.needsRender = true;
  updateStats();
}

function stepSimulation() {
  state.mj.mj_step(state.model, state.data);
  state.needsSceneUpdate = true;
  state.needsRender = true;
  updateStats();
}

function applyGravitySetting() {
  const modelOptions = state.model.opt;
  const gravity = modelOptions.gravity;
  for (let i = 0; i < 3; i += 1) gravity[i] = ui.gravity.checked ? state.originalGravity[i] : 0;
  modelOptions.delete();
}

function bindControls() {
  ui.play.addEventListener("click", () => setPlaying(!state.playing));
  ui.reset.addEventListener("click", resetSimulation);
  ui.step.addEventListener("click", () => {
    setPlaying(false);
    stepSimulation();
  });
  ui.speed.addEventListener("change", () => { state.speed = Number(ui.speed.value); });
  ui.gravity.addEventListener("change", applyGravitySetting);
  ui.geometry.addEventListener("change", configureSceneGroups);
  ui.sites.addEventListener("change", configureSceneGroups);
  ui.wireframe.addEventListener("change", () => {
    state.renderer.wireframe = ui.wireframe.checked;
    state.needsRender = true;
  });
  ui.fit.addEventListener("click", fitCamera);

  const pointer = { active: false, mode: "camera", x: 0, y: 0, pan: false };
  ui.canvas.addEventListener("pointerdown", (event) => {
    pointer.active = true;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    pointer.pan = event.shiftKey || event.button === 1 || event.button === 2;
    pointer.mode = "camera";

    if (event.button === 0 && !event.shiftKey) {
      const hit = pickScene(event.clientX, event.clientY);
      if (hit && bodyCanMove(hit.bodyId)) {
        startObjectDrag(hit, event.clientX, event.clientY);
        pointer.mode = "object";
      }
    }
    ui.canvas.setPointerCapture(event.pointerId);
  });
  ui.canvas.addEventListener("pointermove", (event) => {
    if (!pointer.active) return;
    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    if (pointer.mode === "object") {
      moveObjectDrag(event.clientX, event.clientY);
    } else if (pointer.pan) panCamera(dx, dy);
    else {
      state.orbit.azimuth -= dx * 0.32;
      state.orbit.elevation = clamp(state.orbit.elevation + dy * 0.25, -85, 85);
    }
    state.needsRender = true;
  });
  const endPointer = () => {
    if (pointer.mode === "object") endObjectDrag();
    pointer.active = false;
    pointer.mode = "camera";
  };
  ui.canvas.addEventListener("pointerup", endPointer);
  ui.canvas.addEventListener("pointercancel", endPointer);
  ui.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  ui.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.orbit.distance = clamp(state.orbit.distance * Math.exp(event.deltaY * 0.001), 0.08, 100);
    state.needsRender = true;
  }, { passive: false });
}

function startObjectDrag(hit) {
  state.drag.active = true;
  state.drag.bodyId = hit.bodyId;
  state.drag.geomId = hit.geomId;
  state.drag.distance = hit.distance;
  state.drag.localPoint = bodyWorldToLocal(hit.bodyId, hit.point);
  state.drag.anchor = [...hit.point];
  state.drag.target = [...hit.point];
  state.accumulator = 0;
  state.needsRender = true;
  ui.canvas.classList.add("dragging-object");
  ui.statusText.textContent = `Pulling body ${hit.bodyId}`;
}

function moveObjectDrag(clientX, clientY) {
  if (!state.drag.active) return;
  const ray = screenRay(clientX, clientY);
  state.drag.target = addScaled(ray.origin, ray.direction, state.drag.distance);
  state.needsRender = true;
}

function endObjectDrag() {
  if (!state.drag.active) return;
  state.drag.active = false;
  state.drag.bodyId = -1;
  state.drag.geomId = -1;
  state.qfrcApplied?.fill(0);
  state.accumulator = 0;
  state.needsRender = true;
  ui.canvas.classList.remove("dragging-object");
  ui.statusText.textContent = "MuJoCo 3.11 · ready";
}

function applyMouseSpring() {
  state.qfrcApplied.fill(0);
  if (!state.drag.active) return;

  const anchor = bodyLocalToWorld(state.drag.bodyId, state.drag.localPoint);
  state.drag.anchor = anchor;
  const displacement = subtract(state.drag.target, anchor);
  const mass = Math.max(state.bodyMasses[state.drag.bodyId], 0.05);
  const stiffness = clamp(mass * 500, 50, 600);
  let force = displacement.map((value) => value * stiffness);
  const magnitude = Math.hypot(...force);
  if (magnitude > 150) force = force.map((value) => value * 150 / magnitude);

  state.mj.mj_applyFT(
    state.model,
    state.data,
    force,
    [0, 0, 0],
    anchor,
    state.drag.bodyId,
    state.qfrcApplied,
  );
}

function bodyCanMove(bodyId) {
  for (let body = bodyId; body > 0; body = state.bodyParentIds[body]) {
    if (state.bodyDofCounts[body] > 0) return true;
  }
  return false;
}

function bodyWorldToLocal(bodyId, point) {
  const positionOffset = bodyId * 3;
  const matrixOffset = bodyId * 9;
  const delta = [
    point[0] - state.bodyPositions[positionOffset],
    point[1] - state.bodyPositions[positionOffset + 1],
    point[2] - state.bodyPositions[positionOffset + 2],
  ];
  const matrix = state.bodyMatrices;
  return [
    matrix[matrixOffset] * delta[0] + matrix[matrixOffset + 3] * delta[1] + matrix[matrixOffset + 6] * delta[2],
    matrix[matrixOffset + 1] * delta[0] + matrix[matrixOffset + 4] * delta[1] + matrix[matrixOffset + 7] * delta[2],
    matrix[matrixOffset + 2] * delta[0] + matrix[matrixOffset + 5] * delta[1] + matrix[matrixOffset + 8] * delta[2],
  ];
}

function bodyLocalToWorld(bodyId, point) {
  const positionOffset = bodyId * 3;
  const matrixOffset = bodyId * 9;
  const matrix = state.bodyMatrices;
  return [
    state.bodyPositions[positionOffset] + matrix[matrixOffset] * point[0] + matrix[matrixOffset + 1] * point[1] + matrix[matrixOffset + 2] * point[2],
    state.bodyPositions[positionOffset + 1] + matrix[matrixOffset + 3] * point[0] + matrix[matrixOffset + 4] * point[1] + matrix[matrixOffset + 5] * point[2],
    state.bodyPositions[positionOffset + 2] + matrix[matrixOffset + 6] * point[0] + matrix[matrixOffset + 7] * point[1] + matrix[matrixOffset + 8] * point[2],
  ];
}

function pickScene(clientX, clientY) {
  const ray = screenRay(clientX, clientY);
  const geoms = state.scene.geoms;
  let closest = null;
  try {
    for (let index = 0; index < state.scene.ngeom; index += 1) {
      const geom = geoms.get(index);
      if (geom.objtype === 5 && geom.objid >= 0) {
        const distance = intersectGeom(ray, geom);
        if (distance !== null && distance > 0 && (!closest || distance < closest.distance)) {
          const bodyId = state.geomBodyIds[geom.objid];
          closest = {
            distance,
            bodyId,
            geomId: geom.objid,
            point: addScaled(ray.origin, ray.direction, distance),
          };
        }
      }
      geom.delete();
    }
  } finally {
    geoms.delete();
  }
  return closest;
}

function screenRay(clientX, clientY) {
  const rect = ui.canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = 1 - ((clientY - rect.top) / rect.height) * 2;
  const camera = cameraVectors(state.orbit);
  const halfHeight = Math.tan(radians(42) * 0.5);
  const aspect = rect.width / Math.max(rect.height, 1);
  const direction = normalize([
    camera.forward[0] + camera.right[0] * x * halfHeight * aspect + camera.up[0] * y * halfHeight,
    camera.forward[1] + camera.right[1] * x * halfHeight * aspect + camera.up[1] * y * halfHeight,
    camera.forward[2] + camera.right[2] * x * halfHeight * aspect + camera.up[2] * y * halfHeight,
  ]);
  return { origin: camera.eye, direction };
}

function intersectGeom(ray, geom) {
  const local = rayToGeomLocal(ray, geom);
  if (geom.type === 5) return intersectCylinder(local.origin, local.direction, geom.size[0], geom.size[2]);
  if (geom.type === 2 || geom.type === 4) {
    return intersectEllipsoid(local.origin, local.direction, geom.size);
  }
  if (geom.type === 6) return intersectBox(local.origin, local.direction, geom.size);
  return null;
}

function rayToGeomLocal(ray, geom) {
  const delta = subtract(ray.origin, Array.from(geom.pos));
  const matrix = geom.mat;
  return {
    origin: [
      matrix[0] * delta[0] + matrix[3] * delta[1] + matrix[6] * delta[2],
      matrix[1] * delta[0] + matrix[4] * delta[1] + matrix[7] * delta[2],
      matrix[2] * delta[0] + matrix[5] * delta[1] + matrix[8] * delta[2],
    ],
    direction: [
      matrix[0] * ray.direction[0] + matrix[3] * ray.direction[1] + matrix[6] * ray.direction[2],
      matrix[1] * ray.direction[0] + matrix[4] * ray.direction[1] + matrix[7] * ray.direction[2],
      matrix[2] * ray.direction[0] + matrix[5] * ray.direction[1] + matrix[8] * ray.direction[2],
    ],
  };
}

function intersectCylinder(origin, direction, radius, halfHeight) {
  let nearest = Infinity;
  const a = direction[0] ** 2 + direction[1] ** 2;
  const b = 2 * (origin[0] * direction[0] + origin[1] * direction[1]);
  const c = origin[0] ** 2 + origin[1] ** 2 - radius ** 2;
  const discriminant = b ** 2 - 4 * a * c;
  if (a > 1e-10 && discriminant >= 0) {
    const root = Math.sqrt(discriminant);
    for (const distance of [(-b - root) / (2 * a), (-b + root) / (2 * a)]) {
      const z = origin[2] + distance * direction[2];
      if (distance > 0 && Math.abs(z) <= halfHeight) nearest = Math.min(nearest, distance);
    }
  }
  if (Math.abs(direction[2]) > 1e-10) {
    for (const z of [-halfHeight, halfHeight]) {
      const distance = (z - origin[2]) / direction[2];
      const x = origin[0] + distance * direction[0];
      const y = origin[1] + distance * direction[1];
      if (distance > 0 && x ** 2 + y ** 2 <= radius ** 2) nearest = Math.min(nearest, distance);
    }
  }
  return Number.isFinite(nearest) ? nearest : null;
}

function intersectEllipsoid(origin, direction, size) {
  const scaledOrigin = origin.map((value, index) => value / Math.max(size[index], 1e-8));
  const scaledDirection = direction.map((value, index) => value / Math.max(size[index], 1e-8));
  const a = dot(scaledDirection, scaledDirection);
  const b = 2 * dot(scaledOrigin, scaledDirection);
  const c = dot(scaledOrigin, scaledOrigin) - 1;
  const discriminant = b ** 2 - 4 * a * c;
  if (discriminant < 0) return null;
  const root = Math.sqrt(discriminant);
  const distances = [(-b - root) / (2 * a), (-b + root) / (2 * a)].filter((value) => value > 0);
  return distances.length ? Math.min(...distances) : null;
}

function intersectBox(origin, direction, size) {
  let near = -Infinity;
  let far = Infinity;
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(direction[axis]) < 1e-10) {
      if (Math.abs(origin[axis]) > size[axis]) return null;
      continue;
    }
    const a = (-size[axis] - origin[axis]) / direction[axis];
    const b = (size[axis] - origin[axis]) / direction[axis];
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
    if (near > far) return null;
  }
  if (far < 0) return null;
  return near > 0 ? near : far;
}

function panCamera(dx, dy) {
  const az = radians(state.orbit.azimuth);
  const el = radians(state.orbit.elevation);
  const right = [-Math.sin(az), Math.cos(az), 0];
  const forward = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
  const up = cross(right, forward);
  const scale = state.orbit.distance * 0.0016;
  for (let i = 0; i < 3; i += 1) {
    state.orbit.target[i] += (-dx * right[i] + dy * up[i]) * scale;
  }
}

function fitCamera() {
  state.orbit.target = [...state.bounds.center];
  state.orbit.distance = Math.max(state.bounds.radius * 2.5, 0.4);
  state.orbit.azimuth = 48;
  state.orbit.elevation = 24;
  state.needsRender = true;
}

function computeSceneBounds() {
  if (!state.scene.ngeom) return { center: [0, 0, 0], radius: 1 };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const geoms = state.scene.geoms;
  try {
    for (let index = 0; index < state.scene.ngeom; index += 1) {
      const geom = geoms.get(index);
      const extent = Math.max(geom.size[0], geom.size[1], geom.size[2]);
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], geom.pos[axis] - extent);
        max[axis] = Math.max(max[axis], geom.pos[axis] + extent);
      }
      geom.delete();
    }
  } finally {
    geoms.delete();
  }
  const center = min.map((value, axis) => (value + max[axis]) * 0.5);
  const radius = Math.max(...max.map((value, axis) => value - min[axis])) * 0.55;
  return { center, radius: Math.max(radius, 0.25) };
}

function frame(now) {
  const elapsed = Math.min((now - state.lastFrame) / 1000, 0.1);
  state.lastFrame = now;

  if (state.playing || state.drag.active) {
    state.accumulator += elapsed * (state.playing ? state.speed : 1);
    const timestep = state.timestep;
    let steps = 0;
    while (state.accumulator >= timestep && steps < 40) {
      applyMouseSpring();
      state.mj.mj_step(state.model, state.data);
      state.accumulator -= timestep;
      steps += 1;
    }
    if (steps) {
      state.needsSceneUpdate = true;
      updateStats();
    }
  }

  if (state.drag.active) {
    state.drag.anchor = bodyLocalToWorld(state.drag.bodyId, state.drag.localPoint);
    state.needsRender = true;
  }

  if (ui.rotate.checked) {
    state.orbit.azimuth += elapsed * 7;
    state.needsRender = true;
  }
  if (state.needsSceneUpdate) updateMjvScene();
  if (state.needsRender || state.renderer.dirty) {
    state.renderer.render(state.scene, state.orbit, state.drag, state.geomBodyIds);
    state.needsRender = false;
  }
  requestAnimationFrame(frame);
}

class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
    if (!this.gl) throw new Error("WebGL 2 is required by this viewer.");
    this.wireframe = false;
    this.dirty = true;
    this.program = createProgram(this.gl, MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER);
    this.gridProgram = createProgram(this.gl, GRID_VERTEX_SHADER, GRID_FRAGMENT_SHADER);
    this.viewProjectionLocation = this.gl.getUniformLocation(this.program, "uViewProjection");
    this.lightLocation = this.gl.getUniformLocation(this.program, "uLightDirection");
    this.gridViewProjectionLocation = this.gl.getUniformLocation(this.gridProgram, "uViewProjection");
    this.gridColorLocation = this.gl.getUniformLocation(this.gridProgram, "uColor");
    this.meshes = {
      sphere: this.createMesh(makeSphere(14, 20)),
      cylinder: this.createMesh(makeCylinder(20)),
      box: this.createMesh(makeBox()),
    };
    this.grid = this.createGrid();
    this.interactionLine = this.createInteractionLine();
    this.resizeObserver = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObserver.observe(canvas);
    this.resize();
  }

  createMesh(mesh) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.vertices), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.indices), gl.STATIC_DRAW);

    const instanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
    const stride = 20 * 4;
    for (let column = 0; column < 4; column += 1) {
      const location = 2 + column;
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 4, gl.FLOAT, false, stride, column * 16);
      gl.vertexAttribDivisor(location, 1);
    }
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 4, gl.FLOAT, false, stride, 64);
    gl.vertexAttribDivisor(6, 1);
    gl.bindVertexArray(null);
    return { vao, instanceBuffer, indexCount: mesh.indices.length, instances: [] };
  }

  createGrid() {
    const gl = this.gl;
    const vertices = [];
    const half = 12;
    for (let i = -half; i <= half; i += 1) {
      const major = i === 0 ? 0.22 : i % 5 === 0 ? 0.11 : 0.045;
      vertices.push(-half, i, 0, major, half, i, 0, major);
      vertices.push(i, -half, 0, major, i, half, 0, major);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 12);
    gl.bindVertexArray(null);
    return { vao, vertexCount: vertices.length / 4 };
  }

  createInteractionLine() {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, 8 * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 12);
    gl.bindVertexArray(null);
    return { vao, buffer };
  }

  resize() {
    const scale = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * scale));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * scale));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  render(scene, orbit, drag, geomBodyIds) {
    const gl = this.gl;
    this.resize();
    this.dirty = false;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.045, 0.068, 0.054, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const viewProjection = cameraMatrix(orbit, this.canvas.width / this.canvas.height);
    this.drawGrid(viewProjection);
    for (const mesh of Object.values(this.meshes)) mesh.instances.length = 0;

    const geoms = scene.geoms;
    try {
      for (let index = 0; index < scene.ngeom; index += 1) {
        const geom = geoms.get(index);
        const mesh = geom.type === 5
          ? this.meshes.cylinder
          : (geom.type === 2 || geom.type === 4)
            ? this.meshes.sphere
            : geom.type === 6
              ? this.meshes.box
              : null;
        const selected = drag.active && geom.objtype === 5 && geomBodyIds[geom.objid] === drag.bodyId;
        if (mesh) pushInstance(mesh.instances, geom, selected);
        geom.delete();
      }
    } finally {
      geoms.delete();
    }

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.viewProjectionLocation, false, viewProjection);
    gl.uniform3f(this.lightLocation, 0.42, -0.34, 0.84);
    for (const mesh of Object.values(this.meshes)) this.drawMesh(mesh);
    if (drag.active) this.drawInteraction(viewProjection, drag.anchor, drag.target);
    gl.bindVertexArray(null);
  }

  drawMesh(mesh) {
    if (!mesh.instances.length) return;
    const gl = this.gl;
    gl.bindVertexArray(mesh.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.instances), gl.DYNAMIC_DRAW);
    gl.drawElementsInstanced(
      this.wireframe ? gl.LINES : gl.TRIANGLES,
      mesh.indexCount,
      gl.UNSIGNED_SHORT,
      0,
      mesh.instances.length / 20,
    );
  }

  drawGrid(viewProjection) {
    const gl = this.gl;
    gl.useProgram(this.gridProgram);
    gl.uniformMatrix4fv(this.gridViewProjectionLocation, false, viewProjection);
    gl.uniform3f(this.gridColorLocation, 0.48, 0.62, 0.52);
    gl.bindVertexArray(this.grid.vao);
    gl.drawArrays(gl.LINES, 0, this.grid.vertexCount);
  }

  drawInteraction(viewProjection, anchor, target) {
    const gl = this.gl;
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.gridProgram);
    gl.uniformMatrix4fv(this.gridViewProjectionLocation, false, viewProjection);
    gl.uniform3f(this.gridColorLocation, 0.72, 0.95, 0.32);
    gl.bindVertexArray(this.interactionLine.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.interactionLine.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      anchor[0], anchor[1], anchor[2], 1,
      target[0], target[1], target[2], 1,
    ]), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.LINES, 0, 2);
    gl.enable(gl.DEPTH_TEST);
  }
}

function pushInstance(target, geom, selected = false) {
  const sx = geom.size[0];
  const sy = geom.size[1];
  const sz = geom.size[2];
  const mat = geom.mat;
  target.push(
    mat[0] * sx, mat[3] * sx, mat[6] * sx, 0,
    mat[1] * sy, mat[4] * sy, mat[7] * sy, 0,
    mat[2] * sz, mat[5] * sz, mat[8] * sz, 0,
    geom.pos[0], geom.pos[1], geom.pos[2], 1,
    selected ? 0.72 : geom.rgba[0],
    selected ? 0.95 : geom.rgba[1],
    selected ? 0.32 : geom.rgba[2],
    Math.max(geom.rgba[3], 0.12),
  );
}

function makeCylinder(segments) {
  const vertices = [];
  const indices = [];
  for (let i = 0; i <= segments; i += 1) {
    const angle = (i / segments) * Math.PI * 2;
    const x = Math.cos(angle);
    const y = Math.sin(angle);
    vertices.push(x, y, -1, x, y, 0, x, y, 1, x, y, 0);
  }
  for (let i = 0; i < segments; i += 1) {
    const a = i * 2;
    indices.push(a, a + 1, a + 3, a, a + 3, a + 2);
  }
  for (const z of [-1, 1]) {
    const center = vertices.length / 6;
    vertices.push(0, 0, z, 0, 0, z);
    const ring = vertices.length / 6;
    for (let i = 0; i <= segments; i += 1) {
      const angle = (i / segments) * Math.PI * 2;
      vertices.push(Math.cos(angle), Math.sin(angle), z, 0, 0, z);
    }
    for (let i = 0; i < segments; i += 1) {
      if (z > 0) indices.push(center, ring + i, ring + i + 1);
      else indices.push(center, ring + i + 1, ring + i);
    }
  }
  return { vertices, indices };
}

function makeSphere(rows, columns) {
  const vertices = [];
  const indices = [];
  for (let row = 0; row <= rows; row += 1) {
    const phi = (row / rows) * Math.PI;
    const z = Math.cos(phi);
    const radius = Math.sin(phi);
    for (let column = 0; column <= columns; column += 1) {
      const theta = (column / columns) * Math.PI * 2;
      const x = radius * Math.cos(theta);
      const y = radius * Math.sin(theta);
      vertices.push(x, y, z, x, y, z);
    }
  }
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const a = row * (columns + 1) + column;
      const b = a + columns + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return { vertices, indices };
}

function makeBox() {
  const vertices = [];
  const indices = [];
  const faces = [
    [[1, 0, 0], [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]],
    [[-1, 0, 0], [[-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1]]],
    [[0, 1, 0], [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]]],
    [[0, -1, 0], [[1, -1, -1], [-1, -1, -1], [-1, -1, 1], [1, -1, 1]]],
    [[0, 0, 1], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]],
    [[0, 0, -1], [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]]],
  ];
  for (const [normal, corners] of faces) {
    const start = vertices.length / 6;
    for (const corner of corners) vertices.push(...corner, ...normal);
    indices.push(start, start + 1, start + 2, start, start + 2, start + 3);
  }
  return { vertices, indices };
}

function cameraMatrix(orbit, aspect) {
  const { eye } = cameraVectors(orbit);
  const projection = perspective(radians(42), aspect, Math.max(orbit.distance / 1000, 0.002), orbit.distance * 40 + 100);
  return multiply4(projection, lookAt(eye, orbit.target, [0, 0, 1]));
}

function cameraVectors(orbit) {
  const azimuth = radians(orbit.azimuth);
  const elevation = radians(orbit.elevation);
  const eye = [
    orbit.target[0] + orbit.distance * Math.cos(elevation) * Math.cos(azimuth),
    orbit.target[1] + orbit.distance * Math.cos(elevation) * Math.sin(azimuth),
    orbit.target[2] + orbit.distance * Math.sin(elevation),
  ];
  const forward = normalize(subtract(orbit.target, eye));
  const right = normalize(cross(forward, [0, 0, 1]));
  const up = normalize(cross(right, forward));
  return { eye, forward, right, up };
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(eye, center, up) {
  const z = normalize(subtract(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function multiply4(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3];
    }
  }
  return out;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  return program;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function radians(degrees) { return degrees * Math.PI / 180; }
function subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function addScaled(origin, direction, distance) { return origin.map((value, index) => value + direction[index] * distance); }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize(value) { const length = Math.hypot(...value) || 1; return value.map((component) => component / length); }

const MESH_VERTEX_SHADER = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 aPosition;
  layout(location = 1) in vec3 aNormal;
  layout(location = 2) in vec4 iModel0;
  layout(location = 3) in vec4 iModel1;
  layout(location = 4) in vec4 iModel2;
  layout(location = 5) in vec4 iModel3;
  layout(location = 6) in vec4 iColor;
  uniform mat4 uViewProjection;
  out vec3 vNormal;
  out vec3 vWorldPosition;
  out vec4 vColor;
  void main() {
    mat4 model = mat4(iModel0, iModel1, iModel2, iModel3);
    vec4 world = model * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vNormal = normalize(mat3(model) * aNormal);
    vColor = iColor;
    gl_Position = uViewProjection * world;
  }
`;

const MESH_FRAGMENT_SHADER = `#version 300 es
  precision highp float;
  in vec3 vNormal;
  in vec3 vWorldPosition;
  in vec4 vColor;
  uniform vec3 uLightDirection;
  out vec4 outColor;
  void main() {
    float diffuse = max(dot(normalize(vNormal), normalize(uLightDirection)), 0.0);
    float hemi = 0.5 + 0.5 * max(vNormal.z, -0.5);
    float light = 0.24 + diffuse * 0.62 + hemi * 0.18;
    vec3 color = vColor.rgb * light;
    color += vec3(0.05, 0.08, 0.055) * pow(max(diffuse, 0.0), 8.0);
    outColor = vec4(color, vColor.a);
  }
`;

const GRID_VERTEX_SHADER = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 aPosition;
  layout(location = 1) in float aAlpha;
  uniform mat4 uViewProjection;
  out float vAlpha;
  void main() {
    vAlpha = aAlpha;
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
  }
`;

const GRID_FRAGMENT_SHADER = `#version 300 es
  precision highp float;
  in float vAlpha;
  uniform vec3 uColor;
  out vec4 outColor;
  void main() { outColor = vec4(uColor, vAlpha); }
`;

window.addEventListener("beforeunload", () => {
  for (const object of [state.scene, state.camera, state.perturb, state.option, state.data, state.model]) {
    if (object) object.delete();
  }
});

boot().catch(showError);
