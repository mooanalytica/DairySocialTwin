const state = {
  config: null,
  classes: [],
  groups: [],
  activeGroupId: new URLSearchParams(window.location.search).get("group"),
  annotation: null,
  image: new Image(),
  imageReady: false,
  activeClassId: "walkable_ground",
  currentPoints: [],
  selectedId: null,
  zoom: 1,
  fitZoom: 1,
  dirty: false,
};

const DEFAULT_OVERLAP_PRIORITY = ["resource", "obstacle", "walkable_ground"];
const RECTANGLE_FIT_METHOD = "minimum_area_rotated_bounding_rectangle_from_polygon_points";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const stageShell = document.getElementById("stageShell");
const classButtons = document.getElementById("classButtons");
const polygonList = document.getElementById("polygonList");
const saveStatus = document.getElementById("saveStatus");
const frameMeta = document.getElementById("frameMeta");
const groupSelect = document.getElementById("groupSelect");
const missingFrame = document.getElementById("missingFrame");
const zoomLabel = document.getElementById("zoomLabel");

const buttons = {
  undo: document.getElementById("undoPointBtn"),
  close: document.getElementById("closePolygonBtn"),
  cancel: document.getElementById("cancelPolygonBtn"),
  delete: document.getElementById("deletePolygonBtn"),
  clearLayer: document.getElementById("clearLayerBtn"),
  defaultBlue: document.getElementById("defaultBlueBtn"),
  zoomOut: document.getElementById("zoomOutBtn"),
  fit: document.getElementById("fitBtn"),
  zoomIn: document.getElementById("zoomInBtn"),
  save: document.getElementById("saveBtn"),
};

function classById(classId) {
  return state.classes.find((item) => item.id === classId);
}

function planeGeometryFor(force2DRectangle) {
  if (force2DRectangle) {
    return {
      type: "rectangle",
      enforcedIn2D: true,
      rectangleFitMethod: RECTANGLE_FIT_METHOD,
    };
  }
  return {
    type: "polygon",
    enforcedIn2D: false,
  };
}

function normalizePolygon(polygon) {
  const force2DRectangle =
    Boolean(polygon.force2DRectangle) ||
    (polygon.planeGeometry && polygon.planeGeometry.type === "rectangle");
  polygon.force2DRectangle = force2DRectangle;
  polygon.planeGeometry = planeGeometryFor(force2DRectangle);
  return polygon;
}

function polygonCounts() {
  const counts = {};
  for (const classItem of state.classes) counts[classItem.id] = 0;
  for (const polygon of state.annotation.polygons) {
    counts[polygon.classId] = (counts[polygon.classId] || 0) + 1;
  }
  return counts;
}

function groupApiUrl(path, groupId = state.activeGroupId) {
  if (!groupId) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}group=${encodeURIComponent(groupId)}`;
}

function renderGroupSelect() {
  groupSelect.replaceChildren();
  for (const group of state.groups) {
    const option = document.createElement("option");
    option.value = group.id;
    const status = group.annotationExists ? "done" : group.frameExists ? "frame" : "missing";
    option.textContent = `${group.label || group.id} (${status})`;
    groupSelect.appendChild(option);
  }
  groupSelect.value = state.activeGroupId || "";
}

async function loadConfig(groupId = state.activeGroupId) {
  state.imageReady = false;
  state.currentPoints = [];
  state.selectedId = null;
  if (canvas.width && canvas.height) ctx.clearRect(0, 0, canvas.width, canvas.height);

  const response = await fetch(groupApiUrl("/api/config", groupId));
  if (!response.ok) throw new Error(`config failed: ${response.status}`);
  state.config = await response.json();
  state.groups = state.config.groups || [];
  state.activeGroupId = state.config.activeGroupId;
  state.classes = state.config.classes;
  state.annotation = state.config.annotation;
  if (!("defaultUnannotatedClassId" in state.annotation)) {
    state.annotation.defaultUnannotatedClassId = null;
  }
  if (!Array.isArray(state.annotation.overlapPriorityClassIds)) {
    state.annotation.overlapPriorityClassIds = DEFAULT_OVERLAP_PRIORITY;
  }
  state.annotation.polygons = state.annotation.polygons.map(normalizePolygon);
  state.activeClassId = state.classes[0].id;
  missingFrame.hidden = state.config.frame.exists;
  renderGroupSelect();
  renderClassButtons();
  renderPolygonList();
  updateToolbar();

  const meta = state.config.frame.metadata || {};
  const groupLabel = state.config.activeGroup?.label || state.activeGroupId;
  const frameLabel = meta.sourceVideoName
    ? `${meta.sourceVideoName} @ ${meta.timestamp}`
    : "reference frame";
  frameMeta.textContent = `${groupLabel} - ${frameLabel}`;

  if (state.config.frame.exists) {
    state.image = new Image();
    state.image.onload = () => {
      state.imageReady = true;
      canvas.width = state.image.naturalWidth;
      canvas.height = state.image.naturalHeight;
      fitToStage();
      draw();
    };
    state.image.src = state.config.frame.url;
  } else {
    saveStatus.textContent = `Missing frame for ${groupLabel}`;
  }
  history.replaceState(null, "", `?group=${encodeURIComponent(state.activeGroupId)}`);
}

function renderClassButtons() {
  const counts = polygonCounts();
  classButtons.replaceChildren();
  for (const classItem of state.classes) {
    const button = document.createElement("button");
    button.className = `class-button ${state.activeClassId === classItem.id ? "active" : ""}`;
    button.dataset.classId = classItem.id;
    button.innerHTML = `
      <span class="swatch" style="background:${classItem.color}"></span>
      <span>${classItem.uiLabel}</span>
      <span class="count-pill">${counts[classItem.id] || 0}</span>
    `;
    button.addEventListener("click", () => {
      state.activeClassId = classItem.id;
      renderClassButtons();
      updateToolbar();
      draw();
    });
    classButtons.appendChild(button);
  }
}

function renderPolygonList() {
  polygonList.replaceChildren();
  for (const polygon of state.annotation.polygons) {
    const classItem = classById(polygon.classId);
    const row = document.createElement("div");
    row.className = `polygon-row ${state.selectedId === polygon.id ? "active" : ""}`;
    const selectButton = document.createElement("button");
    selectButton.className = "polygon-select";
    selectButton.innerHTML = `
      <span class="swatch" style="background:${polygon.color || classItem.color}"></span>
      <span class="polygon-name">${classItem.uiLabel} ${polygon.id}</span>
      <span class="polygon-points">${polygon.points.length}</span>
    `;
    selectButton.addEventListener("click", () => {
      state.selectedId = polygon.id;
      renderPolygonList();
      updateToolbar();
      draw();
    });

    const rectangleLabel = document.createElement("label");
    rectangleLabel.className = "rectangle-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(polygon.force2DRectangle);
    checkbox.addEventListener("change", () => {
      polygon.force2DRectangle = checkbox.checked;
      polygon.planeGeometry = planeGeometryFor(checkbox.checked);
      state.selectedId = polygon.id;
      setDirty(true);
      renderPolygonList();
      updateToolbar();
      draw();
    });
    const labelText = document.createElement("span");
    labelText.textContent = "2D Rectangle";
    rectangleLabel.append(checkbox, labelText);

    row.append(selectButton, rectangleLabel);
    polygonList.appendChild(row);
  }
}

function updateToolbar() {
  buttons.close.disabled = state.currentPoints.length < 3;
  buttons.cancel.disabled = state.currentPoints.length === 0;
  buttons.delete.disabled = !state.selectedId;
  buttons.defaultBlue.classList.toggle(
    "active-toggle",
    state.annotation.defaultUnannotatedClassId === "walkable_ground"
  );
  zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function setDirty(value) {
  state.dirty = value;
  if (value) {
    saveStatus.textContent = "Unsaved changes";
  }
}

function fitToStage() {
  if (!state.imageReady) return;
  const availableWidth = Math.max(320, stageShell.clientWidth - 36);
  const availableHeight = Math.max(240, stageShell.clientHeight - 36);
  state.fitZoom = Math.min(
    availableWidth / state.image.naturalWidth,
    availableHeight / state.image.naturalHeight,
    1
  );
  state.zoom = state.fitZoom;
  applyCanvasZoom();
}

function applyCanvasZoom() {
  if (!state.imageReady) return;
  canvas.style.width = `${state.image.naturalWidth * state.zoom}px`;
  canvas.style.height = `${state.image.naturalHeight * state.zoom}px`;
  updateToolbar();
}

function imagePointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  return [
    Math.max(0, Math.min(canvas.width, Number(x.toFixed(2)))),
    Math.max(0, Math.min(canvas.height, Number(y.toFixed(2)))),
  ];
}

function rgba(hex, alpha) {
  const clean = hex.replace("#", "");
  const value = parseInt(clean, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function drawPolygon(points, color, selected = false, force2DRectangle = false) {
  if (points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
  if (points.length >= 3) ctx.closePath();
  ctx.fillStyle = rgba(color, selected ? 0.32 : 0.22);
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 8 : 5;
  ctx.lineJoin = "round";
  if (points.length >= 3) ctx.fill();
  const previousDash = ctx.getLineDash();
  if (force2DRectangle) ctx.setLineDash([18, 10]);
  ctx.stroke();
  ctx.setLineDash(previousDash);

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], selected ? 9 : 6, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = selected ? 4 : 3;
    ctx.strokeStyle = color;
    ctx.stroke();
  }
}

function drawCurrentPolygon() {
  const classItem = classById(state.activeClassId);
  if (!classItem || state.currentPoints.length === 0) return;
  ctx.save();
  ctx.setLineDash([18, 12]);
  drawPolygon(state.currentPoints, classItem.color, true);
  ctx.restore();
}

function draw() {
  if (!state.imageReady) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0);
  if (state.annotation.defaultUnannotatedClassId === "walkable_ground") {
    const classItem = classById("walkable_ground");
    ctx.fillStyle = rgba(classItem.color, 0.14);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  for (const polygon of state.annotation.polygons) {
    drawPolygon(
      polygon.points,
      polygon.color,
      polygon.id === state.selectedId,
      polygon.force2DRectangle
    );
  }
  drawCurrentPolygon();
}

function addPoint(point) {
  state.currentPoints.push(point);
  setDirty(true);
  updateToolbar();
  draw();
}

function closePolygon() {
  if (state.currentPoints.length < 3) return;
  const classItem = classById(state.activeClassId);
  const nextIndex = state.annotation.polygons.length + 1;
  const polygon = {
    id: `${state.activeClassId}-${Date.now().toString(36)}-${nextIndex}`,
    classId: state.activeClassId,
    label: classItem.label,
    color: classItem.color,
    points: state.currentPoints.map((point) => [point[0], point[1]]),
    closed: true,
    force2DRectangle: false,
    planeGeometry: planeGeometryFor(false),
  };
  state.annotation.polygons.push(polygon);
  state.currentPoints = [];
  state.selectedId = polygon.id;
  setDirty(true);
  renderClassButtons();
  renderPolygonList();
  updateToolbar();
  draw();
}

function undoPointOrPolygon() {
  if (state.currentPoints.length) {
    state.currentPoints.pop();
  } else if (state.annotation.polygons.length) {
    const removed = state.annotation.polygons.pop();
    if (state.selectedId === removed.id) state.selectedId = null;
  }
  setDirty(true);
  renderClassButtons();
  renderPolygonList();
  updateToolbar();
  draw();
}

function cancelCurrent() {
  state.currentPoints = [];
  updateToolbar();
  draw();
}

function deleteSelected() {
  if (!state.selectedId) return;
  state.annotation.polygons = state.annotation.polygons.filter(
    (polygon) => polygon.id !== state.selectedId
  );
  state.selectedId = null;
  setDirty(true);
  renderClassButtons();
  renderPolygonList();
  updateToolbar();
  draw();
}

function clearActiveLayer() {
  const removed = state.annotation.polygons.some(
    (polygon) => polygon.classId === state.activeClassId
  );
  state.annotation.polygons = state.annotation.polygons.filter(
    (polygon) => polygon.classId !== state.activeClassId
  );
  if (removed) setDirty(true);
  state.selectedId = null;
  renderClassButtons();
  renderPolygonList();
  updateToolbar();
  draw();
}

function markUnannotatedAsBlue() {
  state.annotation.defaultUnannotatedClassId = "walkable_ground";
  setDirty(true);
  updateToolbar();
  draw();
}

async function saveAnnotation() {
  const payload = {
    ...state.annotation,
    groupId: state.activeGroupId,
    overlapPriorityClassIds: DEFAULT_OVERLAP_PRIORITY,
    polygons: state.annotation.polygons,
  };
  buttons.save.disabled = true;
  saveStatus.textContent = "Saving";
  try {
    const response = await fetch(groupApiUrl("/api/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "save failed");
    state.dirty = false;
    saveStatus.textContent = `Saved ${result.updatedAt}`;
  } catch (error) {
    saveStatus.textContent = `Save failed: ${error.message}`;
  } finally {
    buttons.save.disabled = false;
  }
}

canvas.addEventListener("click", (event) => {
  if (!state.imageReady) return;
  addPoint(imagePointFromEvent(event));
});

groupSelect.addEventListener("change", () => {
  const nextGroupId = groupSelect.value;
  if (state.dirty && !window.confirm("Discard unsaved changes and switch group?")) {
    groupSelect.value = state.activeGroupId;
    return;
  }
  state.dirty = false;
  saveStatus.textContent = "Loading";
  loadConfig(nextGroupId).catch((error) => {
    saveStatus.textContent = `Load failed: ${error.message}`;
  });
});

buttons.undo.addEventListener("click", undoPointOrPolygon);
buttons.close.addEventListener("click", closePolygon);
buttons.cancel.addEventListener("click", cancelCurrent);
buttons.delete.addEventListener("click", deleteSelected);
buttons.clearLayer.addEventListener("click", clearActiveLayer);
buttons.defaultBlue.addEventListener("click", markUnannotatedAsBlue);
buttons.save.addEventListener("click", saveAnnotation);
buttons.zoomOut.addEventListener("click", () => {
  state.zoom = Math.max(state.fitZoom * 0.25, state.zoom / 1.25);
  applyCanvasZoom();
});
buttons.zoomIn.addEventListener("click", () => {
  state.zoom = Math.min(3, state.zoom * 1.25);
  applyCanvasZoom();
});
buttons.fit.addEventListener("click", fitToStage);

window.addEventListener("resize", () => {
  if (Math.abs(state.zoom - state.fitZoom) < 0.001) fitToStage();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    closePolygon();
  } else if (event.key === "Escape") {
    cancelCurrent();
  } else if (event.key === "Backspace" || event.key === "Delete") {
    if (state.currentPoints.length) {
      undoPointOrPolygon();
    } else if (state.selectedId) {
      deleteSelected();
    }
  }
});

window.addEventListener("beforeunload", (event) => {
  if (!state.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

loadConfig().catch((error) => {
  saveStatus.textContent = `Load failed: ${error.message}`;
});
