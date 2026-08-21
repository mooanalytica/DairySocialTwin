const state = {
  config: null,
  groups: [],
  activeGroupId: new URLSearchParams(window.location.search).get("group"),
  annotation: null,
  image: new Image(),
  imageReady: false,
  currentPoints: [],
  selectedId: null,
  zoom: 1,
  fitZoom: 1,
  dirty: false,
  formDirty: false,
  isLoading: false,
  isSaving: false,
  switchInProgress: false,
  loadGeneration: 0,
  lastSaveError: "",
};

const DEFAULT_COLORS = ["#0b7285", "#5f3dc4", "#c92a2a", "#2f9e44", "#e67700", "#1864ab"];

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const stageShell = document.getElementById("stageShell");
const groupSelect = document.getElementById("groupSelect");
const frameMeta = document.getElementById("frameMeta");
const missingFrame = document.getElementById("missingFrame");
const zoneList = document.getElementById("zoneList");
const saveStatus = document.getElementById("saveStatus");
const zoomLabel = document.getElementById("zoomLabel");
const switchDialog = document.getElementById("switchDialog");
const switchTargetLabel = document.getElementById("switchTargetLabel");
const switchDialogStatus = document.getElementById("switchDialogStatus");

const fields = {
  zoneType: document.getElementById("zoneTypeInput"),
  label: document.getElementById("zoneLabelInput"),
  color: document.getElementById("zoneColorInput"),
  zoneId: document.getElementById("zoneIdPreview"),
};

const buttons = {
  newZone: document.getElementById("newZoneBtn"),
  applyZone: document.getElementById("applyZoneBtn"),
  undo: document.getElementById("undoPointBtn"),
  close: document.getElementById("closePolygonBtn"),
  cancel: document.getElementById("cancelPolygonBtn"),
  delete: document.getElementById("deleteZoneBtn"),
  clearAll: document.getElementById("clearAllBtn"),
  zoomOut: document.getElementById("zoomOutBtn"),
  fit: document.getElementById("fitBtn"),
  zoomIn: document.getElementById("zoomInBtn"),
  save: document.getElementById("saveBtn"),
  switchCancel: document.getElementById("switchCancelBtn"),
  switchDiscard: document.getElementById("switchDiscardBtn"),
  switchSave: document.getElementById("switchSaveBtn"),
};

let resolveSwitchDialog = null;

function slugify(value, fallback = "zone") {
  let slug = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!slug) slug = fallback;
  if (!/^[a-z]/.test(slug)) slug = `${fallback}_${slug}`;
  return slug;
}

function groupApiUrl(path, groupId = state.activeGroupId) {
  if (!groupId) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}group=${encodeURIComponent(groupId)}`;
}

function zoneById(zoneId) {
  return state.annotation?.zones?.find((zone) => zone.zone_id === zoneId);
}

function nextColor() {
  const colors = state.config?.defaultColors || DEFAULT_COLORS;
  return colors[state.annotation?.zones?.length % colors.length] || DEFAULT_COLORS[0];
}

function normalizeZone(zone, index, colors = DEFAULT_COLORS) {
  const zoneType = slugify(zone.zone_type || zone.type || zone.label, "zone");
  return {
    zone_id: slugify(zone.zone_id || `${zoneType}_${index + 1}`, zoneType),
    zone_type: zoneType,
    label: String(zone.label || zoneType),
    color: /^#[0-9a-fA-F]{6}$/.test(zone.color || "")
      ? zone.color.toLowerCase()
      : colors[index % colors.length] || DEFAULT_COLORS[0],
    polygon: Array.isArray(zone.polygon) ? zone.polygon : zone.points || [],
  };
}

function activeFormValues() {
  const zoneType = slugify(fields.zoneType.value, "zone");
  return {
    zone_type: zoneType,
    label: fields.label.value.trim() || zoneType,
    color: fields.color.value || nextColor(),
  };
}

function refreshZoneIdPreview() {
  const values = activeFormValues();
  const selected = zoneById(state.selectedId);
  fields.zoneId.textContent = selected
    ? selected.zone_id
    : `${values.zone_type}_${Date.now().toString(36)}`;
}

function setFormFromZone(zone) {
  fields.zoneType.value = zone.zone_type;
  fields.label.value = zone.label;
  fields.color.value = zone.color;
  state.formDirty = false;
  refreshZoneIdPreview();
}

function resetForm() {
  commitSelectedFormEdits();
  const color = nextColor();
  fields.zoneType.value = "resource_area";
  fields.label.value = "";
  fields.color.value = color;
  state.selectedId = null;
  state.formDirty = false;
  refreshZoneIdPreview();
  renderZoneList();
  updateToolbar();
  draw();
}

function renderGroupSelect() {
  groupSelect.replaceChildren();
  for (const group of state.groups) {
    const option = document.createElement("option");
    option.value = group.id;
    const status = group.zonesExists ? `${group.zoneCount || 0} zones` : group.frameExists ? "frame" : "missing";
    option.textContent = `${group.label || group.id} (${status})`;
    groupSelect.appendChild(option);
  }
  groupSelect.value = state.activeGroupId || "";
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("reference frame failed to load"));
    image.src = url;
  });
}

async function loadConfig(groupId = state.activeGroupId) {
  const generation = ++state.loadGeneration;
  state.isLoading = true;
  updateToolbar();

  try {
    const response = await fetch(groupApiUrl("/api/config", groupId));
    if (!response.ok) throw new Error(`config failed: ${response.status}`);
    const nextConfig = await response.json();
    if (generation !== state.loadGeneration) return false;

    const nextAnnotation = nextConfig.annotation || { zones: [] };
    if (!Array.isArray(nextAnnotation.zones)) nextAnnotation.zones = [];
    const colors = nextConfig.defaultColors || DEFAULT_COLORS;
    nextAnnotation.zones = nextAnnotation.zones.map((zone, index) => normalizeZone(zone, index, colors));

    let nextImage = null;
    if (nextConfig.frame?.exists) {
      nextImage = await loadImage(nextConfig.frame.url);
      if (generation !== state.loadGeneration) return false;
    }

    state.config = nextConfig;
    state.groups = nextConfig.groups || [];
    state.activeGroupId = nextConfig.activeGroupId;
    state.annotation = nextAnnotation;
    state.image = nextImage || new Image();
    state.imageReady = Boolean(nextImage);
    state.currentPoints = [];
    state.selectedId = null;
    state.formDirty = false;
    state.dirty = false;

    missingFrame.hidden = Boolean(nextConfig.frame?.exists);
    renderGroupSelect();
    resetForm();
    renderZoneList();

    const meta = nextConfig.frame?.metadata || {};
    const groupLabel = nextConfig.activeGroup?.label || state.activeGroupId;
    const frameLabel = meta.sourceVideoName ? `${meta.sourceVideoName} @ ${meta.timestamp}` : "reference frame";
    frameMeta.textContent = `${groupLabel} - ${frameLabel}`;

    if (nextImage) {
      canvas.width = nextImage.naturalWidth;
      canvas.height = nextImage.naturalHeight;
      fitToStage();
      draw();
      saveStatus.textContent = `Loaded ${nextAnnotation.zones.length} zones`;
    } else {
      if (canvas.width && canvas.height) ctx.clearRect(0, 0, canvas.width, canvas.height);
      saveStatus.textContent = `Missing frame for ${groupLabel}`;
    }
    history.replaceState(null, "", `?group=${encodeURIComponent(state.activeGroupId)}`);
    return true;
  } finally {
    if (generation === state.loadGeneration) {
      state.isLoading = false;
      updateToolbar();
    }
  }
}

function renderZoneList() {
  zoneList.replaceChildren();
  for (const zone of state.annotation?.zones || []) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `zone-row ${state.selectedId === zone.zone_id ? "active" : ""}`;
    row.disabled = state.isLoading || state.isSaving;
    row.addEventListener("click", () => {
      if (state.isLoading || state.isSaving) return;
      commitSelectedFormEdits();
      state.selectedId = zone.zone_id;
      setFormFromZone(zone);
      renderZoneList();
      updateToolbar();
      draw();
    });

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = zone.color;

    const text = document.createElement("span");
    text.className = "zone-text";
    const label = document.createElement("span");
    label.className = "zone-label";
    label.textContent = zone.label;
    const meta = document.createElement("span");
    meta.className = "zone-meta";
    meta.textContent = `${zone.zone_type} - ${zone.polygon.length} points`;
    text.append(label, meta);

    row.append(swatch, text);
    zoneList.appendChild(row);
  }
}

function updateToolbar() {
  const busy = state.isLoading || state.isSaving;
  const zones = state.annotation?.zones || [];
  fields.zoneType.disabled = busy;
  fields.label.disabled = busy;
  fields.color.disabled = busy;
  for (const row of zoneList.querySelectorAll(".zone-row")) row.disabled = busy;
  buttons.close.disabled = busy || state.currentPoints.length < 3;
  buttons.cancel.disabled = busy || state.currentPoints.length === 0;
  buttons.delete.disabled = busy || !state.selectedId;
  buttons.applyZone.disabled = busy || !state.selectedId;
  buttons.clearAll.disabled = busy || zones.length === 0;
  buttons.save.disabled = busy || !state.annotation || !state.config?.frame?.exists;
  buttons.newZone.disabled = busy;
  buttons.undo.disabled = busy;
  groupSelect.disabled = busy || state.switchInProgress;
  canvas.classList.toggle("busy", busy);
  zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  refreshZoneIdPreview();
}

function setDirty(value) {
  state.dirty = value;
  if (value) saveStatus.textContent = "Unsaved changes";
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

function centroid(points) {
  const total = points.reduce(
    (acc, point) => {
      acc.x += point[0];
      acc.y += point[1];
      return acc;
    },
    { x: 0, y: 0 }
  );
  return [total.x / points.length, total.y / points.length];
}

function drawPolygon(points, color, selected = false, label = "") {
  if (points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
  if (points.length >= 3) ctx.closePath();
  ctx.fillStyle = rgba(color, selected ? 0.34 : 0.22);
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 8 : 5;
  ctx.lineJoin = "round";
  if (points.length >= 3) ctx.fill();
  ctx.stroke();

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], selected ? 9 : 6, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = selected ? 4 : 3;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  if (label && points.length >= 3) {
    const [x, y] = centroid(points);
    ctx.font = "26px Segoe UI, Arial, sans-serif";
    const padding = 8;
    const metrics = ctx.measureText(label);
    ctx.fillStyle = "rgba(255, 255, 255, 0.86)";
    ctx.fillRect(x - metrics.width / 2 - padding, y - 23, metrics.width + padding * 2, 34);
    ctx.fillStyle = "#1d252c";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, y - 6);
  }
}

function drawCurrentPolygon() {
  if (state.currentPoints.length === 0) return;
  const values = activeFormValues();
  ctx.save();
  ctx.setLineDash([18, 12]);
  drawPolygon(state.currentPoints, values.color, true, values.label);
  ctx.restore();
}

function draw() {
  if (!state.imageReady) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0);
  for (const zone of state.annotation.zones) {
    drawPolygon(zone.polygon, zone.color, zone.zone_id === state.selectedId, zone.label);
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
  commitSelectedFormEdits();
  const values = activeFormValues();
  const nextIndex = state.annotation.zones.length + 1;
  const zone = {
    zone_id: `${values.zone_type}_${Date.now().toString(36)}_${nextIndex}`,
    zone_type: values.zone_type,
    label: values.label,
    color: values.color,
    polygon: state.currentPoints.map((point) => [point[0], point[1]]),
  };
  state.annotation.zones.push(zone);
  state.currentPoints = [];
  state.selectedId = zone.zone_id;
  setFormFromZone(zone);
  setDirty(true);
  renderZoneList();
  updateToolbar();
  draw();
}

function finalizePendingPolygonForSave() {
  if (state.currentPoints.length === 0) return true;
  if (state.currentPoints.length < 3) {
    saveStatus.textContent = "Save blocked: finish the zone with at least 3 points or cancel it";
    return false;
  }
  closePolygon();
  return true;
}

function undoPointOrZone() {
  if (state.currentPoints.length) {
    state.currentPoints.pop();
  } else if (state.annotation.zones.length) {
    const removed = state.annotation.zones.pop();
    if (state.selectedId === removed.zone_id) state.selectedId = null;
  }
  setDirty(true);
  renderZoneList();
  updateToolbar();
  draw();
}

function cancelCurrent() {
  state.currentPoints = [];
  updateToolbar();
  draw();
}

function commitSelectedFormEdits() {
  const selected = zoneById(state.selectedId);
  if (!selected || !state.formDirty) return false;
  const values = activeFormValues();
  selected.zone_type = values.zone_type;
  selected.label = values.label;
  selected.color = values.color;
  selected.zone_id = slugify(selected.zone_id || `${values.zone_type}_1`, values.zone_type);
  state.formDirty = false;
  setDirty(true);
  renderZoneList();
  updateToolbar();
  draw();
  return true;
}

function applyFormToSelected() {
  if (!zoneById(state.selectedId)) return;
  state.formDirty = true;
  commitSelectedFormEdits();
}

function deleteSelected() {
  if (!state.selectedId) return;
  state.annotation.zones = state.annotation.zones.filter((zone) => zone.zone_id !== state.selectedId);
  state.selectedId = null;
  state.formDirty = false;
  setDirty(true);
  renderZoneList();
  updateToolbar();
  draw();
}

function clearAllZones() {
  if (!state.annotation.zones.length) return;
  if (!window.confirm("Clear all zones for this group?")) return;
  state.annotation.zones = [];
  state.selectedId = null;
  state.formDirty = false;
  state.currentPoints = [];
  setDirty(true);
  renderZoneList();
  updateToolbar();
  draw();
}

async function saveAnnotation() {
  if (state.isSaving || state.isLoading || !state.annotation) return false;
  commitSelectedFormEdits();
  if (!finalizePendingPolygonForSave()) {
    state.lastSaveError = saveStatus.textContent;
    return false;
  }
  const payload = {
    ...state.annotation,
    groupId: state.activeGroupId,
    zones: state.annotation.zones,
  };
  state.isSaving = true;
  state.lastSaveError = "";
  updateToolbar();
  saveStatus.textContent = "Saving";
  try {
    const response = await fetch(groupApiUrl("/api/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let result;
    try {
      result = await response.json();
    } catch (_error) {
      throw new Error(`save failed: HTTP ${response.status}`);
    }
    if (!response.ok || !result.ok) throw new Error(result.error || `save failed: HTTP ${response.status}`);
    state.dirty = false;
    state.formDirty = false;
    const activeGroup = state.groups.find((group) => group.id === state.activeGroupId);
    if (activeGroup) {
      activeGroup.zonesExists = true;
      activeGroup.zoneCount = state.annotation.zones.length;
      renderGroupSelect();
    }
    saveStatus.textContent = `Saved ${result.updatedAt} -> ${result.zonesPath || "sna_zones.json"}`;
    return true;
  } catch (error) {
    state.lastSaveError = error instanceof Error ? error.message : String(error);
    saveStatus.textContent = `Save failed: ${state.lastSaveError}`;
    return false;
  } finally {
    state.isSaving = false;
    updateToolbar();
  }
}

function groupLabelFor(groupId) {
  return state.groups.find((group) => group.id === groupId)?.label || groupId;
}

function setSwitchDialogBusy(busy) {
  buttons.switchSave.disabled = busy;
  buttons.switchDiscard.disabled = busy;
  buttons.switchCancel.disabled = busy;
}

function finishSwitchDialog(action) {
  if (!resolveSwitchDialog) return;
  const resolve = resolveSwitchDialog;
  resolveSwitchDialog = null;
  if (switchDialog.open) switchDialog.close();
  resolve(action);
}

function promptSwitchAction(nextGroupId) {
  switchTargetLabel.textContent = groupLabelFor(nextGroupId);
  switchDialogStatus.textContent = "";
  setSwitchDialogBusy(false);
  return new Promise((resolve) => {
    resolveSwitchDialog = resolve;
    switchDialog.showModal();
    buttons.switchSave.focus();
  });
}

async function performGroupSwitch(nextGroupId) {
  const previousGroupId = state.activeGroupId;
  saveStatus.textContent = `Loading ${groupLabelFor(nextGroupId)}`;
  try {
    const loaded = await loadConfig(nextGroupId);
    if (!loaded) {
      groupSelect.value = previousGroupId;
      saveStatus.textContent = "Switch cancelled";
    }
    return loaded;
  } catch (error) {
    groupSelect.value = previousGroupId;
    const message = error instanceof Error ? error.message : String(error);
    saveStatus.textContent = `Load failed: ${message}`;
    return false;
  }
}

async function requestGroupSwitch(nextGroupId) {
  if (!nextGroupId || nextGroupId === state.activeGroupId || state.switchInProgress) {
    groupSelect.value = state.activeGroupId || "";
    return false;
  }

  state.switchInProgress = true;
  groupSelect.value = state.activeGroupId || "";
  updateToolbar();
  try {
    if (state.dirty) {
      const action = await promptSwitchAction(nextGroupId);
      if (action === "cancel") return false;
    }
    return await performGroupSwitch(nextGroupId);
  } finally {
    state.switchInProgress = false;
    groupSelect.value = state.activeGroupId || "";
    updateToolbar();
  }
}

canvas.addEventListener("click", (event) => {
  if (!state.imageReady || state.isLoading || state.isSaving) return;
  addPoint(imagePointFromEvent(event));
});

for (const field of [fields.zoneType, fields.label, fields.color]) {
  field.addEventListener("input", () => {
    if (state.isLoading || state.isSaving) return;
    if (state.selectedId) {
      state.formDirty = true;
      setDirty(true);
    } else if (state.currentPoints.length) {
      setDirty(true);
    }
    refreshZoneIdPreview();
    draw();
  });
}

groupSelect.addEventListener("change", async () => {
  const nextGroupId = groupSelect.value;
  await requestGroupSwitch(nextGroupId);
});

buttons.switchCancel.addEventListener("click", () => finishSwitchDialog("cancel"));
buttons.switchDiscard.addEventListener("click", () => finishSwitchDialog("discard"));
buttons.switchSave.addEventListener("click", async () => {
  setSwitchDialogBusy(true);
  switchDialogStatus.textContent = "Saving current group";
  const saved = await saveAnnotation();
  if (saved) {
    finishSwitchDialog("save");
    return;
  }
  switchDialogStatus.textContent = `Save failed: ${state.lastSaveError || "unknown error"}`;
  setSwitchDialogBusy(false);
  buttons.switchSave.focus();
});

switchDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  if (!state.isSaving) finishSwitchDialog("cancel");
});

buttons.newZone.addEventListener("click", resetForm);
buttons.applyZone.addEventListener("click", applyFormToSelected);
buttons.undo.addEventListener("click", undoPointOrZone);
buttons.close.addEventListener("click", closePolygon);
buttons.cancel.addEventListener("click", cancelCurrent);
buttons.delete.addEventListener("click", deleteSelected);
buttons.clearAll.addEventListener("click", clearAllZones);
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

function isInteractiveTarget(target) {
  return (
    target instanceof HTMLElement &&
    (target.matches("input, textarea, select, button") || target.isContentEditable)
  );
}

window.addEventListener("keydown", (event) => {
  if (switchDialog.open || state.isLoading || state.isSaving || isInteractiveTarget(event.target)) return;
  if (event.key === "Enter" && state.currentPoints.length >= 3) {
    event.preventDefault();
    closePolygon();
  } else if (event.key === "Escape" && state.currentPoints.length) {
    event.preventDefault();
    cancelCurrent();
  } else if (event.key === "Backspace" || event.key === "Delete") {
    if (state.currentPoints.length) {
      event.preventDefault();
      undoPointOrZone();
    } else if (state.selectedId) {
      event.preventDefault();
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
