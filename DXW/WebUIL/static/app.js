const state = {
  data: null,
  samples: [],
  sampleId: "",
  loadedSampleId: "",
  rotationSettings: { map: {}, plan: {}, ui: {} },
  loadVersion: 0,
  loading: false,
  loadJobId: "",
  frameMap: new Map(),
  tracks: new Map(),
  playing: false,
  lastTick: 0,
  rafId: 0,
  zoom: 1,
  zoomRange: {
    min: 0.25,
    max: 1.6,
  },
  uiRotation: {
    quarterTurns: 0,
    saving: false,
  },
  mapping: {
    semanticWarp: false,
  },
  cattleScale: 3,
  video: {
    enabled: false,
    splitRatio: 0.44,
    fitPending: false,
    ready: false,
    lastFrame: null,
  },
  filters: {
    farmId: "1",
    cameraId: "Gopro1",
  },
  rotation: {
    quarterTurns: 0,
    saving: false,
  },
  planRotation: {
    quarterTurns: 0,
    saving: false,
  },
  stage2: {
    frame: null,
    interactions: [],
    displayFrame: null,
    displayInteractions: [],
    stats: { drawn: 0, friendly: 0, unfriendly: 0, filteredByRed: 0, missingEndpoint: 0 },
    displayStats: { drawn: 0, friendly: 0, unfriendly: 0, filteredByRed: 0, missingEndpoint: 0 },
    pending: false,
    targetFrame: null,
    requestVersion: 0,
    error: "",
  },
  options: {
    base: true,
    reference: false,
    resourceFloorplan: false,
    trails: true,
    corrections: true,
    interactions: true,
  },
};

const canvas = document.getElementById("canvas");
const canvasFrame = document.getElementById("canvasFrame");
const ctx = canvas.getContext("2d");
const stage = document.getElementById("stage");
const splitWorkspace = document.getElementById("splitWorkspace");
const splitHandle = document.getElementById("splitHandle");
const videoPane = document.getElementById("videoPane");
const visualVideo = document.getElementById("visualVideo");
const farmSelect = document.getElementById("farmSelect");
const cameraSelect = document.getElementById("cameraSelect");
const sampleSelect = document.getElementById("sampleSelect");
const frameRange = document.getElementById("frameRange");
const playBtn = document.getElementById("playBtn");
const stepBackBtn = document.getElementById("stepBackBtn");
const stepForwardBtn = document.getElementById("stepForwardBtn");
const rotateLeftBtn = document.getElementById("rotateLeftBtn");
const rotateRightBtn = document.getElementById("rotateRightBtn");
const rotationValue = document.getElementById("rotationValue");
const zoomRange = document.getElementById("zoomRange");
const zoomLabel = document.getElementById("zoomLabel");
const videoToggle = document.getElementById("videoToggle");
const uiRotateLeftBtn = document.getElementById("uiRotateLeftBtn");
const uiRotateRightBtn = document.getElementById("uiRotateRightBtn");
const uiRotationValue = document.getElementById("uiRotationValue");
const semanticWarpToggle = document.getElementById("semanticWarpToggle");
const loadBtn = document.getElementById("loadBtn");
const planRotateLeftBtn = document.getElementById("planRotateLeftBtn");
const planRotateRightBtn = document.getElementById("planRotateRightBtn");
const planRotationValue = document.getElementById("planRotationValue");
const cattleScaleRange = document.getElementById("cattleScaleRange");
const cattleScaleValue = document.getElementById("cattleScaleValue");
const titleLine = document.getElementById("titleLine");
const sourceLine = document.getElementById("sourceLine");
const statusLine = document.getElementById("statusLine");
const frameValue = document.getElementById("frameValue");
const pointValue = document.getElementById("pointValue");
const trackValue = document.getElementById("trackValue");
const movedValue = document.getElementById("movedValue");
const interactionValue = document.getElementById("interactionValue");
const blockedValue = document.getElementById("blockedValue");
const trackLegend = document.getElementById("trackLegend");

const toggles = {
  base: document.getElementById("baseToggle"),
  reference: document.getElementById("referenceToggle"),
  resourceFloorplan: document.getElementById("resourceFloorplanToggle"),
  trails: document.getElementById("trailToggle"),
  corrections: document.getElementById("correctionToggle"),
  interactions: document.getElementById("interactionToggle"),
};

const mappingProfileTypeIds = {
  landscape_identity_to_floorplan: "H",
  portrait_ccw_to_floorplan: "S",
};

const trackColors = [
  "#0077ff",
  "#d7263d",
  "#2fb344",
  "#f08c00",
  "#7b2ff7",
  "#00a6a6",
  "#e64980",
  "#495057",
  "#20c997",
  "#fab005",
  "#5c7cfa",
  "#c92a2a",
];

const TRAIL_VISIBLE_SECONDS = 15;
const TRAIL_FADE_SECONDS = 5;
const INITIAL_ZOOM_MIN = 0.17;
const INITIAL_ZOOM_MAX = 1.6;
const SPLIT_MIN = 0.35;
const SPLIT_MAX = 0.85;
const SQUEEZE_MARGINS_PX = [8, 16, 32, 64];
const DEEP_LINK_PARAMETER_NAMES = ["farmID", "cameraID", "clipID", "segmentID", "frameID"];

function requiredIdentityString(value, fieldName, context) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${context} is missing required ${fieldName}`);
  }
  return value.trim();
}

function requiredIdentityInteger(value, fieldName, context) {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new Error(`${context} is missing required integer ${fieldName}`);
  }
  return value;
}

function visibleGlobalId(globalTrackId, context) {
  const zeroBasedId = requiredIdentityInteger(globalTrackId, "globalTrackId", context);
  if (zeroBasedId < 0) {
    throw new Error(`${context} has a negative globalTrackId`);
  }
  return String(zeroBasedId + 1);
}

function colorForGlobalUuid(globalTrackUuid) {
  const value = requiredIdentityString(globalTrackUuid, "globalTrackUuid", "global identity");
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return trackColors[(hash >>> 0) % trackColors.length];
}

function rgba(hex, alpha) {
  const clean = hex.replace("#", "");
  const value = parseInt(clean, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function pointsForFrame(frame) {
  return state.frameMap.get(frame) || [];
}

function normalizeQuarterTurns(value) {
  return ((Number(value) % 4) + 4) % 4;
}

function currentTypeId() {
  return mappingProfileTypeIds[state.data?.meta?.mappingProfile] || "H";
}

function compareNatural(a, b) {
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

function sampleById(sampleId) {
  return state.samples.find((sample) => sample.id === sampleId) || null;
}

function deepLinkRequestFromLocation() {
  const locationParams = new URLSearchParams(window.location.search);
  const requestParams = new URLSearchParams();
  for (const name of DEEP_LINK_PARAMETER_NAMES) {
    const values = locationParams.getAll(name);
    if (values.length !== 1 || !values[0].trim()) return null;
    requestParams.set(name, values[0].trim());
  }
  return requestParams;
}

async function resolveInitialDeepLink() {
  const requestParams = deepLinkRequestFromLocation();
  if (!requestParams) return null;
  const response = await fetch(`/api/deep-link/resolve?${requestParams.toString()}`);
  if (!response.ok) throw new Error(`deep-link resolve failed: ${response.status}`);
  const payload = await response.json();
  if (!payload.ok) throw new Error(payload.error || "deep-link resolve failed");
  if (!payload.valid) return null;
  if (
    typeof payload.sampleId !== "string" ||
    !payload.sampleId.trim() ||
    !Number.isSafeInteger(payload.frameId) ||
    payload.frameId < 0
  ) {
    throw new Error("deep-link resolver returned an invalid target");
  }
  return { sampleId: payload.sampleId, frameId: payload.frameId };
}

function farmIds() {
  return [...new Set(state.samples.map((sample) => String(sample.farmId || "")).filter(Boolean))].sort(compareNatural);
}

function cameraIdsForFarm(farmId) {
  return [...new Set(
    state.samples
      .filter((sample) => String(sample.farmId || "") === String(farmId))
      .map((sample) => String(sample.cameraId || ""))
      .filter(Boolean)
  )].sort(compareNatural);
}

function syncFiltersToSample(sample) {
  if (!sample) return;
  state.filters.farmId = String(sample.farmId || state.filters.farmId || "");
  state.filters.cameraId = String(sample.cameraId || state.filters.cameraId || "");
}

function samplesForCurrentFilter() {
  return state.samples.filter(
    (sample) =>
      String(sample.farmId || "") === String(state.filters.farmId) &&
      String(sample.cameraId || "") === String(state.filters.cameraId)
  );
}

function cameraNumber(cameraId) {
  const value = String(cameraId || "").trim();
  const match = value.match(/(\d+)$/);
  return match ? String(Number(match[1])) : value;
}

function rotationKey(typeId = currentTypeId()) {
  return `${state.filters.farmId}|${cameraNumber(state.filters.cameraId)}|${typeId}`;
}

function rotationBucket(scope) {
  return (state.rotationSettings && state.rotationSettings[scope]) || {};
}

function rotationDegrees() {
  return state.rotation.quarterTurns * 90;
}

function planRotationDegrees() {
  return state.planRotation.quarterTurns * 90;
}

function uiRotationDegrees() {
  return state.uiRotation.quarterTurns * 90;
}

function currentDisplayMode() {
  return state.mapping.semanticWarp ? "semantic" : "simple";
}

function updateSemanticWarpControl() {
  const available = Boolean(state.data?.meta?.semanticMappingAvailable);
  semanticWarpToggle.disabled = !available;
  if (!available) {
    state.mapping.semanticWarp = false;
    semanticWarpToggle.checked = false;
  } else {
    semanticWarpToggle.checked = state.mapping.semanticWarp;
  }
}

function updateResourceFloorplanControl() {
  const input = toggles.resourceFloorplan;
  if (!input) return;
  const available = Boolean(state.data?.resourceFloorplan?.available);
  input.disabled = !available;
  if (!available) {
    state.options.resourceFloorplan = false;
    input.checked = false;
  } else {
    input.checked = state.options.resourceFloorplan;
  }
}

function updatePlanRotationControls() {
  const hasData = Boolean(state.data);
  planRotateLeftBtn.disabled = !hasData || state.planRotation.saving;
  planRotateRightBtn.disabled = !hasData || state.planRotation.saving;
  planRotationValue.textContent = `${planRotationDegrees()} deg`;
}

function updateUiRotationControls() {
  const hasData = Boolean(state.data);
  uiRotateLeftBtn.disabled = !hasData || state.uiRotation.saving;
  uiRotateRightBtn.disabled = !hasData || state.uiRotation.saving;
  uiRotationValue.textContent = `${uiRotationDegrees()} deg`;
}

function applyUiDisplayRotation() {
  updateUiRotationControls();
  if (!state.data) return;
  const { width, height } = state.data.meta;
  const baseWidth = width * state.zoom;
  const baseHeight = height * state.zoom;
  const turns = state.uiRotation.quarterTurns;
  const rotated = turns % 2 === 1;
  const frameWidth = rotated ? baseHeight : baseWidth;
  const frameHeight = rotated ? baseWidth : baseHeight;
  canvasFrame.style.width = `${frameWidth}px`;
  canvasFrame.style.height = `${frameHeight}px`;
  canvas.style.left = `${(frameWidth - baseWidth) / 2}px`;
  canvas.style.top = `${(frameHeight - baseHeight) / 2}px`;
  canvas.style.transform = `rotate(${uiRotationDegrees()}deg)`;
}

function rotateUiDisplay(direction) {
  if (!state.data) return;
  state.uiRotation.quarterTurns = normalizeQuarterTurns(state.uiRotation.quarterTurns + direction);
  const key = rotationKey();
  state.rotationSettings.ui[key] = state.uiRotation.quarterTurns;
  applyUiDisplayRotation();
  draw();
  saveUiRotationSetting();
}

function resetUiDisplayRotation() {
  state.uiRotation.quarterTurns = 0;
  applyUiDisplayRotation();
}

function pointInCanvas(point) {
  if (!point || !state.data) return false;
  const [x, y] = point;
  return x >= 0 && x < state.data.meta.width && y >= 0 && y < state.data.meta.height;
}

function clampGeometryPoint(point) {
  const width = state.data.meta.width;
  const height = state.data.meta.height;
  return [
    Math.max(0, Math.min(width - 1, Number(point[0]))),
    Math.max(0, Math.min(height - 1, Number(point[1]))),
  ];
}

function distanceSq(a, b) {
  const dx = Number(a[0]) - Number(b[0]);
  const dy = Number(a[1]) - Number(b[1]);
  return dx * dx + dy * dy;
}

function pointOnSegment(point, a, b, tolerance = 1e-7) {
  const [px, py] = point;
  const [ax, ay] = a;
  const [bx, by] = b;
  const cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
  if (Math.abs(cross) > tolerance) return false;
  const dot = (px - ax) * (px - bx) + (py - ay) * (py - by);
  return dot <= tolerance;
}

function pointInPolygon(point, polygon) {
  if (!polygon || !polygon.length) return false;
  let inside = false;
  const [x, y] = point;
  for (let index = 0; index < polygon.length; index += 1) {
    const a = polygon[index];
    const b = polygon[(index + 1) % polygon.length];
    if (pointOnSegment(point, a, b)) return true;
    const [xi, yi] = a;
    const [xj, yj] = b;
    if ((yi > y) !== (yj > y)) {
      const xAtY = ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi;
      if (x < xAtY) inside = !inside;
    }
  }
  return inside;
}

function nearestPointOnSegment(point, a, b) {
  const [px, py] = point;
  const [ax, ay] = a;
  const [bx, by] = b;
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq <= 1e-12) return a;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSq));
  return [ax + t * dx, ay + t * dy];
}

function polygonArea(points) {
  let total = 0;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const nextPoint = points[(index + 1) % points.length];
    total += point[0] * nextPoint[1] - nextPoint[0] * point[1];
  }
  return total / 2;
}

function polygonCentroid(points) {
  const area = polygonArea(points);
  if (Math.abs(area) < 1e-9) {
    const total = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
    return [total[0] / points.length, total[1] / points.length];
  }

  let cx = 0;
  let cy = 0;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const nextPoint = points[(index + 1) % points.length];
    const cross = point[0] * nextPoint[1] - nextPoint[0] * point[1];
    cx += (point[0] + nextPoint[0]) * cross;
    cy += (point[1] + nextPoint[1]) * cross;
  }
  const factor = 1 / (6 * area);
  return [cx * factor, cy * factor];
}

function invalidPolygons() {
  if (state.data?.invalidPolygons?.length) return state.data.invalidPolygons;
  return (state.data?.basePolygons || []).filter((polygon) =>
    polygon.classId === "obstacle" || polygon.classId === "resource"
  );
}

function finalGeometryClass(point) {
  if (!pointInCanvas(point)) return "outside";
  for (const classId of ["resource", "obstacle"]) {
    for (const polygon of invalidPolygons()) {
      if (polygon.classId === classId && pointInPolygon(point, polygon.shapePoints)) return classId;
    }
  }
  return "open_2d_space";
}

function isAllowedGeometryPoint(point) {
  return finalGeometryClass(point) === "open_2d_space";
}

function nearestAllowedPoint(point) {
  const clamped = clampGeometryPoint(point);
  if (isAllowedGeometryPoint(clamped)) {
    return {
      point: clamped,
      reason: distanceSq(point, clamped) > 1e-6 ? "clamped" : "unchanged",
    };
  }

  const candidates = [];
  for (const polygon of invalidPolygons()) {
    const shapePoints = polygon.shapePoints || [];
    if (!pointInPolygon(clamped, shapePoints)) continue;

    let bestBoundary = null;
    let bestDistance = Infinity;
    for (let index = 0; index < shapePoints.length; index += 1) {
      const boundary = nearestPointOnSegment(clamped, shapePoints[index], shapePoints[(index + 1) % shapePoints.length]);
      const currentDistance = distanceSq(clamped, boundary);
      if (currentDistance < bestDistance) {
        bestDistance = currentDistance;
        bestBoundary = boundary;
      }
    }
    if (!bestBoundary) continue;

    const [cx, cy] = polygonCentroid(shapePoints);
    const vx = bestBoundary[0] - cx;
    const vy = bestBoundary[1] - cy;
    const length = Math.hypot(vx, vy) || 1;
    for (const margin of SQUEEZE_MARGINS_PX) {
      const candidate = clampGeometryPoint([
        bestBoundary[0] + (vx / length) * margin,
        bestBoundary[1] + (vy / length) * margin,
      ]);
      if (isAllowedGeometryPoint(candidate)) candidates.push([distanceSq(clamped, candidate), candidate]);
    }
  }

  if (candidates.length) {
    candidates.sort((a, b) => a[0] - b[0]);
    return { point: candidates[0][1], reason: "squeezed_to_boundary" };
  }

  const maxRadius = Math.ceil(Math.hypot(state.data.meta.width, state.data.meta.height)) + 16;
  for (let radius = 8; radius <= maxRadius; radius += 8) {
    for (let step = 0; step < 96; step += 1) {
      const angle = Math.PI * 2 * (step / 96);
      const candidate = clampGeometryPoint([
        clamped[0] + Math.cos(angle) * radius,
        clamped[1] + Math.sin(angle) * radius,
      ]);
      if (isAllowedGeometryPoint(candidate)) {
        return { point: candidate, reason: "squeezed_by_radial_search" };
      }
    }
  }

  return { point: clamped, reason: "no_allowed_point_found" };
}

function rotateCanvasPoint(point) {
  if (!state.data || !point) return null;
  const turns = state.rotation.quarterTurns;
  if (turns === 0) return pointInCanvas(point) ? point : null;
  const width = state.data.meta.width;
  const height = state.data.meta.height;
  const cx = width / 2;
  const cy = height / 2;
  const dx = point[0] - cx;
  const dy = point[1] - cy;
  let rotated;
  if (turns === 1) {
    rotated = [cx + dy, cy - dx];
  } else if (turns === 2) {
    rotated = [cx - dx, cy - dy];
  } else {
    rotated = [cx - dy, cy + dx];
  }
  return pointInCanvas(rotated) ? rotated : null;
}

function pointForCurrentMapping(point) {
  const semantic = state.mapping.semanticWarp && point.semanticPoint;
  const raw = semantic ? (point.semanticRaw || point.raw || point.point) : (point.raw || point.point);
  const source = semantic ? (point.semanticPoint || point.semanticWarped || raw) : (point.point || raw);
  const adjusted = nearestAllowedPoint(source);
  const adjustmentPx = Math.sqrt(distanceSq(raw, adjusted.point));
  if (!semantic) {
    return {
      ...point,
      raw,
      point: adjusted.point,
      adjustmentPx,
      adjustmentReason: adjusted.reason,
    };
  }

  return {
    ...point,
    raw,
    point: adjusted.point,
    adjustmentPx,
    adjustmentReason: `${point.semanticAdjustmentReason || "semantic"};${adjusted.reason}`,
  };
}

function displayPoint(point) {
  const selected = pointForCurrentMapping(point);
  const mapped = rotateCanvasPoint(selected.point);
  if (!mapped) return null;
  return { ...selected, point: mapped, raw: rotateCanvasPoint(selected.raw) };
}

function displayPointsForFrame(frame) {
  return pointsForFrame(frame).map(displayPoint).filter(Boolean);
}

function resetStage2State() {
  state.stage2.requestVersion += 1;
  state.stage2.frame = null;
  state.stage2.interactions = [];
  state.stage2.displayFrame = null;
  state.stage2.displayInteractions = [];
  state.stage2.stats = { drawn: 0, friendly: 0, unfriendly: 0, filteredByRed: 0, missingEndpoint: 0 };
  state.stage2.displayStats = { drawn: 0, friendly: 0, unfriendly: 0, filteredByRed: 0, missingEndpoint: 0 };
  state.stage2.pending = false;
  state.stage2.targetFrame = null;
  state.stage2.error = "";
}

function stopPlayback() {
  if (!state.playing) return;
  state.playing = false;
  playBtn.textContent = "Play";
  state.lastTick = 0;
  cancelTick();
}

function updateLoadControl() {
  loadBtn.disabled = state.loading || !state.sampleId;
  loadBtn.textContent = state.loading ? "..." : "Load";
}

function updateLoadedControls() {
  const hasData = Boolean(state.data);
  playBtn.disabled = !hasData;
  stepBackBtn.disabled = !hasData;
  stepForwardBtn.disabled = !hasData;
  frameRange.disabled = !hasData;
  zoomRange.disabled = !hasData;
  cattleScaleRange.disabled = !hasData;
  for (const input of Object.values(toggles)) input.disabled = !hasData;
  updateResourceFloorplanControl();
  updateLoadControl();
}

function progressText(progress) {
  const label = progress?.label || "Working";
  const current = progress?.current;
  const total = progress?.total;
  if (Number.isFinite(Number(current)) && Number.isFinite(Number(total)) && Number(total) > 0) {
    return `${label}: ${Number(current)}/${Number(total)}`;
  }
  if (Number.isFinite(Number(current))) {
    return `${label}: ${Number(current)}/...`;
  }
  return label;
}

function clearLoadedView(message = "Select a sample, then Load") {
  stopPlayback();
  playBtn.textContent = "Play";
  state.data = null;
  state.loadedSampleId = "";
  state.frameMap.clear();
  state.tracks.clear();
  resetStage2State();
  visualVideo.pause();
  clearVideoSource();
  state.video.enabled = false;
  state.video.ready = false;
  applySplitLayout();
  canvas.width = 0;
  canvas.height = 0;
  canvas.style.width = "0px";
  canvas.style.height = "0px";
  canvasFrame.style.width = "0px";
  canvasFrame.style.height = "0px";
  frameRange.min = "0";
  frameRange.max = "0";
  frameRange.value = "0";
  frameValue.textContent = "0";
  pointValue.textContent = "0";
  trackValue.textContent = "0";
  movedValue.textContent = "0";
  interactionValue.textContent = "0";
  blockedValue.textContent = "0";
  trackLegend.replaceChildren();
  sourceLine.textContent = sampleSelect.selectedOptions[0]?.textContent || "ready";
  statusLine.textContent = message;
  updateSemanticWarpControl();
  updateRotationControls();
  updatePlanRotationControls();
  updateUiRotationControls();
  updateVideoControls({ available: false });
  updateLoadedControls();
}

function currentVideoInfo() {
  return (state.data && state.data.meta && state.data.meta.sample && state.data.meta.sample.video) || {
    available: false,
  };
}

function applySplitLayout() {
  if (!state.video.enabled) {
    splitWorkspace.classList.remove("video-active");
    splitWorkspace.style.gridTemplateColumns = "";
    splitHandle.hidden = true;
    videoPane.hidden = true;
    return;
  }

  const top = Math.round(state.video.splitRatio * 1000);
  const bottom = Math.max(1, 1000 - top);
  splitWorkspace.classList.add("video-active");
  splitHandle.hidden = false;
  videoPane.hidden = false;
  splitWorkspace.style.gridTemplateColumns = `minmax(0, ${top}fr) 8px minmax(160px, ${bottom}fr)`;
}

function setZoom(value) {
  state.zoom = Math.max(state.zoomRange.min, Math.min(state.zoomRange.max, value));
  zoomRange.value = String(Math.round(state.zoom * 100));
  applyCanvasSize();
}

function fitFloorplanToStage() {
  if (!state.data) return;
  const zoom = defaultZoom();
  setZoomRangeFromDefault(zoom);
  setZoom(zoom);
  draw();
}

function scheduleFloorplanFit() {
  if (state.video.fitPending) return;
  state.video.fitPending = true;
  requestAnimationFrame(() => {
    state.video.fitPending = false;
    fitFloorplanToStage();
  });
}

function updateVideoControls(info = currentVideoInfo()) {
  const available = Boolean(info.available);
  videoToggle.disabled = !available;
}

function clearVideoSource() {
  visualVideo.removeAttribute("src");
  visualVideo.load();
}

function loadVideoSource(info) {
  if (!info.available) {
    clearVideoSource();
    return;
  }
  const absoluteUrl = new URL(info.selectedUrl, window.location.href).href;
  if (visualVideo.src !== absoluteUrl) {
    state.video.ready = false;
    visualVideo.src = info.selectedUrl;
    visualVideo.load();
  }
  visualVideo.loop = true;
}

function configureVideoForSample() {
  const info = currentVideoInfo();
  const available = Boolean(info.available);
  state.video.enabled = available;
  state.video.ready = false;
  state.video.lastFrame = null;
  videoToggle.checked = available;
  videoToggle.title = info.available
    ? `${info.selectedKind}: ${info.selectedName}`
    : "No visualization video";
  if (available) {
    loadVideoSource(info);
  } else {
    visualVideo.pause();
    clearVideoSource();
  }
  updateVideoControls(info);
  applySplitLayout();
}

function playVisualVideo() {
  const loadVersion = state.loadVersion;
  const source = visualVideo.currentSrc || visualVideo.src;
  visualVideo.play().catch(() => {
    if (
      loadVersion !== state.loadVersion ||
      source !== (visualVideo.currentSrc || visualVideo.src) ||
      !state.video.enabled ||
      !state.playing
    ) return;
    stopPlayback();
    statusLine.textContent = "Autoplay was blocked; press Play to continue";
  });
}

function setVideoEnabled(enabled) {
  const info = currentVideoInfo();
  const nextEnabled = Boolean(enabled && info.available);
  state.video.enabled = nextEnabled;
  videoToggle.checked = nextEnabled;
  if (nextEnabled) {
    loadVideoSource(info);
  } else {
    visualVideo.pause();
  }
  updateVideoControls(info);
  applySplitLayout();
  scheduleFloorplanFit();
  if (nextEnabled && state.video.ready) playVisualVideo();
}

function clampFrame(frame) {
  const minFrame = Number(frameRange.min);
  const maxFrame = Number(frameRange.max);
  return Math.max(minFrame, Math.min(maxFrame, Math.round(Number(frame))));
}

function timeForFrame(frame) {
  const fps = Number(state.data?.meta?.fps || currentVideoInfo().fps || 30);
  const minFrame = Number(frameRange.min);
  if (!Number.isFinite(fps) || fps <= 0) return 0;
  return Math.max(0, (clampFrame(frame) - minFrame) / fps);
}

function frameForVideoTime() {
  const fps = Number(state.data?.meta?.fps || currentVideoInfo().fps || 30);
  const minFrame = Number(frameRange.min);
  if (!Number.isFinite(fps) || fps <= 0) return minFrame;
  return clampFrame(minFrame + visualVideo.currentTime * fps);
}

function updateFloorplanFromVideo() {
  if (!state.data || !currentVideoInfo().available || !state.video.ready) return;
  const frame = frameForVideoTime();
  if (state.video.lastFrame === frame && Number(frameRange.value) === frame) return;
  state.video.lastFrame = frame;
  frameRange.value = String(frame);
  draw();
  if (state.options.interactions) requestStage2(frame);
}

function seekVideoToFrame(frame) {
  if (!currentVideoInfo().available || !state.video.ready) return false;
  const duration = visualVideo.duration;
  const targetTime = timeForFrame(frame);
  if (!Number.isFinite(duration) || duration <= 0) {
    visualVideo.currentTime = targetTime;
    return true;
  }
  visualVideo.currentTime = Math.min(targetTime, Math.max(0, duration - 0.001));
  return true;
}

function updateSplitFromPointer(clientX) {
  const rect = splitWorkspace.getBoundingClientRect();
  if (rect.width <= 240) return;
  const ratio = (clientX - rect.left) / rect.width;
  state.video.splitRatio = Math.max(SPLIT_MIN, Math.min(SPLIT_MAX, ratio));
  applySplitLayout();
  scheduleFloorplanFit();
}

function pointMapForFrame(frame) {
  const out = new Map();
  for (const point of displayPointsForFrame(frame)) out.set(Number(point.trackId), point.point);
  return out;
}

function buildIndexes() {
  state.frameMap.clear();
  state.tracks.clear();
  for (const item of state.data.frames) {
    state.frameMap.set(item.frame, item.points);
    for (const point of item.points) {
      if (!state.tracks.has(point.trackId)) state.tracks.set(point.trackId, []);
      state.tracks.get(point.trackId).push({ frame: item.frame, ...point });
    }
  }
}

function validateGlobalIdentityPayload(payload) {
  if (!payload || !payload.meta || !Array.isArray(payload.frames)) {
    throw new Error("sample payload is missing meta or frames");
  }
  if (!Array.isArray(payload.meta.globalIdentities)) {
    throw new Error("sample payload is missing required meta.globalIdentities");
  }
  const rowCount = requiredIdentityInteger(payload.meta.rowCount, "meta.rowCount", "sample payload");
  const displayDetectionRowCount = requiredIdentityInteger(
    payload.meta.displayDetectionRowCount,
    "meta.displayDetectionRowCount",
    "sample payload"
  );
  const reidInvalidRowCount = requiredIdentityInteger(
    payload.meta.reidInvalidRowCount,
    "meta.reidInvalidRowCount",
    "sample payload"
  );
  if (displayDetectionRowCount + reidInvalidRowCount !== rowCount) {
    throw new Error("sample payload re-ID validity counts do not add up to meta.rowCount");
  }

  const identitiesByUuid = new Map();
  const uuidsByDisplayId = new Map();
  const uuidsByGlobalTrackId = new Map();
  for (const [index, identity] of payload.meta.globalIdentities.entries()) {
    const context = `meta.globalIdentities[${index}]`;
    if (!identity || typeof identity !== "object") {
      throw new Error(`${context} must be an object`);
    }
    const globalTrackId = requiredIdentityInteger(identity.globalTrackId, "globalTrackId", context);
    const globalTrackUuid = requiredIdentityString(identity.globalTrackUuid, "globalTrackUuid", context);
    const displayGlobalId = requiredIdentityString(identity.displayGlobalId, "displayGlobalId", context);
    if (!Array.isArray(identity.localTrackIds) || !identity.localTrackIds.length) {
      throw new Error(`${context} is missing required localTrackIds`);
    }
    if (!Array.isArray(identity.idStatuses) || !identity.idStatuses.length) {
      throw new Error(`${context} is missing required idStatuses`);
    }
    const localTrackIds = new Set();
    for (const trackId of identity.localTrackIds) {
      localTrackIds.add(requiredIdentityInteger(trackId, "localTrackIds value", context));
    }
    const idStatuses = new Set(
      identity.idStatuses.map((status, statusIndex) =>
        requiredIdentityString(status, `idStatuses[${statusIndex}]`, context)
      )
    );
    if (identitiesByUuid.has(globalTrackUuid)) {
      throw new Error(`duplicate globalTrackUuid in meta.globalIdentities: ${globalTrackUuid}`);
    }
    const previousUuid = uuidsByDisplayId.get(displayGlobalId);
    if (previousUuid && previousUuid !== globalTrackUuid) {
      throw new Error(`displayGlobalId ${displayGlobalId} maps to multiple globalTrackUuid values`);
    }
    const previousUuidForTrackId = uuidsByGlobalTrackId.get(globalTrackId);
    if (previousUuidForTrackId && previousUuidForTrackId !== globalTrackUuid) {
      throw new Error(`globalTrackId ${globalTrackId} maps to multiple globalTrackUuid values`);
    }
    uuidsByDisplayId.set(displayGlobalId, globalTrackUuid);
    uuidsByGlobalTrackId.set(globalTrackId, globalTrackUuid);
    identitiesByUuid.set(globalTrackUuid, { globalTrackId, displayGlobalId, localTrackIds, idStatuses });
  }

  for (const [frameIndex, frame] of payload.frames.entries()) {
    if (!frame || !Array.isArray(frame.points)) {
      throw new Error(`frames[${frameIndex}] is missing points`);
    }
    for (const [pointIndex, point] of frame.points.entries()) {
      const context = `frames[${frameIndex}].points[${pointIndex}]`;
      if (!point || typeof point !== "object") {
        throw new Error(`${context} must be an object`);
      }
      const trackId = requiredIdentityInteger(point.trackId, "trackId", context);
      const globalTrackId = requiredIdentityInteger(point.globalTrackId, "globalTrackId", context);
      const globalTrackUuid = requiredIdentityString(point.globalTrackUuid, "globalTrackUuid", context);
      const displayGlobalId = requiredIdentityString(point.displayGlobalId, "displayGlobalId", context);
      const reidIdStatus = requiredIdentityString(point.reidIdStatus, "reidIdStatus", context);
      const identity = identitiesByUuid.get(globalTrackUuid);
      if (!identity) {
        throw new Error(`${context} references unknown globalTrackUuid ${globalTrackUuid}`);
      }
      if (identity.displayGlobalId !== displayGlobalId) {
        throw new Error(`${context} has inconsistent displayGlobalId for ${globalTrackUuid}`);
      }
      if (identity.globalTrackId !== globalTrackId) {
        throw new Error(`${context} has inconsistent globalTrackId for ${globalTrackUuid}`);
      }
      if (!identity.localTrackIds.has(trackId)) {
        throw new Error(`${context} local trackId is absent from meta.globalIdentities`);
      }
      if (!identity.idStatuses.has(reidIdStatus)) {
        throw new Error(`${context} reidIdStatus is absent from meta.globalIdentities`);
      }
    }
  }
}

function renderLegend() {
  trackLegend.replaceChildren();
  for (const identity of state.data.meta.globalIdentities) {
    const globalTrackUuid = requiredIdentityString(
      identity.globalTrackUuid,
      "globalTrackUuid",
      "meta.globalIdentities legend row"
    );
    const globalId = visibleGlobalId(identity.globalTrackId, "meta.globalIdentities legend row");
    const chip = document.createElement("div");
    chip.className = "track-chip";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = colorForGlobalUuid(globalTrackUuid);
    const label = document.createElement("span");
    label.textContent = globalId;
    chip.append(dot, label);
    trackLegend.appendChild(chip);
  }
}

function renderFarmCameraOptions() {
  const farms = farmIds();
  if (!farms.includes(state.filters.farmId)) {
    state.filters.farmId = farms[0] || "";
  }
  farmSelect.replaceChildren();
  for (const farmId of farms) {
    const option = document.createElement("option");
    option.value = farmId;
    option.textContent = farmId;
    farmSelect.appendChild(option);
  }
  farmSelect.value = state.filters.farmId;

  const cameras = cameraIdsForFarm(state.filters.farmId);
  if (!cameras.includes(state.filters.cameraId)) {
    state.filters.cameraId = cameras[0] || "";
  }
  cameraSelect.replaceChildren();
  for (const cameraId of cameras) {
    const option = document.createElement("option");
    option.value = cameraId;
    option.textContent = cameraId;
    cameraSelect.appendChild(option);
  }
  cameraSelect.value = state.filters.cameraId;
}

function renderSampleOptions() {
  const samples = samplesForCurrentFilter();
  if (!samples.some((sample) => sample.id === state.sampleId)) {
    state.sampleId = (samples[0] && samples[0].id) || "";
  }
  sampleSelect.replaceChildren();
  for (const sample of samples) {
    const option = document.createElement("option");
    option.value = sample.id;
    option.textContent = sample.label;
    sampleSelect.appendChild(option);
  }
  sampleSelect.value = state.sampleId;
}

function titleForMappingProfile(mappingProfile) {
  const typeId = mappingProfileTypeIds[mappingProfile] || "?";
  return `Dairy Floor Plan\n[Farm ${state.filters.farmId}, Camera ${state.filters.cameraId}, ${typeId}]`;
}

function updateRotationControls() {
  const hasData = Boolean(state.data);
  rotateLeftBtn.disabled = !hasData || state.rotation.saving;
  rotateRightBtn.disabled = !hasData || state.rotation.saving;
  rotationValue.textContent = `${rotationDegrees()} deg`;
}

function applyRotationForCurrentProfile() {
  const key = rotationKey();
  state.rotation.quarterTurns = normalizeQuarterTurns(rotationBucket("map")[key] || 0);
  updateRotationControls();
}

function applyPlanRotationForCurrentProfile() {
  const metaTurns = state.data?.meta?.planRotationQuarterTurns;
  const key = rotationKey();
  state.planRotation.quarterTurns = normalizeQuarterTurns(
    metaTurns !== undefined ? metaTurns : rotationBucket("plan")[key] || 0
  );
  updatePlanRotationControls();
}

function applyUiRotationForCurrentProfile() {
  const key = rotationKey();
  state.uiRotation.quarterTurns = normalizeQuarterTurns(rotationBucket("ui")[key] || 0);
  updateUiRotationControls();
}

function normalizeRotationSettings(settings) {
  const out = { map: {}, plan: {}, ui: {} };
  if (!settings || typeof settings !== "object") return out;
  if (settings.map || settings.plan || settings.ui) {
    for (const scope of ["map", "plan", "ui"]) {
      const scoped = settings[scope] || {};
      if (!scoped || typeof scoped !== "object") continue;
      for (const [key, value] of Object.entries(scoped)) {
        out[scope][key] = normalizeQuarterTurns(value);
      }
    }
    return out;
  }
  for (const [key, value] of Object.entries(settings)) {
    out.map[key] = normalizeQuarterTurns(value);
  }
  return out;
}

async function loadRotationSettings() {
  const response = await fetch("/api/rotation");
  if (!response.ok) throw new Error(`rotation settings request failed: ${response.status}`);
  const payload = await response.json();
  if (!payload.ok) throw new Error(payload.error || "rotation settings request failed");
  state.rotationSettings = normalizeRotationSettings(payload.settings);
}

async function saveRotationSetting() {
  const typeId = currentTypeId();
  const key = rotationKey(typeId);
  state.rotationSettings.map[key] = state.rotation.quarterTurns;
  state.rotation.saving = true;
  updateRotationControls();
  try {
    const response = await fetch("/api/rotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: "map",
        farmId: state.filters.farmId,
        cameraId: state.filters.cameraId,
        typeId,
        quarterTurns: state.rotation.quarterTurns,
      }),
    });
    if (!response.ok) throw new Error(`rotation save failed: ${response.status}`);
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "rotation save failed");
    state.rotationSettings = normalizeRotationSettings(payload.settings || state.rotationSettings);
    statusLine.textContent = `MAP rotation saved for Farm ${state.filters.farmId}, Camera ${state.filters.cameraId}, ${typeId}`;
  } catch (error) {
    statusLine.textContent = error.message;
  } finally {
    state.rotation.saving = false;
    updateRotationControls();
  }
}

async function saveUiRotationSetting() {
  const typeId = currentTypeId();
  const key = rotationKey(typeId);
  state.rotationSettings.ui[key] = state.uiRotation.quarterTurns;
  state.uiRotation.saving = true;
  updateUiRotationControls();
  try {
    const response = await fetch("/api/rotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: "ui",
        farmId: state.filters.farmId,
        cameraId: state.filters.cameraId,
        typeId,
        quarterTurns: state.uiRotation.quarterTurns,
      }),
    });
    if (!response.ok) throw new Error(`UI rotation save failed: ${response.status}`);
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "UI rotation save failed");
    state.rotationSettings = normalizeRotationSettings(payload.settings || state.rotationSettings);
    statusLine.textContent = `UI rotation saved for Farm ${state.filters.farmId}, Camera ${state.filters.cameraId}, ${typeId}`;
  } catch (error) {
    statusLine.textContent = error.message;
  } finally {
    state.uiRotation.saving = false;
    updateUiRotationControls();
  }
}

function rotateDisplay(direction) {
  if (!state.data) return;
  state.rotation.quarterTurns = normalizeQuarterTurns(state.rotation.quarterTurns + direction);
  const key = rotationKey();
  state.rotationSettings.map[key] = state.rotation.quarterTurns;
  updateRotationControls();
  draw();
  saveRotationSetting();
}

async function savePlanRotationSetting() {
  const typeId = currentTypeId();
  const key = rotationKey(typeId);
  state.rotationSettings.plan[key] = state.planRotation.quarterTurns;
  state.planRotation.saving = true;
  updatePlanRotationControls();
  try {
    const response = await fetch("/api/rotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: "plan",
        farmId: state.filters.farmId,
        cameraId: state.filters.cameraId,
        typeId,
        quarterTurns: state.planRotation.quarterTurns,
      }),
    });
    if (!response.ok) throw new Error(`plan rotation save failed: ${response.status}`);
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "plan rotation save failed");
    state.rotationSettings = normalizeRotationSettings(payload.settings || state.rotationSettings);
    statusLine.textContent = `Plan rotation saved for Farm ${state.filters.farmId}, Camera ${state.filters.cameraId}, ${typeId}`;
    await load(state.sampleId, { resetUiRotation: false });
  } catch (error) {
    statusLine.textContent = error.message;
  } finally {
    state.planRotation.saving = false;
    updatePlanRotationControls();
  }
}

function rotatePlan(direction) {
  if (!state.data) return;
  state.planRotation.quarterTurns = normalizeQuarterTurns(state.planRotation.quarterTurns + direction);
  const key = rotationKey();
  state.rotationSettings.plan[key] = state.planRotation.quarterTurns;
  updatePlanRotationControls();
  savePlanRotationSetting();
}

function applyCanvasSize() {
  if (!state.data) return;
  const { width, height } = state.data.meta;
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width * state.zoom}px`;
  canvas.style.height = `${height * state.zoom}px`;
  applyUiDisplayRotation();
  zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function fitZoom() {
  if (!state.data) return 1;
  const availableWidth = Math.max(320, stage.clientWidth - 36);
  const availableHeight = Math.max(240, stage.clientHeight - 36);
  return Math.min(availableWidth / state.data.meta.width, availableHeight / state.data.meta.height, 1);
}

function defaultZoom() {
  return Math.max(INITIAL_ZOOM_MIN, Math.min(INITIAL_ZOOM_MAX, fitZoom()));
}

function setZoomRangeFromDefault(zoom) {
  state.zoomRange.min = zoom / 3;
  state.zoomRange.max = zoom * 3;
  zoomRange.min = String(Math.round(state.zoomRange.min * 100));
  zoomRange.max = String(Math.round(state.zoomRange.max * 100));
}

function drawPolygon(points, color, alpha, selected, dashed) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
  ctx.closePath();
  ctx.fillStyle = rgba(color, alpha);
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 8 : 4;
  ctx.lineJoin = "round";
  if (dashed) ctx.setLineDash([18, 10]);
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBase() {
  for (const polygon of state.data.basePolygons) {
    const color = polygon.color || "#868e96";
    const alpha = polygon.classId === "resource" ? 0.42 : 0.36;
    drawPolygon(polygon.shapePoints, color, alpha, false, true);
  }
}

function drawReferenceAreas() {
  for (const polygon of state.data.referencePolygons) {
    const color = polygon.classId === "walkable_ground" ? "#1e63ff" : "#e03131";
    const alpha = polygon.classId === "walkable_ground" ? 0.12 : 0.16;
    drawPolygon(polygon.shapePoints, color, alpha, false, polygon.force2DRectangle);
  }
}

function drawResourceFloorplan() {
  const resourceFloorplan = state.data.resourceFloorplan;
  if (!resourceFloorplan?.available || !resourceFloorplan.zones?.length) return;

  ctx.save();
  for (const zone of resourceFloorplan.zones) {
    const color = zone.color || "#0b7285";
    drawPolygon(zone.points || [], color, 0.2, false, false);
  }

  const labelRotation = -uiRotationDegrees() * Math.PI / 180;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "700 42px Segoe UI, Arial, sans-serif";
  for (const zone of resourceFloorplan.zones) {
    const point = zone.centroid;
    const label = String(zone.label || zone.zoneType || zone.zoneId || "").trim();
    if (!point || !label) continue;
    ctx.save();
    ctx.translate(point[0], point[1]);
    ctx.rotate(labelRotation);
    ctx.lineWidth = 10;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.fillStyle = "#1f2933";
    ctx.strokeText(label, 0, 0);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

function lowerBoundFrame(points, frame) {
  let lo = 0;
  let hi = points.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (points[mid].frame < frame) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function drawTrails(frame) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const fps = state.data.meta.fps || 30;
  const trailFrames = Math.max(1, Math.round(fps * TRAIL_VISIBLE_SECONDS));
  const fadeFrames = Math.max(1, Math.round(fps * TRAIL_FADE_SECONDS));
  const startFrame = frame - trailFrames;
  for (const points of state.tracks.values()) {
    ctx.lineWidth = 5;
    const startIndex = lowerBoundFrame(points, startFrame);
    const endIndex = lowerBoundFrame(points, frame + 1);
    if (endIndex - startIndex < 2) continue;
    for (let index = startIndex + 1; index < endIndex; index += 1) {
      const previous = pointForCurrentMapping(points[index - 1]);
      const current = pointForCurrentMapping(points[index]);
      if (previous.globalTrackUuid !== current.globalTrackUuid) continue;
      const previousPoint = rotateCanvasPoint(previous.point);
      const currentPoint = rotateCanvasPoint(current.point);
      if (!previousPoint || !currentPoint) continue;
      const ageFrames = Math.max(0, frame - current.frame);
      const fade = Math.max(0, Math.min(1, (trailFrames - ageFrames) / fadeFrames));
      if (fade <= 0) continue;
      const color = colorForGlobalUuid(current.globalTrackUuid);
      ctx.beginPath();
      ctx.moveTo(previousPoint[0], previousPoint[1]);
      ctx.lineTo(currentPoint[0], currentPoint[1]);
      ctx.strokeStyle = rgba(color, 0.62 * fade);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawCorrections(points) {
  ctx.save();
  for (const point of points) {
    if (point.adjustmentPx <= 0.01 || !point.raw) continue;
    ctx.beginPath();
    ctx.moveTo(point.raw[0], point.raw[1]);
    ctx.lineTo(point.point[0], point.point[1]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.lineWidth = 8;
    ctx.stroke();
    ctx.strokeStyle = "rgba(30, 37, 40, 0.9)";
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  ctx.restore();
}

function drawInteractions(interactions) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const item of interactions) {
    const a = item.from;
    const b = item.to;
    if (!a || !b) continue;
    const color = item.class === "friendly" ? "#2f9e44" : "#e03131";
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.lineWidth = 24;
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 13;
    ctx.stroke();
  }
  ctx.restore();
}

function remapInteractionsForFrame(interactions, frame) {
  const points = pointMapForFrame(frame);
  const remapped = [];
  for (const item of interactions) {
    const from = points.get(Number(item.tidA));
    const to = points.get(Number(item.tidB));
    if (!from || !to) continue;
    remapped.push({ ...item, from, to });
  }
  return remapped;
}

function displayStatsForInteractions(stats, interactions) {
  return {
    ...stats,
    drawn: interactions.length,
    friendly: interactions.filter((item) => item.class === "friendly").length,
    unfriendly: interactions.filter((item) => item.class === "unfriendly").length,
  };
}

function stage2DisplayForFrame(frame) {
  if (state.stage2.frame === frame) {
    const remapped = remapInteractionsForFrame(state.stage2.interactions, frame);
    return {
      interactions: remapped,
      stats: displayStatsForInteractions(state.stage2.stats, remapped),
      exact: true,
    };
  }

  if (!state.stage2.displayInteractions.length) {
    return {
      interactions: [],
      stats: { drawn: 0, friendly: 0, unfriendly: 0, filteredByRed: 0, missingEndpoint: 0 },
      exact: false,
    };
  }

  const remapped = remapInteractionsForFrame(state.stage2.displayInteractions, frame);
  return {
    interactions: remapped,
    stats: displayStatsForInteractions(state.stage2.displayStats, remapped),
    exact: false,
  };
}

function drawCattle(points) {
  ctx.save();
  const scale = state.cattleScale;
  const radius = 24 * scale;
  const labelOffset = 34 * scale;
  const fontSize = 30 * scale;
  const labelRotation = -uiRotationDegrees() * Math.PI / 180;
  ctx.font = `700 ${fontSize}px Segoe UI, Arial, sans-serif`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  for (const point of points) {
    const [x, y] = point.point;
    const color = colorForGlobalUuid(point.globalTrackUuid);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 6 * scale;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "rgba(0, 0, 0, 0.78)";
    ctx.lineWidth = 7 * scale;
    const label = visibleGlobalId(point.globalTrackId, "cattle point");
    const labelX = x + labelOffset + ctx.measureText(label).width / 2;
    const labelY = y;
    ctx.save();
    ctx.translate(labelX, labelY);
    ctx.rotate(labelRotation);
    ctx.strokeText(label, 0, 0);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

function draw() {
  if (!state.data) return;
  const frame = Number(frameRange.value);
  const points = displayPointsForFrame(frame);
  const stage2Display = stage2DisplayForFrame(frame);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (state.options.base) drawBase();
  if (state.options.reference) drawReferenceAreas();
  if (state.options.resourceFloorplan) drawResourceFloorplan();
  if (state.options.trails) drawTrails(frame);
  if (state.options.interactions && stage2Display.interactions.length) drawInteractions(stage2Display.interactions);
  if (state.options.corrections) drawCorrections(points);
  drawCattle(points);
  frameValue.textContent = String(frame);
  pointValue.textContent = String(points.length);
  movedValue.textContent = String(points.filter((point) => Number(point.adjustmentPx) > 0.01).length);
  interactionValue.textContent = stage2Display.exact || stage2Display.interactions.length
    ? String(stage2Display.stats.drawn)
    : state.stage2.pending ? "..." : "0";
  blockedValue.textContent = stage2Display.exact || stage2Display.interactions.length
    ? String(stage2Display.stats.filteredByRed)
    : "0";
}

function setFrame(frame) {
  const nextFrame = clampFrame(frame);
  frameRange.value = String(nextFrame);
  state.video.lastFrame = nextFrame;
  seekVideoToFrame(nextFrame);
  draw();
  if (state.options.interactions) requestStage2(nextFrame);
}

function scheduleTick() {
  if (state.rafId) return;
  state.rafId = requestAnimationFrame(tick);
}

function cancelTick() {
  if (!state.rafId) return;
  cancelAnimationFrame(state.rafId);
  state.rafId = 0;
}

function tick(timestamp) {
  state.rafId = 0;
  if (!state.playing) return;
  if (currentVideoInfo().available) {
    if (!state.video.ready) return;
    if (visualVideo.paused) playVisualVideo();
    updateFloorplanFromVideo();
    scheduleTick();
    return;
  }
  const fps = state.data.meta.fps || 30;
  if (!state.lastTick || timestamp - state.lastTick >= 1000 / fps) {
    state.lastTick = timestamp;
    const next = Number(frameRange.value) + 1;
    setFrame(next > Number(frameRange.max) ? Number(frameRange.min) : next);
  }
  scheduleTick();
}

function togglePlay() {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? "Pause" : "Play";
  state.lastTick = 0;
  if (state.playing) {
    if (currentVideoInfo().available && state.video.ready) {
      playVisualVideo();
    }
    scheduleTick();
  } else {
    visualVideo.pause();
    cancelTick();
  }
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error(`sample request failed: ${response.status}`);
  const payload = await response.json();
  if (!payload.ok) throw new Error(payload.error || "sample request failed");
  state.samples = payload.samples || [];
  state.sampleId = payload.defaultSampleId || (state.samples[0] && state.samples[0].id) || "";
  syncFiltersToSample(sampleById(state.sampleId));
  renderFarmCameraOptions();
  renderSampleOptions();
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function applyLoadedPayload(payload, options = {}) {
  validateGlobalIdentityPayload(payload);
  state.data = payload;
  state.loadedSampleId = state.data.meta.sample.id;
  state.sampleId = state.loadedSampleId;
  syncFiltersToSample(state.data.meta.sample);
  renderFarmCameraOptions();
  renderSampleOptions();
  sampleSelect.value = state.sampleId;
  applyRotationForCurrentProfile();
  applyPlanRotationForCurrentProfile();
  applyUiRotationForCurrentProfile();
  updateSemanticWarpControl();
  buildIndexes();
  renderLegend();
  const meta = state.data.meta;
  frameRange.min = String(meta.frameMin);
  frameRange.max = String(meta.frameMax);
  const initialFrame = options.initialFrame === undefined
    ? Number(meta.frameMin)
    : options.initialFrame;
  if (
    !Number.isSafeInteger(initialFrame) ||
    initialFrame < Number(meta.frameMin) ||
    initialFrame > Number(meta.frameMax)
  ) {
    throw new Error("resolved frame is outside the loaded sample");
  }
  frameRange.value = String(initialFrame);
  trackValue.textContent = String(meta.globalIdentities.length);
  movedValue.textContent = "0";
  titleLine.textContent = titleForMappingProfile(meta.mappingProfile);
  sourceLine.textContent = meta.sample.label;
  statusLine.textContent = (
    `${meta.frameCount} frames, ${meta.displayDetectionRowCount} detections loaded; `
    + `${meta.reidInvalidRowCount} re-ID-invalid treated as missed`
  );
  configureVideoForSample();

  const zoom = defaultZoom();
  setZoomRangeFromDefault(zoom);
  setZoom(zoom);
  draw();
  updateLoadedControls();
  if (state.options.interactions) requestStage2(Number(frameRange.value));
  state.playing = true;
  playBtn.textContent = "Pause";
  state.lastTick = 0;
  scheduleTick();
}

async function load(sampleId = state.sampleId, options = {}) {
  const loadVersion = state.loadVersion + 1;
  state.loadVersion = loadVersion;
  state.loading = true;
  state.loadJobId = "";
  state.sampleId = sampleId;
  sampleSelect.value = sampleId;
  sampleSelect.disabled = true;
  updateLoadControl();
  clearLoadedView(`Starting ${sampleSelect.selectedOptions[0]?.textContent || sampleId}`);
  if (options.resetUiRotation !== false) resetUiDisplayRotation();
  try {
    const startResponse = await fetch("/api/load/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample: sampleId }),
    });
    if (!startResponse.ok) throw new Error(`load start failed: ${startResponse.status}`);
    const started = await startResponse.json();
    if (!started.ok) throw new Error(started.error || "load start failed");
    if (loadVersion !== state.loadVersion) return;
    state.loadJobId = started.jobId;
    statusLine.textContent = progressText(started.progress);

    while (loadVersion === state.loadVersion) {
      await delay(300);
      const statusResponse = await fetch(`/api/load/status?job=${encodeURIComponent(state.loadJobId)}`);
      if (!statusResponse.ok) throw new Error(`load status failed: ${statusResponse.status}`);
      const payload = await statusResponse.json();
      if (loadVersion !== state.loadVersion) return;
      statusLine.textContent = progressText(payload.progress);
      if (!payload.done) continue;
      if (!payload.ok) throw new Error(payload.error || "load failed");
      if (!payload.data) throw new Error("load completed without data");
      applyLoadedPayload(payload.data, options);
      return;
    }
  } finally {
    if (loadVersion === state.loadVersion) {
      state.loading = false;
      state.loadJobId = "";
      sampleSelect.disabled = false;
      updateLoadedControls();
    }
  }
}

async function requestStage2(frame) {
  const sampleId = state.sampleId;
  const loadVersion = state.loadVersion;
  const requestVersion = state.stage2.requestVersion;
  const displayMode = currentDisplayMode();
  state.stage2.targetFrame = frame;
  if (state.stage2.pending) return;

  while (state.stage2.targetFrame !== null) {
    if (
      loadVersion !== state.loadVersion ||
      sampleId !== state.sampleId ||
      requestVersion !== state.stage2.requestVersion ||
      displayMode !== currentDisplayMode()
    ) return;
    const nextFrame = state.stage2.targetFrame;
    state.stage2.targetFrame = null;
    state.stage2.pending = true;
    state.stage2.error = "";
    if (!state.stage2.displayInteractions.length) {
      interactionValue.textContent = "...";
      statusLine.textContent = `Stage2 loading frame ${nextFrame}`;
    }
    try {
      const response = await fetch(
        `/api/stage2?sample=${encodeURIComponent(sampleId)}&frame=${encodeURIComponent(nextFrame)}` +
          `&displayMode=${encodeURIComponent(displayMode)}`
      );
      if (!response.ok) throw new Error(`Stage2 request failed: ${response.status}`);
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error || "Stage2 request failed");
      if (
        loadVersion !== state.loadVersion ||
        sampleId !== state.sampleId ||
        requestVersion !== state.stage2.requestVersion ||
        displayMode !== currentDisplayMode()
      ) return;
      state.stage2.frame = payload.frame;
      state.stage2.interactions = payload.interactions || [];
      state.stage2.stats = payload.stats || state.stage2.stats;
      state.stage2.displayFrame = payload.frame;
      state.stage2.displayInteractions = state.stage2.interactions;
      state.stage2.displayStats = state.stage2.stats;
      if (Number(frameRange.value) === payload.frame) {
        const stats = state.stage2.stats;
        statusLine.textContent =
          `Stage2 ${stats.drawn} stable links (${stats.friendly} friendly, ${stats.unfriendly} unfriendly), ` +
          `${stats.filteredByRed} red-blocked`;
        draw();
      }
    } catch (error) {
      if (
        loadVersion !== state.loadVersion ||
        sampleId !== state.sampleId ||
        requestVersion !== state.stage2.requestVersion ||
        displayMode !== currentDisplayMode()
      ) return;
      state.stage2.error = error.message;
      state.stage2.frame = null;
      state.stage2.interactions = [];
      statusLine.textContent = error.message;
      draw();
    } finally {
      if (
        loadVersion === state.loadVersion &&
        sampleId === state.sampleId &&
        requestVersion === state.stage2.requestVersion &&
        displayMode === currentDisplayMode()
      ) {
        state.stage2.pending = false;
      }
    }
  }
}

frameRange.addEventListener("input", () => setFrame(Number(frameRange.value)));
playBtn.addEventListener("click", togglePlay);
stepBackBtn.addEventListener("click", () => setFrame(Number(frameRange.value) - 1));
stepForwardBtn.addEventListener("click", () => setFrame(Number(frameRange.value) + 1));
farmSelect.addEventListener("change", () => {
  state.filters.farmId = farmSelect.value;
  state.filters.cameraId = "";
  renderFarmCameraOptions();
  renderSampleOptions();
  state.sampleId = sampleSelect.value;
  state.loadVersion += 1;
  state.loading = false;
  clearLoadedView("Select a sample, then Load");
});
cameraSelect.addEventListener("change", () => {
  state.filters.cameraId = cameraSelect.value;
  renderSampleOptions();
  state.sampleId = sampleSelect.value;
  state.loadVersion += 1;
  state.loading = false;
  clearLoadedView("Select a sample, then Load");
});
sampleSelect.addEventListener("change", () => {
  state.sampleId = sampleSelect.value;
  state.loadVersion += 1;
  state.loading = false;
  clearLoadedView("Press Load for the selected sample");
});
loadBtn.addEventListener("click", () => {
  load(sampleSelect.value).catch((error) => {
    state.loading = false;
    statusLine.textContent = error.message;
    sampleSelect.disabled = false;
    updateLoadedControls();
  });
});
rotateLeftBtn.addEventListener("click", () => rotateDisplay(-1));
rotateRightBtn.addEventListener("click", () => rotateDisplay(1));
planRotateLeftBtn.addEventListener("click", () => rotatePlan(-1));
planRotateRightBtn.addEventListener("click", () => rotatePlan(1));
uiRotateLeftBtn.addEventListener("click", () => rotateUiDisplay(-1));
uiRotateRightBtn.addEventListener("click", () => rotateUiDisplay(1));
semanticWarpToggle.addEventListener("change", () => {
  state.mapping.semanticWarp = semanticWarpToggle.checked && !semanticWarpToggle.disabled;
  resetStage2State();
  draw();
  if (state.options.interactions) requestStage2(Number(frameRange.value));
});
zoomRange.addEventListener("input", () => {
  setZoom(Number(zoomRange.value) / 100);
  draw();
});
videoToggle.addEventListener("change", () => {
  setVideoEnabled(videoToggle.checked);
});
cattleScaleRange.addEventListener("input", () => {
  state.cattleScale = Number(cattleScaleRange.value);
  cattleScaleValue.textContent = `${state.cattleScale.toFixed(1)}x`;
  draw();
});

for (const [key, input] of Object.entries(toggles)) {
  input.addEventListener("change", () => {
    state.options[key] = input.checked;
    draw();
    if (key === "interactions" && input.checked) requestStage2(Number(frameRange.value));
  });
}

window.addEventListener("resize", () => {
  if (!state.data) return;
  if (state.video.enabled) {
    scheduleFloorplanFit();
    return;
  }
  const currentFit = defaultZoom();
  setZoomRangeFromDefault(currentFit);
  if (state.zoom <= currentFit + 0.001) {
    setZoom(currentFit);
    draw();
  } else if (state.zoom < state.zoomRange.min || state.zoom > state.zoomRange.max) {
    setZoom(state.zoom);
    draw();
  }
});

visualVideo.addEventListener("loadedmetadata", () => {
  state.video.ready = true;
  updateVideoControls();
  seekVideoToFrame(Number(frameRange.value));
  updateFloorplanFromVideo();
  if (state.video.enabled && state.playing) playVisualVideo();
});
visualVideo.addEventListener("timeupdate", updateFloorplanFromVideo);
visualVideo.addEventListener("seeked", updateFloorplanFromVideo);
visualVideo.addEventListener("play", () => {
  state.playing = true;
  playBtn.textContent = "Pause";
  scheduleTick();
});
visualVideo.addEventListener("pause", () => {
  if (visualVideo.ended) return;
  if (state.video.enabled && state.playing && !state.video.ready) return;
  state.playing = false;
  playBtn.textContent = "Play";
  cancelTick();
});
visualVideo.addEventListener("error", () => {
  stopPlayback();
  statusLine.textContent = "Video failed to load";
});
splitHandle.addEventListener("pointerdown", (event) => {
  if (!state.video.enabled) return;
  splitHandle.setPointerCapture(event.pointerId);
  updateSplitFromPointer(event.clientX);
  event.preventDefault();
});
splitHandle.addEventListener("pointermove", (event) => {
  if (!state.video.enabled || !splitHandle.hasPointerCapture(event.pointerId)) return;
  updateSplitFromPointer(event.clientX);
});
splitHandle.addEventListener("pointerup", (event) => {
  if (splitHandle.hasPointerCapture(event.pointerId)) {
    splitHandle.releasePointerCapture(event.pointerId);
  }
});
splitHandle.addEventListener("pointercancel", (event) => {
  if (splitHandle.hasPointerCapture(event.pointerId)) {
    splitHandle.releasePointerCapture(event.pointerId);
  }
});

updateRotationControls();
updatePlanRotationControls();
updateUiRotationControls();
updateSemanticWarpControl();
updateLoadedControls();

async function initialize() {
  await Promise.all([loadRotationSettings(), loadSamples()]);
  clearLoadedView("Select a sample, then Load");

  const selectionVersion = state.loadVersion;
  const target = await resolveInitialDeepLink();
  if (!target || selectionVersion !== state.loadVersion) return;
  const sample = sampleById(target.sampleId);
  if (!sample) throw new Error("resolved sample is absent from the selectable samples");

  syncFiltersToSample(sample);
  state.sampleId = sample.id;
  renderFarmCameraOptions();
  renderSampleOptions();
  await load(sample.id, { initialFrame: target.frameId });
}

initialize()
  .catch((error) => {
    statusLine.textContent = error.message;
    sampleSelect.disabled = false;
    updateRotationControls();
    updateLoadedControls();
  });
