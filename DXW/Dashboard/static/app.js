"use strict";

const elements = {
  farmSelect: document.getElementById("farmSelect"),
  cameraSelect: document.getElementById("cameraSelect"),
  segmentSelect: document.getElementById("segmentSelect"),
  figureSelect: document.getElementById("figureSelect"),
  selectedCount: document.getElementById("selectedCount"),
  cattlePanel: document.getElementById("cattlePanel"),
  mobileCattleToggle: document.getElementById("mobileCattleToggle"),
  cattleModeNote: document.getElementById("cattleModeNote"),
  cattleSearch: document.getElementById("cattleSearch"),
  selectAllButton: document.getElementById("selectAllButton"),
  clearButton: document.getElementById("clearButton"),
  resetButton: document.getElementById("resetButton"),
  cattleList: document.getElementById("cattleList"),
  datasetStatus: document.getElementById("datasetStatus"),
  generationStatus: document.getElementById("generationStatus"),
  figureEyebrow: document.getElementById("figureEyebrow"),
  figureHeading: document.getElementById("figureHeading"),
  figureDescription: document.getElementById("figureDescription"),
  sampleMetric: document.getElementById("sampleMetric"),
  selectedMetric: document.getElementById("selectedMetric"),
  renderStatus: document.getElementById("renderStatus"),
  openImageLink: document.getElementById("openImageLink"),
  figureViewport: document.getElementById("figureViewport"),
  figureImage: document.getElementById("figureImage"),
  figureA11ySummary: document.getElementById("figureA11ySummary"),
  loadingState: document.getElementById("loadingState"),
  loadingTitle: document.getElementById("loadingTitle"),
  loadingDetail: document.getElementById("loadingDetail"),
  errorState: document.getElementById("errorState"),
  errorMessage: document.getElementById("errorMessage"),
  retryButton: document.getElementById("retryButton"),
  cowInfoCard: document.getElementById("cowInfoCard"),
  cowInfoId: document.getElementById("cowInfoId"),
  cowInfoPhoto: document.getElementById("cowInfoPhoto"),
  cowInfoAppearances: document.getElementById("cowInfoAppearances"),
  cowInfoClose: document.getElementById("cowInfoClose"),
};

const state = {
  phase: "booting",
  bootstrap: null,
  activeFigure: null,
  figuresByKey: new Map(),
  cattleById: new Map(),
  selections: new Map(),
  initialSelections: new Map(),
  requestVersion: 0,
  abortController: null,
  renderTimer: null,
  objectUrl: null,
  figureMap: null,
  infoCardTimer: null,
  infoRequestVersion: 0,
  infoAbortController: null,
  infoPhotoUrl: null,
  clientId: `client-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`,
};

const INFO_CARD_LIFETIME_MS = 30_000;
const INFO_CARD_GAP_PX = 14;
const VIEWPORT_EDGE_PX = 12;
const INTERACTIVE_FIGURES = new Set([
  "01",
  "02",
  "04A",
  "04B",
  "05A",
  "05B",
  "05C",
  "05D",
  "05E",
  "05F",
  "07",
  "08",
  "09",
]);

function closeCowInfoCard() {
  state.infoRequestVersion += 1;
  if (state.infoAbortController) {
    state.infoAbortController.abort();
    state.infoAbortController = null;
  }
  window.clearTimeout(state.infoCardTimer);
  state.infoCardTimer = null;
  elements.cowInfoCard.hidden = true;
  elements.cowInfoId.textContent = "";
  elements.cowInfoPhoto.removeAttribute("src");
  elements.cowInfoPhoto.alt = "";
  elements.cowInfoAppearances.replaceChildren();
  if (state.infoPhotoUrl) {
    URL.revokeObjectURL(state.infoPhotoUrl);
    state.infoPhotoUrl = null;
  }
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function positionCowInfoCard(clientX, clientY) {
  const cardWidth = elements.cowInfoCard.offsetWidth;
  const cardHeight = elements.cowInfoCard.offsetHeight;
  const maximumLeft = Math.max(
    VIEWPORT_EDGE_PX,
    window.innerWidth - cardWidth - VIEWPORT_EDGE_PX,
  );
  const maximumTop = Math.max(
    VIEWPORT_EDGE_PX,
    window.innerHeight - cardHeight - VIEWPORT_EDGE_PX,
  );

  let left = clientX + INFO_CARD_GAP_PX;
  if (left + cardWidth > window.innerWidth - VIEWPORT_EDGE_PX) {
    left = clientX - INFO_CARD_GAP_PX - cardWidth;
  }

  elements.cowInfoCard.style.left =
    `${Math.round(clamp(left, VIEWPORT_EDGE_PX, maximumLeft))}px`;
  elements.cowInfoCard.style.top =
    `${Math.round(clamp(clientY, VIEWPORT_EDGE_PX, maximumTop))}px`;
}

function showCowInfoCard(cowId, photoUrl, appearances, clientX, clientY) {
  const fragment = document.createDocumentFragment();
  for (const appearance of appearances) {
    const item = document.createElement("li");
    item.className = "cow-info-appearance";
    const link = document.createElement("a");
    link.className = "cow-info-appearance-link";
    link.href = appearance.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = appearance.display;
    link.setAttribute(
      "aria-label",
      `Open ${appearance.clip}-${appearance.segment}-${appearance.startFrame}, ` +
        `${appearance.display}, in WebUIL (opens in a new tab)`,
    );
    item.append(link);
    fragment.append(item);
  }

  elements.cowInfoId.textContent = cowId;
  elements.cowInfoPhoto.src = photoUrl;
  elements.cowInfoPhoto.alt = `Dairy cattle ${cowId} at its first appearance`;
  elements.cowInfoAppearances.replaceChildren(fragment);
  elements.cowInfoCard.hidden = false;
  positionCowInfoCard(clientX, clientY);
  state.infoCardTimer = window.setTimeout(closeCowInfoCard, INFO_CARD_LIFETIME_MS);
}

function setLoading(title, detail) {
  closeCowInfoCard();
  state.figureMap = null;
  state.phase = "rendering";
  elements.figureViewport.setAttribute("aria-busy", "true");
  elements.figureImage.hidden = true;
  elements.loadingTitle.textContent = title;
  elements.loadingDetail.textContent = detail;
  elements.loadingState.hidden = false;
  elements.errorState.hidden = true;
}

function hideLoading() {
  elements.figureViewport.setAttribute("aria-busy", "false");
  elements.loadingState.hidden = true;
}

function showError(message) {
  closeCowInfoCard();
  state.figureMap = null;
  state.phase = "error";
  elements.figureViewport.setAttribute("aria-busy", "false");
  elements.loadingState.hidden = true;
  elements.figureImage.hidden = true;
  elements.errorMessage.textContent = message;
  elements.errorState.hidden = false;
  elements.renderStatus.textContent = "Figure unavailable";
  disableImageLink();
}

function disableImageLink() {
  elements.openImageLink.classList.add("is-disabled");
  elements.openImageLink.setAttribute("aria-disabled", "true");
  elements.openImageLink.removeAttribute("href");
}

function setImageLink(url) {
  elements.openImageLink.href = url;
  elements.openImageLink.classList.remove("is-disabled");
  elements.openImageLink.setAttribute("aria-disabled", "false");
}

function currentFigure() {
  return state.figuresByKey.get(state.activeFigure);
}

function currentSelection() {
  return state.selections.get(state.activeFigure) || new Set();
}

function figureUsesSelection() {
  return currentFigure()?.cattleMode === "filter";
}

function updateSelectionSummary() {
  const enabled = figureUsesSelection();
  const count = enabled ? currentSelection().size : 62;
  elements.selectedCount.textContent = enabled ? `${count} / 62` : "All 62";
  elements.selectedMetric.textContent = enabled ? `${count} cattle` : "All cattle";
  elements.cattleModeNote.textContent = enabled
    ? "Choose cattle to include. Changes are preserved while switching figures."
    : "All cattle — this figure is not cattle-filtered.";
}

function updateCattleControls() {
  const enabled = figureUsesSelection();
  elements.cattleSearch.disabled = !enabled;
  elements.selectAllButton.disabled = !enabled;
  elements.clearButton.disabled = !enabled;
  elements.resetButton.disabled = !enabled;
  const selected = currentSelection();
  for (const option of elements.cattleList.querySelectorAll(".cattle-option")) {
    const checkbox = option.querySelector("input");
    checkbox.disabled = !enabled;
    checkbox.checked = enabled && selected.has(checkbox.value);
    option.classList.toggle("is-disabled", !enabled);
  }
  updateSelectionSummary();
}

function updateFigureCopy() {
  const figure = currentFigure();
  if (!figure) {
    return;
  }
  elements.figureEyebrow.textContent =
    `SNA results · Farm ${state.bootstrap.dataset.farm} / Camera ${state.bootstrap.dataset.camera}`;
  elements.figureHeading.textContent = figure.heading;
  elements.figureDescription.textContent = figure.description;
  elements.figureA11ySummary.textContent =
    `${figure.label}. ${figure.description} ` +
    "The image uses the fixed full-dataset visual scale where applicable.";
}

function setCattleCollapsed(collapsed) {
  elements.cattlePanel.classList.toggle("is-collapsed", collapsed);
  elements.mobileCattleToggle.setAttribute("aria-expanded", String(!collapsed));
  elements.mobileCattleToggle.textContent = collapsed ? "Show" : "Hide";
}

function filterCattleList() {
  const query = elements.cattleSearch.value.trim().toLowerCase();
  for (const option of elements.cattleList.querySelectorAll(".cattle-option")) {
    const label = option.dataset.searchValue;
    option.classList.toggle("is-hidden", Boolean(query) && !label.includes(query));
  }
}

function buildCattleList(cattle) {
  const fragment = document.createDocumentFragment();
  for (const item of cattle) {
    const label = document.createElement("label");
    label.className = "cattle-option";
    label.dataset.searchValue = `${item.label} ${item.id}`.toLowerCase();

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = item.id;
    checkbox.setAttribute("aria-label", `Dairy cattle ${item.label}`);

    const color = document.createElement("span");
    color.className = "identity-color";
    color.style.backgroundColor = item.color;
    color.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.className = "identity-label";
    text.textContent = `ID ${item.label.padStart(2, "0")}`;

    label.append(checkbox, color, text);
    fragment.append(label);
  }
  elements.cattleList.replaceChildren(fragment);
  elements.cattleList.setAttribute("aria-busy", "false");
}

function populateFigureSelect(figures) {
  const fragment = document.createDocumentFragment();
  for (const figure of figures) {
    const option = document.createElement("option");
    option.value = figure.key;
    const zoneSuffix = figure.zone ? ` · ${figure.zone.replaceAll("_", " ")}` : "";
    option.textContent = `${figure.label}${zoneSuffix}`;
    fragment.append(option);
  }
  elements.figureSelect.replaceChildren(fragment);
}

function setControlsReady() {
  elements.figureSelect.disabled = false;
  updateCattleControls();
}

function scheduleRender(delay = 150) {
  closeCowInfoCard();
  state.figureMap = null;
  window.clearTimeout(state.renderTimer);
  state.renderTimer = window.setTimeout(() => {
    renderActiveFigure();
  }, delay);
}

async function parseErrorResponse(response) {
  try {
    const payload = await response.json();
    return payload?.error?.message || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

function requireExactObjectKeys(value, expectedKeys, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  const actualKeys = Object.keys(value);
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key) => !expectedKeys.includes(key))
  ) {
    throw new Error(`${name} has unexpected or missing fields`);
  }
  return value;
}

function requireAppearanceTime(value, name) {
  if (
    typeof value !== "string" ||
    !/^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$/.test(value)
  ) {
    throw new Error(`${name} must use valid HH:MM:SS time`);
  }
  return value;
}

function requireAppearanceHref(value, appearance, name) {
  if (typeof value !== "string" || value.length === 0 || value !== value.trim()) {
    throw new Error(`${name} must be a non-empty trimmed string`);
  }

  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${name} must be an absolute URL`);
  }
  if (
    url.protocol !== "http:" ||
    url.host !== "172.17.6.39:9922" ||
    url.pathname !== "/" ||
    url.hash !== "" ||
    url.username !== "" ||
    url.password !== ""
  ) {
    throw new Error(`${name} must target the canonical WebUIL root`);
  }

  const expectedKeys = ["farmID", "cameraID", "clipID", "segmentID", "frameID"];
  const actualKeys = Array.from(url.searchParams.keys());
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new Error(`${name} must contain each WebUIL parameter exactly once`);
  }
  const clipId = url.searchParams.get("clipID");
  if (
    !/^[1-9]\d*$/.test(url.searchParams.get("farmID") || "") ||
    !/^[1-9]\d*$/.test(url.searchParams.get("cameraID") || "") ||
    !clipId ||
    clipId !== clipId.trim()
  ) {
    throw new Error(`${name} contains an invalid farmID, cameraID, or clipID`);
  }
  const expectedSegmentId =
    `${appearance.clip}-${appearance.segment}-${appearance.startFrame}`;
  if (
    url.searchParams.get("segmentID") !== expectedSegmentId ||
    url.searchParams.get("frameID") !== String(appearance.startFrame)
  ) {
    throw new Error(`${name} does not match its appearance frame`);
  }
  return url.toString();
}

function validateAppearanceResponse(payload, expectedCowId) {
  const response = requireExactObjectKeys(
    payload,
    ["ok", "cowId", "appearances"],
    "Cattle appearance response",
  );
  if (response.ok !== true) {
    throw new Error("Cattle appearance response was not successful");
  }
  if (response.cowId !== expectedCowId) {
    throw new Error(
      `Cattle appearance response is for ${String(response.cowId)}, expected ${expectedCowId}`,
    );
  }
  if (!Array.isArray(response.appearances)) {
    throw new Error("Cattle appearance response appearances must be an array");
  }

  let previousFrameCount = Number.POSITIVE_INFINITY;
  return response.appearances.map((value, index) => {
    const name = `Cattle appearance ${index}`;
    const appearance = requireExactObjectKeys(
      value,
      [
        "clip",
        "segment",
        "startFrame",
        "endFrameExclusive",
        "frameCount",
        "startTime",
        "endTime",
        "display",
        "href",
      ],
      name,
    );
    if (
      typeof appearance.clip !== "string" ||
      appearance.clip.length === 0 ||
      appearance.clip !== appearance.clip.trim()
    ) {
      throw new Error(`${name} clip must be a non-empty trimmed string`);
    }
    if (!Number.isSafeInteger(appearance.segment) || appearance.segment < 1) {
      throw new Error(`${name} segment must be a positive integer`);
    }
    if (!Number.isSafeInteger(appearance.startFrame) || appearance.startFrame < 0) {
      throw new Error(`${name} startFrame must be a non-negative integer`);
    }
    if (
      !Number.isSafeInteger(appearance.endFrameExclusive) ||
      appearance.endFrameExclusive <= appearance.startFrame
    ) {
      throw new Error(`${name} endFrameExclusive must be after startFrame`);
    }
    if (!Number.isSafeInteger(appearance.frameCount) || appearance.frameCount <= 0) {
      throw new Error(`${name} frameCount must be a positive integer`);
    }
    if (
      appearance.frameCount !==
      appearance.endFrameExclusive - appearance.startFrame
    ) {
      throw new Error(`${name} frameCount does not match its frame bounds`);
    }
    if (appearance.frameCount > previousFrameCount) {
      throw new Error("Cattle appearances must be sorted by frameCount descending");
    }

    const startTime = requireAppearanceTime(appearance.startTime, `${name} startTime`);
    const endTime = requireAppearanceTime(appearance.endTime, `${name} endTime`);
    const expectedDisplay = `${startTime}-${endTime}`;
    if (appearance.display !== expectedDisplay) {
      throw new Error(`${name} display does not match its source fields`);
    }
    const href = requireAppearanceHref(appearance.href, appearance, `${name} href`);

    previousFrameCount = appearance.frameCount;
    return {
      clip: appearance.clip,
      segment: appearance.segment,
      startFrame: appearance.startFrame,
      endFrameExclusive: appearance.endFrameExclusive,
      frameCount: appearance.frameCount,
      startTime,
      endTime,
      display: appearance.display,
      href,
    };
  });
}

async function readCowPhotoBlob(response) {
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }

  const contentType = (response.headers.get("Content-Type") || "")
    .split(";", 1)[0]
    .trim()
    .toLowerCase();
  if (contentType !== "image/jpeg") {
    throw new Error("Cattle photo response must use image/jpeg content type");
  }

  const blob = await response.blob();
  if (
    !(blob instanceof Blob) ||
    blob.size <= 0 ||
    blob.type.toLowerCase() !== "image/jpeg"
  ) {
    throw new Error("Cattle photo response was not a non-empty JPEG blob");
  }
  return blob;
}

function isCurrentCowInfoRequest(controller, requestVersion) {
  return (
    !controller.signal.aborted &&
    requestVersion === state.infoRequestVersion &&
    state.infoAbortController === controller
  );
}

async function loadCowInfoCard(cowId, clientX, clientY) {
  closeCowInfoCard();
  if (!/^G\d{4}$/.test(cowId) || !state.cattleById.has(cowId)) {
    console.error(`Cannot load information for unknown cattle ID ${String(cowId)}`);
    return;
  }

  const generation = state.bootstrap?.dataset?.generationId;
  if (typeof generation !== "string" || generation.length === 0) {
    console.error("Cannot load cattle information without a generation ID");
    return;
  }

  const requestVersion = state.infoRequestVersion;
  const controller = new AbortController();
  state.infoAbortController = controller;
  const query = new URLSearchParams({cow: cowId, generation});
  const requestQuery = query.toString();
  let pendingPhotoUrl = null;

  try {
    const [appearanceResponse, photoResponse] = await Promise.all([
      fetch(`/api/cow-appearances?${requestQuery}`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      }),
      fetch(`/api/cow-photo?${requestQuery}`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      }),
    ]);
    if (!appearanceResponse.ok) {
      throw new Error(await parseErrorResponse(appearanceResponse));
    }

    const [payload, photoBlob] = await Promise.all([
      appearanceResponse.json().catch(() => {
        throw new Error("Cattle appearance response was not valid JSON");
      }),
      readCowPhotoBlob(photoResponse),
    ]);
    const appearances = validateAppearanceResponse(payload, cowId);
    if (!isCurrentCowInfoRequest(controller, requestVersion)) {
      return;
    }

    pendingPhotoUrl = URL.createObjectURL(photoBlob);
    const decodedImage = new Image();
    decodedImage.src = pendingPhotoUrl;
    try {
      await decodedImage.decode();
    } catch {
      throw new Error("Cattle photo response could not be decoded as JPEG");
    }
    if (
      !decodedImage.complete ||
      decodedImage.naturalWidth <= 0 ||
      decodedImage.naturalHeight <= 0
    ) {
      throw new Error("Cattle photo response decoded without valid dimensions");
    }
    if (!isCurrentCowInfoRequest(controller, requestVersion)) {
      return;
    }

    state.infoPhotoUrl = pendingPhotoUrl;
    pendingPhotoUrl = null;
    state.infoAbortController = null;
    showCowInfoCard(cowId, state.infoPhotoUrl, appearances, clientX, clientY);
  } catch (error) {
    if (state.infoAbortController === controller) {
      state.infoAbortController = null;
    }
    if (controller.signal.aborted || requestVersion !== state.infoRequestVersion) {
      return;
    }
    console.error(
      `Unable to load information for ${cowId}:`,
      error instanceof Error ? error.message : error,
    );
  } finally {
    if (pendingPhotoUrl) {
      URL.revokeObjectURL(pendingPhotoUrl);
    }
  }
}

function requireUnitCoordinate(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`Figure map field ${name} must be a number from 0 to 1`);
  }
  return value;
}

function validateFigureMap(payload, expectedFigure) {
  if (!payload || typeof payload !== "object" || payload.ok !== true) {
    throw new Error(payload?.error?.message || "Figure map response was not valid");
  }
  if (payload.figure !== expectedFigure) {
    throw new Error(
      `Figure map is for ${String(payload.figure)}, expected ${expectedFigure}`,
    );
  }
  if (!Number.isInteger(payload.width) || payload.width <= 0) {
    throw new Error("Figure map width must be a positive integer");
  }
  if (!Number.isInteger(payload.height) || payload.height <= 0) {
    throw new Error("Figure map height must be a positive integer");
  }
  if (!Array.isArray(payload.regions)) {
    throw new Error("Figure map regions must be an array");
  }

  const regions = payload.regions.map((region, index) => {
    if (!region || typeof region !== "object") {
      throw new Error(`Figure map region ${index} was not valid`);
    }
    if (typeof region.cowId !== "string" || region.cowId.length === 0) {
      throw new Error(`Figure map region ${index} has no cattle ID`);
    }
    if (region.shape === "ellipse") {
      const cx = requireUnitCoordinate(region.cx, `regions[${index}].cx`);
      const cy = requireUnitCoordinate(region.cy, `regions[${index}].cy`);
      const rx = requireUnitCoordinate(region.rx, `regions[${index}].rx`);
      const ry = requireUnitCoordinate(region.ry, `regions[${index}].ry`);
      if (rx === 0 || ry === 0) {
        throw new Error(`Figure map ellipse ${index} must have positive radii`);
      }
      if (region.pick !== undefined && region.pick !== "nearest") {
        throw new Error(`Figure map ellipse ${index} has an unsupported pick rule`);
      }
      return {
        shape: "ellipse",
        cowId: region.cowId,
        cx,
        cy,
        rx,
        ry,
        pick: region.pick,
      };
    }
    if (region.shape === "rect") {
      const x0 = requireUnitCoordinate(region.x0, `regions[${index}].x0`);
      const y0 = requireUnitCoordinate(region.y0, `regions[${index}].y0`);
      const x1 = requireUnitCoordinate(region.x1, `regions[${index}].x1`);
      const y1 = requireUnitCoordinate(region.y1, `regions[${index}].y1`);
      if (x1 < x0 || y1 < y0) {
        throw new Error(`Figure map rectangle ${index} has reversed bounds`);
      }
      return {shape: "rect", cowId: region.cowId, x0, y0, x1, y1};
    }
    throw new Error(`Figure map region ${index} has an unsupported shape`);
  });

  return {
    figure: payload.figure,
    width: payload.width,
    height: payload.height,
    regions,
  };
}

function regionContains(region, x, y) {
  if (region.shape === "rect") {
    return x >= region.x0 && x <= region.x1 && y >= region.y0 && y <= region.y1;
  }
  const horizontal = (x - region.cx) / region.rx;
  const vertical = (y - region.cy) / region.ry;
  return horizontal * horizontal + vertical * vertical <= 1;
}

function pickFigureRegion(figureMap, x, y) {
  const hits = figureMap.regions.filter((region) => regionContains(region, x, y));
  if (hits.length === 0) {
    return null;
  }

  const nearestHits = hits.filter(
    (region) => region.shape === "ellipse" && region.pick === "nearest",
  );
  if (nearestHits.length === 0) {
    return hits[0];
  }

  return nearestHits.reduce((nearest, candidate) => {
    const nearestX = (x - nearest.cx) * figureMap.width;
    const nearestY = (y - nearest.cy) * figureMap.height;
    const candidateX = (x - candidate.cx) * figureMap.width;
    const candidateY = (y - candidate.cy) * figureMap.height;
    const nearestDistance = nearestX * nearestX + nearestY * nearestY;
    const candidateDistance = candidateX * candidateX + candidateY * candidateY;
    return candidateDistance < nearestDistance ? candidate : nearest;
  });
}

function handleFigureImageClick(event) {
  const figureMap = state.figureMap;
  if (
    state.phase !== "ready" ||
    !figureMap ||
    figureMap.figure !== state.activeFigure ||
    !INTERACTIVE_FIGURES.has(state.activeFigure)
  ) {
    return;
  }

  const bounds = elements.figureImage.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) {
    return;
  }
  const x = (event.clientX - bounds.left) / bounds.width;
  const y = (event.clientY - bounds.top) / bounds.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) {
    return;
  }

  const region = pickFigureRegion(figureMap, x, y);
  if (region) {
    void loadCowInfoCard(region.cowId, event.clientX, event.clientY);
  }
}

async function renderActiveFigure() {
  const figure = currentFigure();
  if (!figure) {
    return;
  }
  const requestVersion = ++state.requestVersion;
  if (state.abortController) {
    state.abortController.abort();
  }
  const controller = new AbortController();
  state.abortController = controller;

  const selected = figure.cattleMode === "filter"
    ? [...currentSelection()].sort()
    : [];
  const query = new URLSearchParams({
    figure: figure.key,
    cattle: selected.join(","),
    generation: state.bootstrap.dataset.generationId,
    client: state.clientId,
    request: String(requestVersion),
  });

  setLoading(
    `Rendering ${figure.label}`,
    figure.cattleMode === "filter"
      ? `${selected.length} cattle selected`
      : "Using all precomputed result rows",
  );
  elements.renderStatus.textContent = `Rendering ${figure.label}…`;
  disableImageLink();

  try {
    const requestQuery = query.toString();
    const [response, mapResponse] = await Promise.all([
      fetch(`/api/figure?${requestQuery}`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      }),
      fetch(`/api/figure-map?${requestQuery}`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      }),
    ]);
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }
    if (!mapResponse.ok) {
      throw new Error(await parseErrorResponse(mapResponse));
    }
    const [blob, mapPayload] = await Promise.all([
      response.blob(),
      mapResponse.json().catch(() => {
        throw new Error("Figure map response was not valid JSON");
      }),
    ]);
    const figureMap = validateFigureMap(mapPayload, figure.key);
    if (requestVersion !== state.requestVersion) {
      return;
    }
    const nextUrl = URL.createObjectURL(blob);
    const previousUrl = state.objectUrl;

    const decodedImage = new Image();
    try {
      await new Promise((resolve, reject) => {
        decodedImage.onload = resolve;
        decodedImage.onerror = () => reject(new Error("The rendered PNG could not be decoded"));
        decodedImage.src = nextUrl;
      });
      if (
        decodedImage.naturalWidth !== figureMap.width ||
        decodedImage.naturalHeight !== figureMap.height
      ) {
        throw new Error(
          "The rendered PNG dimensions do not match the figure map",
        );
      }
    } catch (error) {
      URL.revokeObjectURL(nextUrl);
      throw error;
    }
    if (requestVersion !== state.requestVersion) {
      URL.revokeObjectURL(nextUrl);
      return;
    }
    state.objectUrl = nextUrl;
    state.figureMap = figureMap;
    if (previousUrl) {
      URL.revokeObjectURL(previousUrl);
    }
    elements.figureImage.src = nextUrl;
    const displayedCount = figure.cattleMode === "filter" ? selected.length : 62;
    elements.figureImage.alt =
      `${figure.label}: ${figure.heading}, ${displayedCount} cattle`;
    elements.figureImage.hidden = false;
    elements.errorState.hidden = true;
    hideLoading();
    setImageLink(nextUrl);
    state.phase = "ready";
    elements.renderStatus.textContent =
      figure.cattleMode === "filter"
        ? `${figure.label} ready · ${selected.length} cattle`
        : `${figure.label} ready · all cattle`;
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    if (requestVersion === state.requestVersion) {
      showError(error.message || "Unknown rendering error");
    }
  }
}

function handleFigureChange() {
  closeCowInfoCard();
  state.activeFigure = elements.figureSelect.value;
  updateFigureCopy();
  updateCattleControls();
  filterCattleList();
  scheduleRender(0);
}

function handleCattleChange(event) {
  const checkbox = event.target.closest('input[type="checkbox"]');
  if (!checkbox || !figureUsesSelection()) {
    return;
  }
  const selected = currentSelection();
  if (checkbox.checked) {
    selected.add(checkbox.value);
  } else {
    selected.delete(checkbox.value);
  }
  updateSelectionSummary();
  scheduleRender();
}

function replaceSelection(values) {
  if (!figureUsesSelection()) {
    return;
  }
  state.selections.set(state.activeFigure, new Set(values));
  updateCattleControls();
  scheduleRender(0);
}

async function bootstrap() {
  state.phase = "booting";
  setLoading("Loading dashboard data", "Validating the precomputed result contract");
  elements.renderStatus.textContent = "Connecting to the local result server…";
  elements.errorState.hidden = true;
  try {
    const response = await fetch("/api/bootstrap", {
      method: "GET",
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload?.error?.message || "Bootstrap response was not valid");
    }
    state.bootstrap = payload;
    state.figuresByKey = new Map(payload.figures.map((item) => [item.key, item]));
    state.cattleById = new Map(payload.cattle.map((item) => [item.id, item]));
    state.selections = new Map();
    state.initialSelections = new Map();
    for (const figure of payload.figures) {
      const initial = new Set(figure.initialSelection);
      state.selections.set(figure.key, new Set(initial));
      state.initialSelections.set(figure.key, new Set(initial));
    }
    state.activeFigure = payload.initialFigure;

    elements.farmSelect.value = payload.dataset.farm;
    elements.cameraSelect.value = payload.dataset.camera;
    elements.segmentSelect.value = payload.dataset.segment;
    elements.sampleMetric.textContent = payload.dataset.sampleId;
    elements.datasetStatus.textContent = `${payload.dataset.sampleId} ready`;
    elements.generationStatus.textContent =
      `generation ${payload.dataset.generationId.slice(0, 8)}`;

    populateFigureSelect(payload.figures);
    buildCattleList(payload.cattle);
    elements.figureSelect.value = state.activeFigure;
    updateFigureCopy();
    setControlsReady();
    filterCattleList();
    await renderActiveFigure();
  } catch (error) {
    showError(error.message || "Unable to load dashboard data");
    elements.datasetStatus.textContent = "Result contract unavailable";
    elements.generationStatus.textContent = "See server log";
  }
}

elements.figureSelect.addEventListener("change", handleFigureChange);
elements.cattleList.addEventListener("change", handleCattleChange);
elements.cattleSearch.addEventListener("input", filterCattleList);
elements.selectAllButton.addEventListener("click", () => {
  replaceSelection(state.bootstrap.cattle.map((item) => item.id));
});
elements.clearButton.addEventListener("click", () => {
  replaceSelection([]);
});
elements.resetButton.addEventListener("click", () => {
  replaceSelection(state.initialSelections.get(state.activeFigure) || []);
});
elements.retryButton.addEventListener("click", () => {
  if (state.bootstrap && currentFigure()) {
    renderActiveFigure();
  } else {
    bootstrap();
  }
});
elements.figureImage.addEventListener("click", handleFigureImageClick);
elements.cowInfoClose.addEventListener("click", closeCowInfoCard);
elements.mobileCattleToggle.addEventListener("click", () => {
  setCattleCollapsed(!elements.cattlePanel.classList.contains("is-collapsed"));
});

window.addEventListener("beforeunload", () => {
  closeCowInfoCard();
  if (state.abortController) {
    state.abortController.abort();
  }
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
  }
});

setCattleCollapsed(window.matchMedia("(max-width: 860px)").matches);
bootstrap();
