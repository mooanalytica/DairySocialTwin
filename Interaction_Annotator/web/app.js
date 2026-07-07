const state = {
  clips: [],
  current: null,
  boxes: [],
  selectedBoxId: null,
  adding: false,
  dragging: null,
  fps: 30,
  frameCount: 1,
  mediaWidth: 0,
  mediaHeight: 0,
  videoRect: { x: 0, y: 0, w: 1, h: 1, scale: 1 },
};

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

function setStatus(text) {
  $("save-state").textContent = text;
}

function frameToSeconds(frame) {
  return frame / state.fps;
}

function secondsToFrame(seconds) {
  return Math.max(0, Math.min(state.frameCount - 1, Math.round(seconds * state.fps)));
}

function selectedBox() {
  return state.boxes.find((box) => box.id === state.selectedBoxId) || null;
}

function updateSaveAvailability() {
  const saveButton = $("save");
  const disabled = !state.current || state.boxes.length === 0;
  saveButton.disabled = disabled;
  saveButton.title = disabled ? "Add at least one ROI before saving" : "";
}

function clipDisplayName(clip) {
  const interactionClass = clip.interaction_class ? ` [${clip.interaction_class}]` : "";
  return `${clip.clip_id}${interactionClass}  ${clip.source_video_rel_path}`;
}

function reviewFrameCount(clip) {
  if (clip.review_frame_count) return Number(clip.review_frame_count);
  return Number(clip.source_end_frame) - Number(clip.source_start_frame) + 1;
}

function clipDimensions() {
  const width = state.mediaWidth || Number(state.current?.width || 1);
  const height = state.mediaHeight || Number(state.current?.height || 1);
  return {
    width: Math.max(1, width),
    height: Math.max(1, height),
  };
}

function isPendingClip(clip) {
  return clip.clip_exists && clip.annotation_status !== "annotated";
}

function clipValenceClass(clip) {
  if (clip.annotation_status !== "annotated") return "";
  if (clip.annotation_valence === "friendly") return "valence-friendly";
  if (clip.annotation_valence === "unfriendly") return "valence-unfriendly";
  if (clip.annotation_valence === "fake_interaction") return "valence-fake";
  if (clip.annotation_valence === "mixed_friendly_unfriendly") return "valence-mixed";
  return "valence-unfriendly";
}

function boxValenceClass(valence) {
  if (valence === "friendly") return "valence-friendly";
  if (valence === "fake_interaction") return "valence-fake";
  return "valence-unfriendly";
}

function findNextPendingClip(startIndex = 0, excludeClipId = "") {
  if (!state.clips.length) return null;
  const normalizedStart = ((startIndex % state.clips.length) + state.clips.length) % state.clips.length;
  for (let step = 0; step < state.clips.length; step += 1) {
    const clip = state.clips[(normalizedStart + step) % state.clips.length];
    if (clip.clip_id !== excludeClipId && isPendingClip(clip)) return clip;
  }
  return state.clips.find((clip) => clip.clip_exists) || null;
}

async function loadClips() {
  const response = await fetch("/api/clips");
  const payload = await response.json();
  state.clips = payload.clips || [];
  $("clip-count").textContent = String(state.clips.length);
  renderClipList();
  const firstPlayable = state.clips.find(isPendingClip) || state.clips.find((clip) => clip.clip_exists);
  if (firstPlayable) {
    await selectClip(firstPlayable.clip_id);
  }
}

function renderClipList() {
  const list = $("clip-list");
  list.textContent = "";
  for (const clip of state.clips) {
    const button = document.createElement("button");
    button.className = "clip-item";
    button.dataset.clipId = clip.clip_id;
    const valenceClass = clipValenceClass(clip);
    if (valenceClass) button.classList.add(valenceClass);
    if (state.current && state.current.clip_id === clip.clip_id) button.classList.add("active");
    button.disabled = !clip.clip_exists;
    const name = document.createElement("span");
    name.className = "clip-name";
    name.textContent = clipDisplayName(clip);
    const meta = document.createElement("span");
    meta.className = "clip-meta";
    const duration = clip.review_duration_s || clip.cache_duration_s || clip.duration_s;
    const start = clip.review_source_start_frame || clip.source_start_frame;
    const end = clip.review_source_end_frame || clip.source_end_frame;
    meta.textContent = `${duration}s  frames ${start}-${end}`;
    button.append(name, meta);
    button.addEventListener("click", () => selectClip(clip.clip_id));
    list.append(button);
  }
}

function scrollCurrentClipIntoView() {
  if (!state.current) return;
  const active = [...$("clip-list").querySelectorAll(".clip-item")]
    .find((button) => button.dataset.clipId === state.current.clip_id);
  if (active) active.scrollIntoView({ block: "nearest" });
}

async function selectClip(clipId) {
  const clip = state.clips.find((item) => item.clip_id === clipId);
  if (!clip || !clip.clip_exists) return;
  state.current = clip;
  state.fps = Number(clip.fps);
  state.frameCount = reviewFrameCount(clip);
  state.mediaWidth = 0;
  state.mediaHeight = 0;
  state.boxes = [];
  state.selectedBoxId = null;
  state.adding = false;
  video.src = `/media/${clip.clip_id}.mp4`;
  $("seek").max = String(state.frameCount - 1);
  $("seek").value = "0";
  $("start-frame").max = String(state.frameCount - 1);
  $("end-frame").max = String(state.frameCount - 1);
  $("start-frame").value = "0";
  $("end-frame").value = String(state.frameCount - 1);
  await loadAnnotation(clip.clip_id);
  renderClipList();
  scrollCurrentClipIntoView();
  renderBoxes();
  draw();
  setStatus("Idle");
}

async function loadAnnotation(clipId) {
  const response = await fetch(`/api/annotation?clip_id=${encodeURIComponent(clipId)}`);
  const payload = await response.json();
  const rows = payload.rows || [];
  state.boxes = rows.map((row, idx) => ({
    id: crypto.randomUUID(),
    x: Number(row.roi_x),
    y: Number(row.roi_y),
    w: Number(row.roi_w),
    h: Number(row.roi_h),
    valence: row.valence || "friendly",
    fine_class: row.fine_class || "",
    notes: row.notes || "",
  }));
  const savedStartRaw = state.current?.selected_cache_start_frame;
  const savedEndRaw = state.current?.selected_cache_end_frame;
  const savedStart = Number(savedStartRaw);
  const savedEnd = Number(savedEndRaw);
  if (savedStartRaw !== "" && savedEndRaw !== "" && Number.isFinite(savedStart) && Number.isFinite(savedEnd)) {
    $("start-frame").value = String(savedStart);
    $("end-frame").value = String(savedEnd);
  } else if (rows.length) {
    $("start-frame").value = rows[0].start_frame || "0";
    $("end-frame").value = rows[0].end_frame || String(state.frameCount - 1);
  }
  state.selectedBoxId = state.boxes[0]?.id || null;
  applySelectedBoxControls();
}

function applySelectedBoxControls() {
  const box = selectedBox();
  if (!box) return;
  $("valence").value = box.valence;
  $("fine-class").value = box.fine_class || "";
}

function renderBoxes() {
  const list = $("box-list");
  list.textContent = "";
  state.boxes.forEach((box, index) => {
    const button = document.createElement("button");
    button.className = "box-item";
    button.classList.add(boxValenceClass(box.valence));
    if (box.id === state.selectedBoxId) button.classList.add("active");
    const title = document.createElement("strong");
    title.textContent = box.valence;
    const meta = document.createElement("span");
    meta.textContent = `${Math.round(box.x)}, ${Math.round(box.y)}, ${Math.round(box.w)}, ${Math.round(box.h)}`;
    button.append(title, meta);
    button.addEventListener("click", () => {
      state.selectedBoxId = box.id;
      applySelectedBoxControls();
      renderBoxes();
      draw();
    });
    list.append(button);
  });
  updateSaveAvailability();
}

function enforceBounds() {
  const minFrames = Math.max(1, Math.round(state.fps));
  const startInput = $("start-frame");
  const endInput = $("end-frame");
  let start = Number(startInput.value);
  let end = Number(endInput.value);
  if (end - start + 1 < minFrames) {
    if (document.activeElement === startInput) start = Math.max(0, end - minFrames + 1);
    else end = Math.min(state.frameCount - 1, start + minFrames - 1);
  }
  if (end - start + 1 < minFrames) {
    start = 0;
    end = Math.min(state.frameCount - 1, minFrames - 1);
  }
  startInput.value = String(start);
  endInput.value = String(end);
}

function updateVideoRect() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

  const { width: vw, height: vh } = clipDimensions();
  const scale = Math.min(rect.width / vw, rect.height / vh);
  const w = vw * scale;
  const h = vh * scale;
  state.videoRect = {
    x: (rect.width - w) / 2,
    y: (rect.height - h) / 2,
    w,
    h,
    scale,
  };
}

function clipToCanvas(box) {
  const r = state.videoRect;
  return {
    x: r.x + box.x * r.scale,
    y: r.y + box.y * r.scale,
    w: box.w * r.scale,
    h: box.h * r.scale,
  };
}

function valenceColor(valence) {
  if (valence === "friendly") return "#38c783";
  if (valence === "fake_interaction") return "#5da8ff";
  return "#ff4d58";
}

function canvasToClip(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const r = state.videoRect;
  const { width, height } = clipDimensions();
  const clipX = Math.max(0, Math.min(width, (x - r.x) / r.scale));
  const clipY = Math.max(0, Math.min(height, (y - r.y) / r.scale));
  return { x: clipX, y: clipY };
}

function draw() {
  updateVideoRect();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "18px Segoe UI, Arial";
  state.boxes.forEach((box, index) => {
    const c = clipToCanvas(box);
    const selected = box.id === state.selectedBoxId;
    const color = valenceColor(box.valence);
    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeStyle = color;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.strokeRect(c.x, c.y, c.w, c.h);
    if (selected) {
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#edf0f2";
      ctx.strokeRect(c.x - 4, c.y - 4, c.w + 8, c.h + 8);
      ctx.restore();
    }
    const label = box.valence;
    const labelWidth = ctx.measureText(label).width + 12;
    ctx.fillRect(c.x, Math.max(0, c.y - 28), labelWidth, 26);
    ctx.fillStyle = "#edf0f2";
    ctx.fillText(label, c.x + 6, Math.max(18, c.y - 9));
  });
}

function pointerDown(event) {
  if (!state.current) return;
  const point = canvasToClip(event.clientX, event.clientY);
  if (state.adding) {
    const box = {
      id: crypto.randomUUID(),
      x: point.x,
      y: point.y,
      w: 1,
      h: 1,
      valence: $("valence").value,
      fine_class: $("fine-class").value,
      notes: "",
    };
    state.boxes.push(box);
    state.selectedBoxId = box.id;
    state.dragging = { type: "create", start: point, boxId: box.id };
    state.adding = false;
    canvas.setPointerCapture(event.pointerId);
    renderBoxes();
    draw();
    return;
  }

  for (const box of [...state.boxes].reverse()) {
    if (point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h) {
      state.selectedBoxId = box.id;
      state.dragging = {
        type: "move",
        start: point,
        boxId: box.id,
        original: { x: box.x, y: box.y, w: box.w, h: box.h },
      };
      canvas.setPointerCapture(event.pointerId);
      applySelectedBoxControls();
      renderBoxes();
      draw();
      return;
    }
  }
}

function pointerMove(event) {
  if (!state.dragging) return;
  const point = canvasToClip(event.clientX, event.clientY);
  const box = state.boxes.find((item) => item.id === state.dragging.boxId);
  if (!box) return;
  const { width: maxW, height: maxH } = clipDimensions();
  if (state.dragging.type === "create") {
    const x = Math.min(state.dragging.start.x, point.x);
    const y = Math.min(state.dragging.start.y, point.y);
    box.x = x;
    box.y = y;
    box.w = Math.max(1, Math.abs(point.x - state.dragging.start.x));
    box.h = Math.max(1, Math.abs(point.y - state.dragging.start.y));
  } else if (state.dragging.type === "move") {
    const dx = point.x - state.dragging.start.x;
    const dy = point.y - state.dragging.start.y;
    box.x = Math.max(0, Math.min(maxW - box.w, state.dragging.original.x + dx));
    box.y = Math.max(0, Math.min(maxH - box.h, state.dragging.original.y + dy));
  }
  renderBoxes();
  draw();
}

function pointerUp(event) {
  if (state.dragging) {
    canvas.releasePointerCapture(event.pointerId);
  }
  state.dragging = null;
}

function updateTimeUI() {
  if (!state.current) return;
  const frame = secondsToFrame(video.currentTime || 0);
  $("seek").value = String(frame);
  const duration = Number(state.current.review_duration_s || state.current.cache_duration_s || state.current.duration_s);
  $("time-label").textContent = `${(video.currentTime || 0).toFixed(3)} / ${duration.toFixed(3)}`;
  const end = Number($("end-frame").value);
  const start = Number($("start-frame").value);
  if (frame > end) {
    video.currentTime = frameToSeconds(start);
    video.play().catch(() => {});
  }
}

async function saveCurrent() {
  if (!state.current) return;
  if (state.boxes.length === 0) {
    setStatus("Add ROI before saving");
    updateSaveAvailability();
    return;
  }
  enforceBounds();
  setStatus("Saving");
  const payload = {
    clip_id: state.current.clip_id,
    start_frame: Number($("start-frame").value),
    end_frame: Number($("end-frame").value),
    media_width: clipDimensions().width,
    media_height: clipDimensions().height,
    boxes: state.boxes.map((box) => ({
      x: box.x,
      y: box.y,
      w: box.w,
      h: box.h,
      valence: box.valence,
      fine_class: box.fine_class || "",
      notes: box.notes || "",
    })),
  };
  const response = await fetch("/api/annotation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "save failed" }));
    setStatus(error.error || "Save failed");
    return;
  }
  const result = await response.json();
  if (result.clip) {
    state.current = { ...state.current, ...result.clip };
    state.frameCount = reviewFrameCount(state.current);
    const index = state.clips.findIndex((clip) => clip.clip_id === state.current.clip_id);
    if (index >= 0) state.clips[index] = state.current;
  }
  state.current.annotation_status = state.boxes.length ? "annotated" : "empty";
  const currentIndex = state.clips.findIndex((clip) => clip.clip_id === state.current.clip_id);
  const nextClip = findNextPendingClip(currentIndex + 1, state.current.clip_id);
  if (nextClip) {
    await selectClip(nextClip.clip_id);
    setStatus("Saved, next pending");
  } else {
    renderClipList();
    scrollCurrentClipIntoView();
    setStatus("Saved");
  }
}

async function deleteCurrentClip() {
  if (!state.current) return;
  const clipId = state.current.clip_id;
  const currentIndex = state.clips.findIndex((clip) => clip.clip_id === clipId);
  const ok = confirm(`Delete ${clipId}? This removes the video clip and its annotation CSV.`);
  if (!ok) return;

  video.pause();
  setStatus("Deleting");
  const response = await fetch(`/api/clip?clip_id=${encodeURIComponent(clipId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "delete failed" }));
    setStatus(error.error || "Delete failed");
    return;
  }
  const result = await response.json();
  const doneText = result.queued_cache || result.queued_cache_vis || result.queued_clip || result.queued_annotation
    ? "Queued cleanup"
    : "Deleted";

  state.clips = state.clips.filter((clip) => clip.clip_id !== clipId);
  state.current = null;
  state.boxes = [];
  state.selectedBoxId = null;
  video.removeAttribute("src");
  video.load();
  $("clip-count").textContent = String(state.clips.length);
  renderClipList();
  renderBoxes();
  draw();
  const nextPlayable = findNextPendingClip(currentIndex, clipId);
  if (nextPlayable) {
    await selectClip(nextPlayable.clip_id);
  }
  setStatus(doneText);
}

$("add-box").addEventListener("click", () => {
  state.adding = true;
  setStatus("Add ROI");
});
$("delete-box").addEventListener("click", () => {
  state.boxes = state.boxes.filter((box) => box.id !== state.selectedBoxId);
  state.selectedBoxId = state.boxes[0]?.id || null;
  applySelectedBoxControls();
  renderBoxes();
  draw();
});
$("clear-boxes").addEventListener("click", () => {
  state.boxes = [];
  state.selectedBoxId = null;
  renderBoxes();
  draw();
});
$("save").addEventListener("click", saveCurrent);
$("delete-clip").addEventListener("click", deleteCurrentClip);
$("play-pause").addEventListener("click", () => {
  if (video.paused) {
    video.play().catch(() => {});
    $("play-pause").textContent = "Pause";
  } else {
    video.pause();
    $("play-pause").textContent = "Play";
  }
});
$("replay").addEventListener("click", () => {
  video.currentTime = frameToSeconds(Number($("start-frame").value));
  video.play().catch(() => {});
});
$("seek").addEventListener("input", (event) => {
  video.currentTime = frameToSeconds(Number(event.target.value));
});
$("start-frame").addEventListener("input", enforceBounds);
$("end-frame").addEventListener("input", enforceBounds);
$("valence").addEventListener("change", (event) => {
  const box = selectedBox();
  if (box) {
    box.valence = event.target.value;
    renderBoxes();
    draw();
  }
});
$("fine-class").addEventListener("change", (event) => {
  const box = selectedBox();
  if (box) box.fine_class = event.target.value;
});
canvas.addEventListener("pointerdown", pointerDown);
canvas.addEventListener("pointermove", pointerMove);
canvas.addEventListener("pointerup", pointerUp);
window.addEventListener("resize", draw);
video.addEventListener("loadedmetadata", () => {
  state.mediaWidth = video.videoWidth || Number(state.current?.width || 1);
  state.mediaHeight = video.videoHeight || Number(state.current?.height || 1);
  video.currentTime = frameToSeconds(Number($("start-frame").value));
  video.play().catch(() => {});
  draw();
});
video.addEventListener("timeupdate", updateTimeUI);
video.addEventListener("play", () => ($("play-pause").textContent = "Pause"));
video.addEventListener("pause", () => ($("play-pause").textContent = "Play"));

loadClips().catch((error) => {
  setStatus(String(error));
});
