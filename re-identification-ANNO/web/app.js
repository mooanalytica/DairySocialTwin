(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const elements = {
    shell: document.querySelector(".app-shell"),
    reviewCount: $("reviewCount"),
    completionTrack: $("completionTrack"),
    completionBar: $("completionBar"),
    pendingCount: $("pendingCount"),
    occurrenceList: $("occurrenceList"),
    videoHeading: $("videoHeading"),
    videoStatus: $("videoStatus"),
    video: $("reviewVideo"),
    videoMessage: $("videoMessage"),
    playPauseButton: $("playPauseButton"),
    videoProgress: $("videoProgress"),
    timeDisplay: $("timeDisplay"),
    decisionStatus: $("decisionStatus"),
    detailOccurrence: $("detailOccurrence"),
    detailGid: $("detailGid"),
    detailClip: $("detailClip"),
    detailTrack: $("detailTrack"),
    detailFrames: $("detailFrames"),
    detailDuration: $("detailDuration"),
    currentReview: $("currentReview"),
    acceptButton: $("acceptButton"),
    invalidButton: $("invalidButton"),
    updateAction: document.querySelector(".update-action"),
    newIdInput: $("newIdInput"),
    updateButton: $("updateButton"),
    submitMessage: $("submitMessage"),
    fatalError: $("fatalError"),
  };

  const state = {
    payload: null,
    occurrences: [],
    selectedId: null,
    listElements: new Map(),
    submitting: false,
    videoSelection: 0,
  };

  function isReviewed(item) {
    return item.review !== null;
  }

  function selectedOccurrence() {
    return state.occurrences.find((item) => item.occurrence_id === state.selectedId) || null;
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      cache: "no-store",
      credentials: "same-origin",
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });

    if (!response.ok) {
      let detail = "";
      try {
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          const body = await response.json();
          detail = typeof body.error === "string"
            ? body.error
            : body.error?.message || body.message || body.detail || "";
        } else {
          detail = (await response.text()).trim();
        }
      } catch (_error) {
        detail = "";
      }
      throw new Error(detail || `Request failed (${response.status})`);
    }

    if (response.status === 204) return null;
    const text = await response.text();
    return text ? JSON.parse(text) : null;
  }

  function validateState(payload) {
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.occurrences)) {
      throw new Error("The server returned an invalid state document.");
    }

    const ids = new Set();
    for (const item of payload.occurrences) {
      if (!item || typeof item !== "object" || typeof item.occurrence_id !== "string") {
        throw new Error("An occurrence in the server response is invalid.");
      }
      if (ids.has(item.occurrence_id)) {
        throw new Error(`Duplicate occurrence ID: ${item.occurrence_id}`);
      }
      if (item.review !== null && (typeof item.review !== "object" || !item.review.action)) {
        throw new Error(`Invalid review data for ${item.occurrence_id}.`);
      }
      if (!Number.isFinite(item.playback_start_sec) || item.playback_start_sec < 0) {
        throw new Error(`Invalid playback start time for ${item.occurrence_id}.`);
      }
      ids.add(item.occurrence_id);
    }
    return payload;
  }

  function gidColors(gid) {
    let hash = 2166136261;
    for (const character of String(gid || "pending")) {
      hash ^= character.charCodeAt(0);
      hash = Math.imul(hash, 16777619);
    }
    const hue = Math.abs(hash) % 360;
    return [`hsl(${hue} 68% 55%)`, `hsl(${hue} 38% 18%)`];
  }

  function formatClock(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
    const whole = Math.floor(seconds);
    const hours = Math.floor(whole / 3600);
    const minutes = Math.floor((whole % 3600) / 60);
    const remainder = whole % 60;
    return hours
      ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
      : `${minutes}:${String(remainder).padStart(2, "0")}`;
  }

  function reviewLabel(review) {
    if (!review) return "Pending";
    if (review.action === "accept") return "Accepted";
    if (review.action === "invalid_multiple_cows") return "Invalid: multiple cows";
    if (review.action === "update_id") return `Updated to ${review.reviewed_global_id}`;
    return review.action;
  }

  function renderProgress() {
    const total = state.occurrences.length;
    const reviewed = state.occurrences.filter(isReviewed).length;
    const pending = total - reviewed;
    const percent = total ? (reviewed / total) * 100 : 0;
    elements.reviewCount.textContent = `Reviewed ${reviewed.toLocaleString()} of ${total.toLocaleString()}`;
    elements.pendingCount.textContent = `${pending.toLocaleString()} pending`;
    elements.completionBar.style.width = `${percent}%`;
    elements.completionTrack.setAttribute("aria-valuenow", String(Math.round(percent)));
  }

  function renderList() {
    state.listElements.clear();
    const fragment = document.createDocumentFragment();

    if (!state.occurrences.length) {
      const message = document.createElement("p");
      message.className = "empty-message";
      message.textContent = "No occurrences are available.";
      fragment.append(message);
    }

    for (const item of state.occurrences) {
      const button = document.createElement("button");
      const reviewed = isReviewed(item);
      const [color, soft] = gidColors(item.display_global_id);
      button.type = "button";
      button.className = `occurrence-item${reviewed ? " reviewed" : ""}`;
      button.dataset.occurrenceId = item.occurrence_id;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", "false");
      button.style.setProperty("--gid-color", color);
      button.style.setProperty("--gid-soft", soft);

      const top = document.createElement("span");
      top.className = "item-topline";
      const occurrenceId = document.createElement("span");
      occurrenceId.className = "item-id";
      occurrenceId.textContent = item.occurrence_id;
      const gid = document.createElement("span");
      gid.className = "item-gid";
      gid.textContent = item.display_global_id || "No GID";
      top.append(occurrenceId, gid);

      const meta = document.createElement("span");
      meta.className = "item-meta";
      const source = document.createElement("span");
      source.textContent = `${item.clip_id || "Unknown clip"} · ${formatClock(item.start_time_sec)}`;
      const mark = document.createElement("span");
      mark.className = reviewed ? "item-review-mark" : "";
      mark.textContent = reviewed ? "Reviewed" : "Pending";
      meta.append(source, mark);
      button.append(top, meta);
      fragment.append(button);
      state.listElements.set(item.occurrence_id, button);
    }

    elements.occurrenceList.replaceChildren(fragment);
  }

  function syncListSelection(scroll = false) {
    for (const [id, button] of state.listElements) {
      const selected = id === state.selectedId;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-selected", String(selected));
      if (selected) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
    const selected = state.listElements.get(state.selectedId);
    if (scroll && selected) selected.scrollIntoView({ block: "nearest" });
  }

  function setActionDisabled(disabled) {
    elements.acceptButton.disabled = disabled;
    elements.invalidButton.disabled = disabled;
    elements.newIdInput.disabled = disabled;
    elements.updateButton.disabled = disabled;
  }

  function renderDetails(item) {
    elements.detailOccurrence.textContent = item.occurrence_id;
    elements.detailGid.textContent = item.display_global_id || "—";
    elements.detailClip.textContent = item.clip_id || "—";
    elements.detailTrack.textContent = item.legacy_track_id ?? "—";
    elements.detailFrames.textContent = `${item.start_frame ?? "—"} – ${item.end_frame ?? "—"}`;
    elements.detailDuration.textContent = Number.isFinite(Number(item.span_duration_sec))
      ? `${Number(item.span_duration_sec).toFixed(2)} s`
      : "—";

    const review = item.review;
    const reviewed = isReviewed(item);
    elements.decisionStatus.textContent = reviewed ? "Reviewed" : "Pending";
    elements.decisionStatus.className = `status-badge${reviewed ? " reviewed" : ""}`;
    elements.currentReview.className = `current-review${reviewed ? " reviewed" : ""}`;
    elements.currentReview.textContent = reviewed
      ? `Current decision: ${reviewLabel(review)}${review.reviewed_at_utc ? ` · ${review.reviewed_at_utc}` : ""}`
      : "This occurrence has not been reviewed.";

    elements.acceptButton.classList.toggle("current", review?.action === "accept");
    elements.invalidButton.classList.toggle("current", review?.action === "invalid_multiple_cows");
    elements.updateAction.classList.toggle("current", review?.action === "update_id");
    elements.newIdInput.value = review?.action === "update_id"
      ? String(Number(String(review.reviewed_global_id).replace(/^G/, "")))
      : "";
    setActionDisabled(state.submitting || !item.clip_ready);
  }

  function resetVideo(message, status = "Waiting", statusClass = "") {
    elements.video.pause();
    elements.video.removeAttribute("src");
    elements.video.load();
    elements.videoMessage.hidden = false;
    elements.videoMessage.textContent = message;
    elements.videoStatus.textContent = status;
    elements.videoStatus.className = `status-badge${statusClass ? ` ${statusClass}` : ""}`;
    elements.playPauseButton.disabled = true;
    elements.playPauseButton.textContent = "Play";
    elements.videoProgress.disabled = true;
    elements.videoProgress.max = "1";
    elements.videoProgress.value = "0";
    elements.timeDisplay.textContent = "0:00 / 0:00";
  }

  function loadVideo(item) {
    state.videoSelection += 1;
    const selection = state.videoSelection;
    const playbackStart = item.playback_start_sec;
    resetVideo("Loading cached clip…", "Loading");
    if (!item.clip_ready || !item.clip_url) {
      resetVideo("This cached clip is not ready yet. Reload the page after clip generation completes.", "Not ready");
      return;
    }

    const startPlayback = () => {
      if (selection !== state.videoSelection) return;
      const duration = Number.isFinite(elements.video.duration) ? elements.video.duration : 0;
      const target = Math.min(playbackStart, duration);
      elements.video.currentTime = target;
      elements.videoProgress.value = String(target);
      elements.timeDisplay.textContent = `${formatClock(target)} / ${formatClock(duration)}`;
      elements.video.play().catch(() => {
        elements.videoStatus.textContent = "Paused";
      });
    };
    elements.video.addEventListener("loadedmetadata", startPlayback, { once: true });
    elements.video.src = item.clip_url;
    elements.video.load();
  }

  function selectOccurrence(id, { scroll = false } = {}) {
    const item = state.occurrences.find((candidate) => candidate.occurrence_id === id);
    if (!item) return;
    state.selectedId = id;
    elements.videoHeading.textContent = `${item.occurrence_id} · ${item.display_global_id}`;
    elements.submitMessage.textContent = "";
    elements.submitMessage.className = "submit-message";
    syncListSelection(scroll);
    renderDetails(item);
    loadVideo(item);
  }

  function nextUnreviewedIndex(afterIndex = -1) {
    if (!state.occurrences.length) return -1;
    for (let offset = 1; offset <= state.occurrences.length; offset += 1) {
      const index = (afterIndex + offset) % state.occurrences.length;
      if (!isReviewed(state.occurrences[index])) return index;
    }
    return -1;
  }

  function applyState(payload, selection = "next") {
    state.payload = validateState(payload);
    state.occurrences = payload.occurrences;
    renderProgress();
    renderList();

    let index = -1;
    if (selection && typeof selection === "object" && selection.afterId) {
      const previousIndex = state.occurrences.findIndex((item) => item.occurrence_id === selection.afterId);
      index = nextUnreviewedIndex(previousIndex);
      if (index < 0) index = Math.max(0, previousIndex);
    } else {
      index = nextUnreviewedIndex(-1);
      if (index < 0 && state.occurrences.length) index = 0;
    }

    if (index >= 0) selectOccurrence(state.occurrences[index].occurrence_id, { scroll: true });
    else {
      state.selectedId = null;
      resetVideo("No occurrence clips are available.");
      setActionDisabled(true);
    }
    elements.shell.setAttribute("aria-busy", "false");
  }

  async function loadInitialState() {
    try {
      applyState(await requestJson("/api/state"));
    } catch (error) {
      elements.shell.setAttribute("aria-busy", "false");
      elements.occurrenceList.innerHTML = '<p class="empty-message">Unable to load occurrences.</p>';
      elements.fatalError.hidden = false;
      elements.fatalError.textContent = `Unable to load review state: ${error.message} Reload the page to try again.`;
    }
  }

  async function submitReview(action) {
    const item = selectedOccurrence();
    if (!item || state.submitting || !item.clip_ready) return;

    const body = { action };
    if (action === "update_id") {
      const raw = elements.newIdInput.value.trim();
      const newId = Number(raw);
      if (!/^\d{1,2}$/.test(raw) || !Number.isInteger(newId) || newId < 1 || newId > 62) {
        elements.newIdInput.setCustomValidity("Enter a whole number from 1 to 62.");
        elements.newIdInput.reportValidity();
        elements.newIdInput.focus();
        return;
      }
      elements.newIdInput.setCustomValidity("");
      body.new_id = newId;
    }

    state.submitting = true;
    setActionDisabled(true);
    elements.submitMessage.className = "submit-message";
    elements.submitMessage.textContent = "Saving review…";
    const submittedId = item.occurrence_id;
    let saved = false;

    try {
      await requestJson(`/api/reviews/${encodeURIComponent(submittedId)}`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      saved = true;
      const payload = await requestJson("/api/state");
      state.submitting = false;
      applyState(payload, { afterId: submittedId });
      elements.submitMessage.className = "submit-message success";
      elements.submitMessage.textContent = `Saved ${submittedId}.`;
    } catch (error) {
      state.submitting = false;
      setActionDisabled(false);
      elements.submitMessage.className = "submit-message error";
      elements.submitMessage.textContent = saved
        ? `The review was saved, but refreshed state could not be loaded: ${error.message}`
        : `Could not save review: ${error.message}`;
    }
  }

  function moveSelection(direction) {
    if (!state.occurrences.length) return;
    const current = state.occurrences.findIndex((item) => item.occurrence_id === state.selectedId);
    const next = Math.min(state.occurrences.length - 1, Math.max(0, current + direction));
    selectOccurrence(state.occurrences[next].occurrence_id, { scroll: true });
    state.listElements.get(state.selectedId)?.focus({ preventScroll: true });
  }

  function togglePlayback() {
    if (elements.playPauseButton.disabled) return;
    if (elements.video.paused) elements.video.play().catch(() => {});
    else elements.video.pause();
  }

  elements.occurrenceList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-occurrence-id]");
    if (button) selectOccurrence(button.dataset.occurrenceId);
  });
  elements.acceptButton.addEventListener("click", () => submitReview("accept"));
  elements.invalidButton.addEventListener("click", () => submitReview("invalid_multiple_cows"));
  elements.updateButton.addEventListener("click", () => submitReview("update_id"));
  elements.newIdInput.addEventListener("input", () => elements.newIdInput.setCustomValidity(""));
  elements.playPauseButton.addEventListener("click", togglePlayback);
  elements.video.addEventListener("click", togglePlayback);
  elements.video.addEventListener("loadedmetadata", () => {
    const duration = Number.isFinite(elements.video.duration) ? elements.video.duration : 0;
    elements.videoProgress.max = String(duration || 1);
    elements.videoProgress.disabled = !duration;
    elements.playPauseButton.disabled = !duration;
    elements.videoMessage.hidden = true;
    elements.videoStatus.textContent = elements.video.paused ? "Ready" : "Playing";
    elements.videoStatus.className = "status-badge ready";
    elements.timeDisplay.textContent = `${formatClock(0)} / ${formatClock(duration)}`;
  });
  elements.video.addEventListener("timeupdate", () => {
    if (document.activeElement !== elements.videoProgress) {
      elements.videoProgress.value = String(elements.video.currentTime || 0);
    }
    elements.timeDisplay.textContent = `${formatClock(elements.video.currentTime)} / ${formatClock(elements.video.duration)}`;
  });
  elements.video.addEventListener("play", () => {
    elements.playPauseButton.textContent = "Pause";
    elements.videoStatus.textContent = "Playing";
  });
  elements.video.addEventListener("pause", () => {
    elements.playPauseButton.textContent = "Play";
    if (elements.video.currentSrc) elements.videoStatus.textContent = "Paused";
  });
  elements.video.addEventListener("error", () => {
    elements.videoMessage.hidden = false;
    elements.videoMessage.textContent = "The cached clip could not be loaded.";
    elements.videoStatus.textContent = "Error";
    elements.videoStatus.className = "status-badge error";
    elements.playPauseButton.disabled = true;
    elements.videoProgress.disabled = true;
  });
  elements.videoProgress.addEventListener("input", () => {
    const target = Number(elements.videoProgress.value);
    if (Number.isFinite(target)) elements.video.currentTime = target;
  });

  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const tag = event.target.tagName;
    const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    if (typing) {
      if (event.target === elements.newIdInput && event.key === "Enter") submitReview("update_id");
      return;
    }
    if (event.key === " " && tag !== "BUTTON") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      moveSelection(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveSelection(-1);
    } else if (event.key.toLowerCase() === "a") {
      submitReview("accept");
    } else if (event.key.toLowerCase() === "i") {
      submitReview("invalid_multiple_cows");
    } else if (event.key.toLowerCase() === "u") {
      elements.newIdInput.focus();
      elements.newIdInput.select();
    }
  });

  loadInitialState();
})();
