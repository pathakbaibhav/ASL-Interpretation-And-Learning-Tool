// ASL Interpreter frontend.
//
// Two engines:
//  - "browser": MediaPipe runs in the page (WebAssembly/GPU) and only the 21
//    hand landmarks are POSTed to /api/predict. Lowest latency, and video
//    never leaves the device.
//  - "server": JPEG frames are POSTed to /api/predict/frame and the server
//    runs MediaPipe. Used when the browser can't load MediaPipe.

const TASKS_VERSION = "0.10.14";
const TASKS_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const GAUGE_LENGTH = 157.08; // Length of the gauge arc (pi * r, r = 50).
const MASTERY_GOAL = 3; // Successful holds before a letter counts as mastered.
const SERVER_FRAME_WIDTH = 480;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), stageEmpty: $("stageEmpty"), startBtn: $("startBtn"),
  cameraError: $("cameraError"), fps: $("fpsBadge"), rtt: $("rttBadge"), light: $("lightBadge"),
  engine: $("engineSelect"), backend: $("backendSelect"), holdRange: $("holdRange"), holdOut: $("holdOut"),
  confRange: $("confRange"), confOut: $("confOut"), letter: $("letter"), gaugeFill: $("gaugeFill"),
  confValue: $("confValue"), holdFill: $("holdFill"), holdHint: $("holdHint"), alts: $("alts"),
  sentence: $("sentence"), status: $("status"), targetLetter: $("targetLetter"), refImage: $("refImage"),
  practiceFill: $("practiceFill"), practiceHint: $("practiceHint"), streak: $("streak"), best: $("best"),
  mastered: $("mastered"), letterGrid: $("letterGrid"), randomToggle: $("randomToggle"),
};
const overlayCtx = els.overlay.getContext("2d");

// ---------------------------------------------------------------- storage --
// Per-viewer conveniences only; everything works if storage is unavailable.
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

const settings = load("asl.settings", { engine: "browser", backend: "ensemble", holdMs: 1500, minConf: 0.8 });
const progress = load("asl.progress", { best: 0, successes: {} });

// ------------------------------------------------------------------ state --
const state = {
  mode: "translate",
  running: false,
  inFlight: false,
  landmarker: null,
  lastVideoTime: -1,
  text: "",
  hold: { label: null, start: 0 },
  target: "A",
  streak: 0,
  pausedUntil: 0,
  frames: 0,
  fpsTimer: performance.now(),
};
const letters = [...els.letterGrid.querySelectorAll(".grid-letter")].map((b) => b.dataset.letter);

// --------------------------------------------------------------- settings --
function applySettings() {
  els.engine.value = settings.engine;
  els.backend.value = settings.backend;
  els.holdRange.value = settings.holdMs / 1000;
  els.confRange.value = settings.minConf;
  els.holdOut.textContent = `${(settings.holdMs / 1000).toFixed(1)}s`;
  els.confOut.textContent = `${Math.round(settings.minConf * 100)}%`;
}

els.holdRange.addEventListener("input", () => {
  settings.holdMs = Number(els.holdRange.value) * 1000;
  applySettings(); save("asl.settings", settings);
});
els.confRange.addEventListener("input", () => {
  settings.minConf = Number(els.confRange.value);
  applySettings(); save("asl.settings", settings);
});
els.backend.addEventListener("change", () => {
  settings.backend = els.backend.value; save("asl.settings", settings);
});
els.engine.addEventListener("change", async () => {
  settings.engine = els.engine.value; save("asl.settings", settings);
  if (settings.engine === "browser" && !state.landmarker) await loadLandmarker();
});

// -------------------------------------------------------------- mediapipe --
async function loadLandmarker() {
  setStatus("Loading MediaPipe in the browser…");
  try {
    const { FilesetResolver, HandLandmarker } = await import(`${TASKS_URL}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${TASKS_URL}/wasm`);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: "/models/hand_landmarker.task", delegate },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      state.landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU"));
    } catch {
      state.landmarker = await HandLandmarker.createFromOptions(fileset, options("CPU"));
    }
    await refreshStatus();
  } catch (err) {
    console.warn("Browser MediaPipe unavailable, using server frames", err);
    settings.engine = "server";
    applySettings();
    setStatus("Browser MediaPipe unavailable, sending frames to the server instead");
  }
}

// ----------------------------------------------------------------- camera --
async function startCamera() {
  els.cameraError.hidden = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    els.overlay.width = els.video.videoWidth;
    els.overlay.height = els.video.videoHeight;
    els.stageEmpty.hidden = true;
    if (settings.engine === "browser" && !state.landmarker) await loadLandmarker();
    state.running = true;
    requestAnimationFrame(loop);
  } catch (err) {
    els.cameraError.textContent = `Couldn't open the camera: ${err.message || err.name}`;
    els.cameraError.hidden = false;
  }
}
els.startBtn.addEventListener("click", startCamera);

// ------------------------------------------------------------------- loop --
function loop() {
  if (!state.running) return;
  requestAnimationFrame(loop);
  tickFps();

  if (settings.engine === "browser" && state.landmarker) {
    if (els.video.currentTime === state.lastVideoTime) return;
    state.lastVideoTime = els.video.currentTime;
    const result = state.landmarker.detectForVideo(els.video, performance.now());
    state.frames++;
    const hand = result.landmarks?.[0];
    const points = hand ? hand.map((p) => [p.x, p.y]) : null;
    drawHand(points);
    if (!points) return onPrediction(null);
    if (!state.inFlight) sendLandmarks(points);
  } else if (!state.inFlight) {
    sendFrame();
  }
}

async function postJson(url, body, headers = {}) {
  const started = performance.now();
  const response = await fetch(url, { method: "POST", body, headers });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  els.rtt.textContent = `${Math.round(performance.now() - started)} / ${data.latency_ms.toFixed(1)} ms`;
  return data;
}

async function sendLandmarks(points) {
  state.inFlight = true;
  try {
    const data = await postJson(
      "/api/predict",
      JSON.stringify({ landmarks: points, backend: settings.backend }),
      { "Content-Type": "application/json" },
    );
    onPrediction(data);
  } catch (err) {
    setStatus(`Prediction failed: ${err.message}`);
  } finally {
    state.inFlight = false;
  }
}

const frameCanvas = document.createElement("canvas");
async function sendFrame() {
  const { videoWidth: w, videoHeight: h } = els.video;
  if (!w) return;
  state.inFlight = true;
  try {
    frameCanvas.width = SERVER_FRAME_WIDTH;
    frameCanvas.height = Math.round((h / w) * SERVER_FRAME_WIDTH);
    frameCanvas.getContext("2d").drawImage(els.video, 0, 0, frameCanvas.width, frameCanvas.height);
    const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", 0.75));
    const data = await postJson(
      `/api/predict/frame?backend=${encodeURIComponent(settings.backend)}`, blob,
      { "Content-Type": "image/jpeg" },
    );
    state.frames++;
    els.light.hidden = !data.lighting_corrected;
    drawHand(data.hand_detected ? data.landmarks : null);
    onPrediction(data.hand_detected ? data : null);
  } catch (err) {
    setStatus(`Frame upload failed: ${err.message}`);
    await new Promise((r) => setTimeout(r, 500));
  } finally {
    state.inFlight = false;
  }
}

function tickFps() {
  const now = performance.now();
  if (now - state.fpsTimer >= 1000) {
    els.fps.textContent = `${Math.round((state.frames * 1000) / (now - state.fpsTimer))} fps`;
    state.frames = 0;
    state.fpsTimer = now;
  }
}

function drawHand(points) {
  const { width: w, height: h } = els.overlay;
  overlayCtx.clearRect(0, 0, w, h);
  if (!points) return;
  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = "rgba(91, 140, 255, 0.9)";
  for (const [a, b] of HAND_CONNECTIONS) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(points[a][0] * w, points[a][1] * h);
    overlayCtx.lineTo(points[b][0] * w, points[b][1] * h);
    overlayCtx.stroke();
  }
  overlayCtx.fillStyle = "#e8ecf4";
  for (const [x, y] of points) {
    overlayCtx.beginPath();
    overlayCtx.arc(x * w, y * h, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

// ------------------------------------------------------------ predictions --
function onPrediction(result) {
  const label = result?.label ?? null;
  const confidence = result?.confidence ?? 0;
  const confident = label && confidence >= settings.minConf;

  els.letter.textContent = label ? displayLabel(label) : "–";
  els.letter.className = `letter ${confident ? "good" : "low"}`;
  els.confValue.textContent = Math.round(confidence * 100);
  els.gaugeFill.style.strokeDashoffset = GAUGE_LENGTH * (1 - confidence);
  els.gaugeFill.style.stroke = confident ? "var(--good)" : "var(--bad)";
  els.alts.innerHTML = (result?.top_k ?? [])
    .map((t) => `<li>${displayLabel(t.label)} ${Math.round(t.confidence * 100)}%</li>`)
    .join("");

  if (state.mode === "translate") {
    const p = updateHold(confident ? label : null, (held) => addText(held === "space" ? " " : held));
    els.holdFill.style.width = `${p * 100}%`;
    els.holdHint.textContent = !label ? "Show your hand to the camera"
      : confident ? `Hold “${displayLabel(label)}” to add it` : "Not confident yet, adjust your hand";
  } else {
    practiceStep(label, confident);
  }
}

// Returns hold progress 0..1 and calls onCommit when a sign is held long enough.
function updateHold(label, onCommit) {
  const now = performance.now();
  if (!label || now < state.pausedUntil) {
    state.hold.label = null;
    return 0;
  }
  if (label !== state.hold.label) state.hold = { label, start: now };
  const p = Math.min(1, (now - state.hold.start) / settings.holdMs);
  if (p >= 1) {
    onCommit(label);
    state.hold.start = now; // Holding longer repeats the letter (e.g. "LL").
  }
  return p;
}

function displayLabel(label) {
  return label === "space" ? "␣" : label;
}

// -------------------------------------------------------------- translate --
function renderText() {
  els.sentence.textContent = state.text;
  els.sentence.insertAdjacentHTML("beforeend", '<span class="caret"></span>');
}
function addText(s) { state.text += s; renderText(); }

$("spaceBtn").addEventListener("click", () => addText(" "));
$("backBtn").addEventListener("click", () => { state.text = state.text.slice(0, -1); renderText(); });
$("clearBtn").addEventListener("click", () => { state.text = ""; renderText(); });
$("copyBtn").addEventListener("click", () => navigator.clipboard?.writeText(state.text));
$("speakBtn").addEventListener("click", () => {
  if (!state.text.trim() || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(state.text.toLowerCase()));
});

document.addEventListener("keydown", (e) => {
  if (state.mode !== "translate" || e.target.closest("select, input, button")) return;
  if (e.key === "Backspace") { state.text = state.text.slice(0, -1); renderText(); e.preventDefault(); }
  if (e.key === " ") { addText(" "); e.preventDefault(); }
});

// --------------------------------------------------------------- practice --
function setTarget(letter) {
  state.target = letter;
  els.targetLetter.textContent = letter;
  els.targetLetter.classList.remove("success");
  els.refImage.src = `/signs/${letter}.png`;
  els.refImage.alt = `Reference handshape for ${letter}`;
  renderGrid();
}

function nextTarget() {
  if (els.randomToggle.checked) {
    // Prefer letters that aren't mastered yet.
    const pool = letters.filter((l) => l !== state.target && (progress.successes[l] ?? 0) < MASTERY_GOAL);
    const choices = pool.length ? pool : letters.filter((l) => l !== state.target);
    setTarget(choices[Math.floor(Math.random() * choices.length)]);
  } else {
    setTarget(letters[(letters.indexOf(state.target) + 1) % letters.length]);
  }
}

function practiceStep(label, confident) {
  const matches = confident && label === state.target;
  const p = updateHold(matches ? label : null, () => {
    state.streak++;
    progress.successes[state.target] = (progress.successes[state.target] ?? 0) + 1;
    progress.best = Math.max(progress.best, state.streak);
    save("asl.progress", progress);
    els.targetLetter.classList.add("success");
    els.practiceHint.textContent = "Nice! Next letter…";
    state.pausedUntil = performance.now() + 900;
    renderStats();
    setTimeout(nextTarget, 900);
  });
  els.practiceFill.style.width = `${p * 100}%`;
  if (performance.now() >= state.pausedUntil) {
    els.practiceHint.textContent = !label ? "Show the handshape to the camera"
      : matches ? "Great, hold it…"
      : `Seeing “${displayLabel(label)}”. Compare with the reference`;
  }
}

function renderStats() {
  els.streak.textContent = state.streak;
  els.best.textContent = progress.best;
  els.mastered.textContent = letters.filter((l) => (progress.successes[l] ?? 0) >= MASTERY_GOAL).length;
  renderGrid();
}

function renderGrid() {
  for (const button of els.letterGrid.children) {
    const l = button.dataset.letter;
    button.classList.toggle("current", l === state.target);
    button.classList.toggle("mastered", (progress.successes[l] ?? 0) >= MASTERY_GOAL);
  }
}

els.letterGrid.addEventListener("click", (e) => {
  const button = e.target.closest(".grid-letter");
  if (button) setTarget(button.dataset.letter);
});
$("skipBtn").addEventListener("click", () => { state.streak = 0; renderStats(); nextTarget(); });

// ------------------------------------------------------------------- tabs --
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    state.mode = tab.dataset.mode;
    state.hold.label = null;
    for (const t of document.querySelectorAll(".tab")) {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    }
    for (const panel of document.querySelectorAll("[data-panel]")) {
      panel.hidden = panel.dataset.panel !== state.mode;
    }
  });
}

// ----------------------------------------------------------------- status --
function setStatus(text) { els.status.textContent = text; }

async function refreshStatus() {
  try {
    const health = await (await fetch("/api/health")).json();
    const engine = settings.engine === "browser" && state.landmarker ? "browser MediaPipe" : "server MediaPipe";
    setStatus(`API ok · ${health.labels.length} signs · ${engine}`);
  } catch {
    setStatus("API unreachable");
  }
}

applySettings();
renderText();
setTarget(state.target);
renderStats();
refreshStatus();
