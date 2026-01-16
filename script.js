// --- iGaze: WebGazer dwell -> pending selection -> yes/no/stop -> speak ---
// Requires: webgazer.js loaded + index.html defines addWord(), speakCurrent(), undo(), clearAll()

const DWELL_MS = 900;
const COOLDOWN_MS = 1100;

let lastGaze = null;
let hoverEl = null;
let hoverSince = 0;
let lastFireAt = 0;

let pendingWord = null;
let pendingEl = null;

function nowMs(){ return performance.now(); }

function setHover(el){
  if (hoverEl && hoverEl !== el) hoverEl.classList.remove("gaze-hover");
  hoverEl = el;
  if (hoverEl) hoverEl.classList.add("gaze-hover");
}

function setPending(el, word){
  if (pendingEl) pendingEl.classList.remove("gaze-pending");
  pendingEl = el;
  pendingWord = word || null;
  if (pendingEl) pendingEl.classList.add("gaze-pending");
}

function stopAudio(){
  const audio = document.getElementById("player");
  if (audio){
    audio.pause();
    audio.currentTime = 0;
  }
  window.speechSynthesis?.cancel?.();
}

function handleSelection(el, api){
  const t = nowMs();
  if (t - lastFireAt < COOLDOWN_MS) return;
  lastFireAt = t;

  // SPEAK / UNDO / CLEAR buttons
  if (el.id === "btnSpeak" || el.id === "btnSpeakHero") return api.speakCurrent();
  if (el.id === "btnUndo") return api.undo();
  if (el.id === "btnClear") return api.clearAll();

  // Tiles
  const tile = el.closest(".tile");
  if (!tile) return;

  const w = tile.dataset.word;

  // Special behavior for YES/NO/STOP tiles when there is a pending word
  if (w === "yes"){
    if (pendingWord){
      api.addWord(pendingWord);
      setPending(null, null);
      return;
    }
    // No pending -> treat as a normal word
    api.addWord("yes");
    return;
  }

  if (w === "no"){
    if (pendingWord){
      // Cancel the pending selection
      setPending(null, null);
      return;
    }
    api.addWord("no");
    return;
  }

  if (w === "stop"){
    // STOP always stops audio + cancels pending
    stopAudio();
    setPending(null, null);
    return;
  }

  // Any other word tile: stage as pending (requires YES to commit)
  setPending(tile, w);
}

function gazeLoop(api){
  if (!lastGaze) return requestAnimationFrame(() => gazeLoop(api));

  const x = Math.max(0, Math.min(window.innerWidth - 1, lastGaze.x));
  const y = Math.max(0, Math.min(window.innerHeight - 1, lastGaze.y));

  const el = document.elementFromPoint(x, y);

  // We allow gaze on tiles + control buttons
  const target =
    el?.closest(".tile") ||
    el?.closest("#btnSpeak,#btnSpeakHero,#btnUndo,#btnClear");

  if (!target){
    setHover(null);
    hoverSince = 0;
    return requestAnimationFrame(() => gazeLoop(api));
  }

  if (target !== hoverEl){
    setHover(target);
    hoverSince = nowMs();
  } else {
    if (nowMs() - hoverSince >= DWELL_MS){
      handleSelection(target, api);
      hoverSince = nowMs() + 999999; // prevent immediate refire
    }
  }

  requestAnimationFrame(() => gazeLoop(api));
}

async function startGaze(api){
  const calStatus = document.getElementById("calStatus");
  if (calStatus) calStatus.textContent = "Calibration status: Requesting camera permission…";

  if (!window.webgazer){
    if (calStatus) calStatus.textContent = "Calibration status: WebGazer not loaded.";
    return;
  }

  await webgazer.setGazeListener((data) => {
    if (!data) return;
    lastGaze = { x: data.x, y: data.y };
  }).begin();

  // Hide overlays (optional)
  webgazer.showVideo(false);
  webgazer.showFaceOverlay(false);
  webgazer.showFaceFeedbackBox(false);

  if (calStatus) calStatus.textContent = "Calibration status: Camera enabled. Look + click around to calibrate…";

  requestAnimationFrame(() => gazeLoop(api));
}

function initGaze(){
  const api = window.iGazeApi;
  if (!api) return;
  startGaze(api).catch(err => {
    console.error(err);
    const calStatus = document.getElementById("calStatus");
    if (calStatus) calStatus.textContent = "Calibration status: Camera failed. Check browser permissions.";
  });
}

window.addEventListener("igaze-ready", initGaze);
window.addEventListener("DOMContentLoaded", initGaze);