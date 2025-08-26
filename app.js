const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
ctx.imageSmoothingEnabled = false;

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";

function clearCanvas() {
  ctx.save();
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}
clearCanvas();

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  if (e.touches && e.touches[0]) {
    return { x: e.touches[0].clientX - r.left, y: e.touches[0].clientY - r.top };
    }
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function getOriginalPngBase64() {
  return canvas.toDataURL("image/png");
}

const HOST = "http://localhost:8000";

// --------- Probability chart ----------
const container = document.getElementById("probability-container");
let bars = [];

function createBarsOnce() {
  if (bars.length) return;

  for (let i = 0; i < 10; i++) {
    const bar = document.createElement("div");
    bar.style.height = "0%";

    const label = document.createElement("span");
    label.textContent = String(i);

    bar.appendChild(label);
    container.appendChild(bar);
    bars.push(bar);
  }
}

function updateBars(probabilities) {
  const probs = (probabilities || []).slice(0, 10);
  while (probs.length < 10) probs.push(0);

  probs.forEach((p, i) => {
    const h = Math.max(0, Math.min(100, p * 100));
    bars[i].style.height = `${h.toFixed(2)}%`;
  });
}

createBarsOnce();
updateBars(Array(10).fill(0));

// --------- Pooling during drawing ----------
let drawing = false;
let pollTimer = null;
let inFlight = false;

function startPolling() {
  if (pollTimer !== null) return;
  pollTimer = setInterval(async () => {
    if (!drawing || inFlight) return;
    inFlight = true;
    try {
      const image_b64 = getOriginalPngBase64();
      const res = await fetch(`${HOST}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64 })
      });
      if (res.ok) {
        const out = await res.json();
        if (!out.error && Array.isArray(out.proba)) {
          updateBars(out.proba);
        }
      } else {
        console.error("Predict HTTP", res.status);
      }
    } catch (err) {
      console.error(err);
    } finally {
      inFlight = false;
    }
  }, 200);
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// --------- Mouse events ---------
canvas.addEventListener("mousedown", (e) => {
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  startPolling();
});

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  stopPolling();
});

canvas.addEventListener("mouseout", () => {
  drawing = false;
  stopPolling();
});

// --------- Touchscreen events ---------
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  startPolling();
}, { passive: false });

canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}, { passive: false });

canvas.addEventListener("touchend", () => {
  drawing = false;
  stopPolling();
}, { passive: true });

document.addEventListener("keydown", (e) => {
  if (e.key === "Backspace") {
    const t = e.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
    if (!typing) {
      e.preventDefault();
      clearCanvas();
      updateBars(Array(10).fill(0));
    }
  }
});
