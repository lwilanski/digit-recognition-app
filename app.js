const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
ctx.imageSmoothingEnabled = false;

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";

function getOriginalPngBase64() {
  return canvas.toDataURL("image/png");
}

let drawing = false;

canvas.addEventListener("mousedown", (e) => {
  drawing = true;
  const r = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - r.left, e.clientY - r.top);
});

canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseout", () => drawing = false);

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - r.left, e.clientY - r.top);
  ctx.stroke();
});

document.getElementById("clearBtn").onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

const HOST = "http://localhost:8000";

function displayProbabilities(probabilities) {
  const container = document.getElementById("probability-container");
  container.innerHTML = ''; // Wyczyść poprzednie słupki

  probabilities.forEach((prob, idx) => {
    const bar = document.createElement("div");

    // Ustawiamy wysokość słupka na podstawie prawdopodobieństwa
    bar.style.height = `${prob * 100}%`;  // Wysokość odpowiadająca procentowi

    // Dodajemy etykietę (numer klasy)
    const label = document.createElement("span");
    label.innerText = `${idx}`;

    // Dodajemy etykietę do słupka
    bar.appendChild(label);

    // Dodajemy słupek do kontenera
    container.appendChild(bar);
  });
}

// const labelInput = document.getElementById("labelInput");
// const classifyBtn = document.getElementById("classifyBtn");

// function getValidLabel() {
//   const v = labelInput.value.trim();
//   if (!/^\d$/.test(v)) return null;
//   const n = parseInt(v, 10);
//   return (n >= 0 && n <= 9) ? n : null;
// }

// labelInput.addEventListener("keydown", (e) => {
//   if (e.key === "Enter") classifyBtn.click();
// });

// function updateBtnState() {
//   classifyBtn.disabled = (getValidLabel() === null);
// }
// labelInput.addEventListener("input", updateBtnState);
// updateBtnState();

// document.getElementById("classifyBtn").onclick = async () => {
//   const label = getValidLabel();
//   if (label === null) {
//     alert("Podaj etykietę 0-9.");
//     return;
//   }

//   const image_b64 = getOriginalPngBase64();

//   try {
//     const res = await fetch(`${HOST}/predict`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ image_b64, label })
//     });

//     if (!res.ok) {
//       const msg = await res.text();
//       throw new Error(`HTTP ${res.status}: ${msg}`);
//     }

//     const out = await res.json();
//     if (out.error) throw new Error(out.error);

//     document.getElementById("probability-container").textContent =
//       `Prediction: ${out.prediction}`;
//     displayProbabilities(out.proba);

//   } catch (err) {
//     console.error(err);
//     alert("Błąd podczas klasyfikacji/zapisu: " + err.message);
//   }
// };

document.getElementById("classifyBtn").onclick = async () => {
  const image_b64 = getOriginalPngBase64();
  const res = await fetch(`${HOST}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_b64 })
  });
  const out = await res.json();

  if (out.error) {
    console.error(out.error);
    return;
  }

  document.getElementById("probability-container").textContent = `Prediction: ${out.prediction}`;

  displayProbabilities(out.proba);
};
