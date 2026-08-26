import { extractHC } from "./features.js";
import { AutoTokenizer, AutoModel, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// All inference (feature extraction, embedding, and the trained regressors) runs
// entirely in this page, in your browser. The transcript you paste is never sent
// anywhere — only static model files are fetched (from this site + the model CDN
// used to load the embedding model on first run).
env.allowLocalModels = false;

// The Python training pipeline used sentence-transformers, whose all-MiniLM-L6-v2
// config truncates to 256 tokens. transformers.js's default pipeline() wrapper does
// NOT apply that truncation, which silently produces very different embeddings for
// long transcripts — so we call the tokenizer/model directly and truncate explicitly
// to match.
const MAX_TOKENS = 256;

const TARGETS = ["correct", "semantic", "unrelated", "phonemic", "nonword", "mixed", "neologism", "nr"];
const LABELS = {
  correct: "Correct Responses",
  semantic: "Semantic Errors",
  unrelated: "Unrelated Errors",
  phonemic: "Phonemic Errors",
  nonword: "Nonword Errors",
  mixed: "Mixed Errors",
  neologism: "Neologisms",
  nr: "No Response",
};

const els = {
  status: document.getElementById("status"),
  text: document.getElementById("text"),
  wordCount: document.getElementById("wordCount"),
  predictBtn: document.getElementById("predictBtn"),
  clearBtn: document.getElementById("clearBtn"),
  results: document.getElementById("results"),
  condition: () => document.querySelector('input[name="condition"]:checked').value,
};

let manifest = null;
let wordFreqMap = null;
let tokenizer = null;
let embedModel = null;
let onnxSessions = {};
let ready = false;

async function embed(text) {
  const inputs = await tokenizer(text, { truncation: true, max_length: MAX_TOKENS, padding: true });
  const output = await embedModel(inputs);
  const hidden = output.last_hidden_state; // [1, seqLen, hSize]
  const mask = inputs.attention_mask; // [1, seqLen]
  const [, seqLen, hSize] = hidden.dims;
  const hdata = hidden.data;
  const mdata = mask.data;

  const sums = new Float64Array(hSize);
  let maskSum = 0;
  for (let i = 0; i < seqLen; i++) {
    const m = Number(mdata[i]);
    maskSum += m;
    for (let j = 0; j < hSize; j++) sums[j] += hdata[i * hSize + j] * m;
  }
  let norm = 0;
  const pooled = new Float64Array(hSize);
  for (let j = 0; j < hSize; j++) {
    pooled[j] = sums[j] / Math.max(maskSum, 1e-9);
    norm += pooled[j] * pooled[j];
  }
  norm = Math.sqrt(norm);
  return Array.from(pooled, (v) => v / norm);
}

function setStatus(msg, cls = "") {
  els.status.textContent = msg;
  els.status.className = cls;
}

async function loadAll() {
  setStatus("Loading word-frequency data…");
  const [manifestResp, wfResp] = await Promise.all([
    fetch("models/manifest.json"),
    fetch("models/wordfreq_en.json"),
  ]);
  manifest = await manifestResp.json();
  const wfObj = await wfResp.json();
  wordFreqMap = new Map(Object.entries(wfObj));

  setStatus("Loading sentence embedding model (first load only, then cached)…");
  tokenizer = await AutoTokenizer.from_pretrained("Xenova/all-MiniLM-L6-v2");
  embedModel = await AutoModel.from_pretrained("Xenova/all-MiniLM-L6-v2");

  setStatus("Loading prediction models…");
  const ort = window.ort;
  ort.env.wasm.numThreads = 1;
  for (const cond of Object.keys(manifest)) {
    for (const [target, info] of Object.entries(manifest[cond].targets)) {
      if (info.type === "onnx") {
        const key = `${cond}|${target}`;
        onnxSessions[key] = await ort.InferenceSession.create(`models/${info.file}`);
      }
    }
  }

  ready = true;
  setStatus("Ready — paste a transcript and click Predict.", "ready");
  els.predictBtn.disabled = false;
}

function imputeAndScale(x, cond) {
  const m = manifest[cond];
  const out = new Float32Array(m.n_features);
  for (let i = 0; i < m.n_features; i++) {
    let v = x[i];
    if (v === null || v === undefined || Number.isNaN(v)) v = m.imputer_statistics[i];
    out[i] = (v - m.scaler_mean[i]) / m.scaler_scale[i];
  }
  return out;
}

async function predictOne(cond, target, scaledX) {
  const info = manifest[cond].targets[target];
  if (info.type === "linear") {
    let s = info.intercept;
    for (let i = 0; i < scaledX.length; i++) s += scaledX[i] * info.coef[i];
    return s;
  }
  const session = onnxSessions[`${cond}|${target}`];
  const tensor = new window.ort.Tensor("float32", scaledX, [1, scaledX.length]);
  const out = await session.run({ input: tensor });
  const outputName = Object.keys(out)[0];
  return out[outputName].data[0];
}

async function predict(text, cond) {
  const hc = extractHC(text, wordFreqMap);
  const emb = await embed(text);
  const x = hc.concat(emb);
  const scaledX = imputeAndScale(x, cond);

  const raw = {};
  for (const t of TARGETS) {
    if (!manifest[cond].targets[t]) continue;
    const v = await predictOne(cond, t, scaledX);
    raw[t] = Math.max(0, v);
  }
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  if (total > 0) {
    for (const t of Object.keys(raw)) raw[t] = (raw[t] * 175.0) / total;
  }
  return raw;
}

function renderResults(result) {
  els.results.innerHTML = "";
  for (const t of TARGETS) {
    const row = document.createElement("div");
    row.className = "resultRow";
    const label = document.createElement("span");
    label.textContent = LABELS[t];
    const value = document.createElement("span");
    value.className = "resultValue";
    value.textContent = t in result ? result[t].toFixed(1) : "—";
    row.appendChild(label);
    row.appendChild(value);
    els.results.appendChild(row);
  }
}

els.text.addEventListener("input", () => {
  const words = els.text.value.trim().split(/\s+/).filter(Boolean);
  els.wordCount.textContent = words.length ? `${words.length} words` : "";
});

els.clearBtn.addEventListener("click", () => {
  els.text.value = "";
  els.wordCount.textContent = "";
  els.results.innerHTML = "";
});

els.predictBtn.addEventListener("click", async () => {
  if (!ready) return;
  const text = els.text.value.trim();
  const wc = text.split(/\s+/).filter(Boolean).length;
  if (!text) {
    alert("Please paste a discourse transcript first.");
    return;
  }
  if (wc < 10) {
    alert(`Only ${wc} words detected. Please provide a longer transcript.`);
    return;
  }
  els.predictBtn.disabled = true;
  els.predictBtn.textContent = "Working…";
  setStatus("Running prediction…");
  try {
    const result = await predict(text, els.condition());
    renderResults(result);
    setStatus(`Done — ${els.condition()} model, ${wc} words`, "ready");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    els.predictBtn.disabled = false;
    els.predictBtn.textContent = "Predict PNT Scores";
  }
});

loadAll().catch((err) => {
  console.error(err);
  setStatus(`Failed to load models: ${err.message}`, "error");
});
