// Handcrafted feature extraction — JS port of extract_hc() in pnt_cli.py / train_final_models.py.
// Faithfully reproduces the ORIGINAL Python regex behavior, including that the base word list
// only matches already-lowercase words (`\b[a-z]+\b`, no case-insensitive flag) — this is how
// the deployed model was trained, so it must match exactly for predictions to line up.

export const FEATURE_ORDER = [
  "word_count", "mattr", "root_ttr", "brunet_w", "honore_r", "hapax_ratio", "type_count", "hapax_count",
  "um_count", "filled_pause_rate", "sent_count", "avg_sent_len", "repetition_rate", "revision_rate",
  "mean_word_freq", "min_word_freq", "median_word_freq", "low_freq_ratio", "freq_sd",
  "flesch_ease", "flesch_grade", "gunning_fog", "smog_index",
  "noun_rate", "verb_rate", "adj_rate", "adv_rate", "pron_rate", "det_rate", "prep_rate", "conj_rate", "num_rate",
];

function countMatches(text, re) {
  const m = text.match(re);
  return m ? m.length : 0;
}

function mattr(words, win = 50) {
  if (words.length < win) {
    return new Set(words).size / Math.max(words.length, 1);
  }
  let sum = 0;
  let count = 0;
  for (let i = 0; i <= words.length - win; i++) {
    sum += new Set(words.slice(i, i + win)).size / win;
    count++;
  }
  return sum / count;
}

// Syllable-count heuristic (vowel-group based), used only for the four readability
// features (flesch_ease, flesch_grade, gunning_fog, smog_index). This approximates
// textstat's pyphen-based hyphenation and will not match it exactly on every word,
// but affects only 4 of 416 total model inputs.
function countSyllables(word) {
  word = word.toLowerCase().replace(/[^a-z]/g, "");
  if (!word) return 0;
  if (word.length <= 3) return 1;
  word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, "");
  word = word.replace(/^y/, "");
  const matches = word.match(/[aeiouy]{1,2}/g);
  return matches ? Math.max(matches.length, 1) : 1;
}

function readability(text, words, sentCount) {
  const wordCount = words.length;
  const syllables = words.reduce((s, w) => s + countSyllables(w), 0);
  const complexWords = words.filter((w) => countSyllables(w) >= 3).length;
  const S = Math.max(sentCount, 1);
  const W = Math.max(wordCount, 1);

  const fleschEase = 206.835 - 1.015 * (W / S) - 84.6 * (syllables / W);
  const fleschGrade = 0.39 * (W / S) + 11.8 * (syllables / W) - 15.59;
  const gunningFog = 0.4 * (W / S + 100 * (complexWords / W));
  const smogIndex = wordCount > 0
    ? 1.043 * Math.sqrt(complexWords * (30 / S)) + 3.1291
    : 0;

  return {
    flesch_ease: fleschEase,
    flesch_grade: fleschGrade,
    gunning_fog: gunningFog,
    smog_index: smogIndex,
  };
}

// wordFreqMap: Map<string, number> loaded from models/wordfreq_en.json.
// Words not in the top-60k list fall back to a small default frequency.
export function extractHC(text, wordFreqMap) {
  const lowerWordMatches = text.match(/\b[a-z]+\b/g) || [];
  const words = lowerWordMatches.map((w) => w.toLowerCase());
  const n = words.length;
  const vSet = new Set(words);
  const v = vSet.size;

  const wordCounts = {};
  for (const w of words) wordCounts[w] = (wordCounts[w] || 0) + 1;
  let hapax = 0;
  for (const w of vSet) if (wordCounts[w] === 1) hapax++;

  const f = {};
  f.word_count = n;
  f.mattr = mattr(words, 50);
  f.root_ttr = v / Math.sqrt(Math.max(n, 1));
  f.brunet_w = v > 0 ? Math.pow(n, Math.pow(v, -0.165)) : 0.0;
  f.honore_r = (v > hapax && n > 1) ? (100 * Math.log(n)) / (1 - hapax / Math.max(v, 1)) : 0.0;
  f.hapax_ratio = hapax / Math.max(v, 1);
  f.type_count = v;
  f.hapax_count = hapax;

  f.um_count = countMatches(text, /\bum\b|\buh\b|\ber\b|\behm\b/gi);
  f.filled_pause_rate = f.um_count / Math.max(n, 1);

  const sents = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
  f.sent_count = sents.length;
  f.avg_sent_len = n / Math.max(sents.length, 1);

  let reps = 0;
  for (let i = 1; i < words.length; i++) if (words[i] === words[i - 1]) reps++;
  f.repetition_rate = reps / Math.max(n, 1);

  f.revision_rate = countMatches(text, /\b(i mean|no wait|actually|sorry)\b/gi) / Math.max(n, 1);

  const freqs = words
    .filter((w) => /^[a-z]+$/.test(w))
    .map((w) => (wordFreqMap.has(w) ? wordFreqMap.get(w) : 0.0));
  if (freqs.length) {
    const sorted = [...freqs].sort((a, b) => a - b);
    const mean = freqs.reduce((a, b) => a + b, 0) / freqs.length;
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    const variance = freqs.reduce((s, x) => s + (x - mean) ** 2, 0) / freqs.length;
    f.mean_word_freq = mean;
    f.min_word_freq = sorted[0];
    f.median_word_freq = median;
    f.low_freq_ratio = freqs.filter((x) => x < 1e-5).length / freqs.length;
    f.freq_sd = Math.sqrt(variance);
  } else {
    f.mean_word_freq = 0.0;
    f.min_word_freq = 0.0;
    f.median_word_freq = 0.0;
    f.low_freq_ratio = 0.0;
    f.freq_sd = 0.0;
  }

  const rd = readability(text, words, sents.length);
  f.flesch_ease = rd.flesch_ease;
  f.flesch_grade = rd.flesch_grade;
  f.gunning_fog = rd.gunning_fog;
  f.smog_index = rd.smog_index;

  const rate = (re) => countMatches(text, re) / Math.max(n, 1);
  f.noun_rate = rate(/\b(the|a|an)\s+\w+/gi);
  f.verb_rate = rate(/\b(is|are|was|were|has|have|had|will|would|can|could|should|do|does|did|go|goes|went|see|saw|take|took|make|made|get|got|come|came)\b/gi);
  f.adj_rate = rate(/\b(big|small|little|old|young|good|bad|long|short|great|high|low|new|first|last|own|right|next|early|hard|easy|large|same|other|true|whole)\b/gi);
  f.adv_rate = rate(/\b(very|really|just|also|well|back|even|still|way|already|now|too|only|then|here|there|always|often|never|sometimes)\b/gi);
  f.pron_rate = rate(/\b(i|he|she|it|we|they|me|him|her|us|them|my|his|its|our|their|you|your)\b/gi);
  f.det_rate = rate(/\b(the|a|an|this|that|these|those)\b/gi);
  f.prep_rate = rate(/\b(in|on|at|to|for|of|with|from|by|about|as|into|through|during|before|after|above|below|between|out|up|down)\b/gi);
  f.conj_rate = rate(/\b(and|but|or|nor|so|yet|for|although|because|since|while|if|when|where|which|who|that)\b/gi);
  f.num_rate = rate(/\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b/gi);

  return FEATURE_ORDER.map((k) => f[k]);
}
