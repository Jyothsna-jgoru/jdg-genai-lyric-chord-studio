"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { analysis: null, result: null, projectId: null, manualEdits: [], arrangementStale: false, generationTimer: null, audio: null, sources: [], timer: null, events: [], startTime: 0, paused: false, playbackToken: 0 };

function setStatus(message, error = false) {
  const node = $("#status"); node.textContent = message; node.setAttribute("data-error", String(error));
}

function markArrangementStale() {
  if (!state.result) return;
  state.arrangementStale = true; stopPlayback();
  $("#chord-sheet").setAttribute("data-stale", "true");
  setStatus("Lyrics or musical controls changed. Generate the complete song again before playback.");
}

function markArrangementCurrent() {
  state.arrangementStale = false;
  $("#chord-sheet").removeAttribute("data-stale");
}

function setGenerationInputsDisabled(disabled) {
  document.querySelectorAll("#lyrics, #analyze, .controls-grid input, .controls-grid select, #section-editor input, #section-editor textarea")
    .forEach(node => node.disabled = disabled);
}

function beginGeneration(mode, sectionName = "") {
  const panel = $("#generation-progress"), generateButton = $("#generate"), regenerateButton = $("#regenerate");
  const title = mode === "section" ? `Regenerating ${sectionName}…` : "Creating your complete arrangement…";
  const stages = mode === "section"
    ? [[0, "Preparing the selected section."], [7, "Running the local AI model."], [20, "Validating chords and rebuilding the melody."], [40, "Finishing the updated section."]]
    : [[0, "Analyzing lyric structure and musical controls."], [7, "Running the local AI model for each section."], [22, "Validating harmony and shaping the lyric melody."], [42, "Finalizing the editable chord sheet and playback."]];
  const startedAt = Date.now();
  panel.hidden = false; $("#generation-progress-title").textContent = title;
  generateButton.disabled = true; regenerateButton.disabled = true; setGenerationInputsDisabled(true);
  const activeButton = mode === "section" ? regenerateButton : generateButton;
  activeButton.setAttribute("aria-busy", "true");
  activeButton.textContent = mode === "section" ? "Regenerating…" : "Generating song…";
  const update = () => {
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    const stage = [...stages].reverse().find(([threshold]) => elapsed >= threshold) || stages[0];
    $("#generation-progress-detail").textContent = stage[1];
    $("#generation-elapsed").textContent = `${elapsed} second${elapsed === 1 ? "" : "s"} elapsed`;
  };
  update(); state.generationTimer = window.setInterval(update, 1000);
}

function endGeneration() {
  if (state.generationTimer) window.clearInterval(state.generationTimer);
  state.generationTimer = null; $("#generation-progress").hidden = true; setGenerationInputsDisabled(false);
  const generateButton = $("#generate"), regenerateButton = $("#regenerate");
  generateButton.disabled = false; regenerateButton.disabled = false;
  generateButton.removeAttribute("aria-busy"); regenerateButton.removeAttribute("aria-busy");
  generateButton.textContent = "Generate complete song"; regenerateButton.textContent = "Regenerate selected section";
}

async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  if (!response.ok) {
    let detail = response.statusText; try { const body = await response.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch (_) {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function controls() {
  return { key: $("#key").value, scale: $("#scale").value, genre: $("#genre").value, mood: $("#mood").value,
    tempo: Number($("#tempo").value), time_signature: $("#time-signature").value, difficulty: $("#difficulty").value,
    chord_density: Number($("#density").value), variation: Number($("#variation").value), seed: Number($("#seed").value),
    decoding_method: $("#decoding").value, num_beams: 4, temperature: .8, top_k: 40 };
}

function editedSections() {
  return [...document.querySelectorAll(".section-edit")].map(node => ({ name: node.querySelector("input").value.trim() || "Verse",
    lines: node.querySelector("textarea").value.split(/\r?\n/).filter(line => line.trim()), features: [] }));
}

function renderSectionEditor(sections) {
  const host = $("#section-editor"); host.replaceChildren();
  sections.forEach(section => {
    const wrapper = document.createElement("div"); wrapper.className = "section-edit";
    const nameLabel = document.createElement("label"); nameLabel.textContent = "Section name";
    const name = document.createElement("input"); name.value = section.name; name.maxLength = 40; nameLabel.append(name);
    const lyricsLabel = document.createElement("label"); lyricsLabel.textContent = "Section lyrics";
    const lyrics = document.createElement("textarea"); lyrics.rows = Math.max(2, section.lines.length); lyrics.value = section.lines.join("\n"); lyricsLabel.append(lyrics);
    wrapper.append(nameLabel, lyricsLabel); host.append(wrapper);
  });
}

async function analyze() {
  setStatus("Analyzing lyrics locally…");
  try {
    state.analysis = await api("/lyrics/analyze", { method: "POST", body: JSON.stringify({ lyrics: $("#lyrics").value }) });
    renderSectionEditor(state.analysis.sections); setStatus(`Detected ${state.analysis.sections.length} section(s) and ${state.analysis.total_lines} lyric lines.`);
  } catch (error) { setStatus(error.message, true); }
}

function generationPayload() {
  return { project_name: $("#project-name").value, lyrics: $("#lyrics").value,
    sections: document.querySelectorAll(".section-edit").length ? editedSections() : null, controls: controls(), previous_progression: [] };
}

async function generate() {
  const payload = generationPayload();
  setStatus("Generating and validating chord progressions…"); beginGeneration("song");
  try {
    state.result = await api("/chords/generate", { method: "POST", body: JSON.stringify(payload) });
    state.analysis = state.result.analysis; state.manualEdits = []; markArrangementCurrent(); renderResult(); updateTechnical();
    setStatus(`Generated ${state.result.sections.length} section(s). Source: ${state.result.generation_source}.`);
  } catch (error) { setStatus(error.message, true); }
  finally { endGeneration(); }
}

function alignmentMap() {
  const map = new Map();
  (state.result?.alignments || []).forEach(item => map.set(`${item.section_name}|${item.line_index}|${item.word_index}`, item));
  return map;
}

function renderResult() {
  if (!state.result) return;
  const map = alignmentMap(), host = $("#chord-sheet"), sectionSelect = $("#selected-section");
  host.replaceChildren(); sectionSelect.replaceChildren();
  state.result.analysis.sections.forEach(section => {
    const option = document.createElement("option"); option.value = section.name; option.textContent = section.name; sectionSelect.append(option);
    const wrapper = document.createElement("div"); wrapper.className = "sheet-section";
    const heading = document.createElement("h3"); heading.textContent = `[${section.name}]`; wrapper.append(heading);
    section.lines.forEach((line, lineIndex) => {
      const row = document.createElement("div"); row.className = "song-line";
      const words = line.split(/\s+/).filter(Boolean);
      words.forEach((word, wordIndex) => {
        const cell = document.createElement("span"); cell.className = "word-cell";
        const chord = document.createElement("input"); chord.setAttribute("aria-label", `Chord above ${word}`); chord.maxLength = 14;
        chord.dataset.section = section.name; chord.dataset.line = String(lineIndex); chord.dataset.word = String(wordIndex);
        chord.value = map.get(`${section.name}|${lineIndex}|${wordIndex}`)?.chord || "";
        chord.addEventListener("change", collectManualEdits);
        const lyric = document.createElement("span"); lyric.textContent = word; cell.append(chord, lyric); row.append(cell);
      });
      wrapper.append(row);
    }); host.append(wrapper);
  });
  $("#key").value = state.result.controls.key; $("#transpose-key").value = state.result.controls.key;
}

function collectManualEdits() {
  if (!state.result) return;
  const existing = new Map(state.result.alignments.map(item => [`${item.section_name}|${item.line_index}|${item.word_index}`, item]));
  const alignments = [], edits = [];
  document.querySelectorAll(".word-cell input").forEach(input => {
    const chord = input.value.trim(); if (!chord) return;
    const key = `${input.dataset.section}|${input.dataset.line}|${input.dataset.word}`;
    const original = existing.get(key);
    const item = original ? { ...original, chord } : { section_name: input.dataset.section, line_index: Number(input.dataset.line), word_index: Number(input.dataset.word), roman_numeral: "manual", chord, beats: 4 };
    alignments.push(item); if (!original || original.chord !== chord) edits.push({ ...item, previous_chord: original?.chord || null });
  });
  state.result.alignments = alignments; state.manualEdits = edits;
}

async function validateEdits() {
  collectManualEdits();
  for (const item of state.result.alignments) {
    const checked = await api("/chords/validate", { method: "POST", body: JSON.stringify({ chord: item.chord, key: state.result.controls.key, scale: state.result.controls.scale, difficulty: state.result.controls.difficulty }) });
    if (!checked.valid) throw new Error(`${item.chord}: ${checked.reason}`);
  }
}

async function regenerateSection() {
  if (!state.result) return setStatus("Generate a song first.", true);
  if (state.arrangementStale) return setStatus("Inputs changed. Generate the complete song before regenerating one section.", true);
  const selectedSection = $("#selected-section").value;
  setStatus("Regenerating the selected section…"); beginGeneration("section", selectedSection);
  try {
    const payload = { ...generationPayload(), section_name: selectedSection };
    const nextSeed = (payload.controls.seed + 1) % 2147483647;
    payload.controls.seed = nextSeed; $("#seed").value = String(nextSeed);
    const partial = await api("/chords/regenerate-section", { method: "POST", body: JSON.stringify(payload) });
    const name = payload.section_name;
    state.result.sections = state.result.sections.filter(item => item.section_name !== name).concat(partial.sections);
    state.result.alignments = state.result.alignments.filter(item => item.section_name !== name).concat(partial.alignments);
    state.result.repairs = state.result.repairs.filter(item => item.section_name !== name).concat(partial.repairs);
    state.result.raw_model_outputs[name] = partial.raw_model_outputs[name] || "";
    state.result.generation_source = partial.generation_source; state.result.inference_latency_ms = partial.inference_latency_ms; state.result.controls.seed = nextSeed;
    markArrangementCurrent(); renderResult(); updateTechnical(); setStatus(`${name} regenerated with seed ${nextSeed}.`);
  } catch (error) { setStatus(error.message, true); }
  finally { endGeneration(); }
}

async function transpose() {
  if (!state.result) return setStatus("Generate a song first.", true);
  if (state.arrangementStale) return setStatus("Inputs changed. Generate the complete song before transposing.", true);
  try {
    await validateEdits(); state.result = await api("/transpose", { method: "POST", body: JSON.stringify({ result: state.result, target_key: $("#transpose-key").value }) });
    renderResult(); setStatus(`Transposed to ${state.result.controls.key}.`);
  } catch (error) { setStatus(error.message, true); }
}

async function refreshProjects() {
  const projects = await api("/projects"), select = $("#project-list"); select.replaceChildren(new Option("Select a saved project", ""));
  projects.forEach(project => select.add(new Option(project.name, String(project.id))));
}

async function saveProject() {
  if (!state.result) return setStatus("Generate a song before saving.", true);
  if (state.arrangementStale) return setStatus("Inputs changed. Generate the complete song again before saving.", true);
  try {
    await validateEdits(); state.result.project_name = $("#project-name").value;
    const payload = { name: $("#project-name").value, lyrics: $("#lyrics").value, result: state.result, manual_chord_edits: state.manualEdits };
    const project = await api(state.projectId ? `/projects/${state.projectId}` : "/projects", { method: state.projectId ? "PUT" : "POST", body: JSON.stringify(payload) });
    state.projectId = project.id; await refreshProjects(); $("#project-list").value = String(project.id); setStatus("Project saved.");
  } catch (error) { setStatus(error.message, true); }
}

async function openProject() {
  const id = Number($("#project-list").value); if (!id) return setStatus("Choose a saved project.", true);
  try {
    const project = await api(`/projects/${id}`); state.projectId = id; state.result = project.result; state.manualEdits = project.manual_chord_edits;
    $("#project-name").value = project.name; $("#lyrics").value = project.lyrics; markArrangementCurrent(); renderSectionEditor(state.result.analysis.sections); renderResult(); updateTechnical(); setStatus("Project opened.");
  } catch (error) { setStatus(error.message, true); }
}

async function renameProject() {
  if (!state.projectId) return setStatus("Open or save a project first.", true);
  try { const project = await api(`/projects/${state.projectId}/rename`, { method: "PATCH", body: JSON.stringify({ name: $("#project-name").value }) }); await refreshProjects(); $("#project-list").value = String(project.id); setStatus("Project renamed."); }
  catch (error) { setStatus(error.message, true); }
}

async function duplicateProject() {
  if (!state.projectId) return setStatus("Open or save a project first.", true);
  try { const project = await api(`/projects/${state.projectId}/duplicate`, { method: "POST" }); state.projectId = project.id; $("#project-name").value = project.name; await refreshProjects(); $("#project-list").value = String(project.id); setStatus("Project duplicated."); }
  catch (error) { setStatus(error.message, true); }
}

async function deleteProject() {
  if (!state.projectId) return setStatus("Open or save a project first.", true);
  if (!window.confirm(`Delete “${$("#project-name").value}”? This cannot be undone.`)) return;
  try { await api(`/projects/${state.projectId}?confirm=true`, { method: "DELETE" }); state.projectId = null; await refreshProjects(); setStatus("Project deleted."); }
  catch (error) { setStatus(error.message, true); }
}

function audioContext() {
  if (!state.audio || state.audio.state === "closed") state.audio = new (window.AudioContext || window.webkitAudioContext)();
  return state.audio;
}

function clearScheduledPlayback() {
  state.sources.forEach(source => { try { source.stop(); } catch (_) {} }); state.sources = [];
  if (state.timer) clearInterval(state.timer); state.timer = null; state.events = []; state.paused = false; $("#now-playing").textContent = "Stopped.";
}

function stopPlayback() { state.playbackToken += 1; clearScheduledPlayback(); }

function scheduleNote(context, destination, midi, when, duration, timbre, gainValue, voice = "chord") {
  const oscillator = context.createOscillator(), gain = context.createGain(), filter = context.createBiquadFilter();
  oscillator.frequency.value = 440 * Math.pow(2, (midi - 69) / 12);
  oscillator.type = voice === "melody" ? (timbre === "synth" ? "triangle" : "sine") : timbre === "pad" ? "sine" : timbre === "pluck" ? "triangle" : timbre === "synth" ? "square" : "triangle";
  filter.type = "lowpass"; filter.frequency.value = voice === "melody" ? 3200 : timbre === "synth" ? 2400 : 1300;
  const attack = voice === "melody" ? .018 : timbre === "pad" ? .18 : .012, release = timbre === "pluck" ? Math.min(.7, duration) : Math.min(.25, duration / 3);
  gain.gain.setValueAtTime(.0001, when); gain.gain.exponentialRampToValueAtTime(Math.max(.0001, gainValue), when + attack);
  if (timbre === "pluck") gain.gain.exponentialRampToValueAtTime(.0001, when + Math.min(duration, .85));
  else { gain.gain.setValueAtTime(gainValue, Math.max(when + attack, when + duration - release)); gain.gain.exponentialRampToValueAtTime(.0001, when + duration); }
  oscillator.connect(filter).connect(gain).connect(destination); oscillator.start(when); oscillator.stop(when + duration + .05); state.sources.push(oscillator);
}

async function play() {
  if (!state.result) return setStatus("Generate a song first.", true);
  if (state.arrangementStale) return setStatus("This arrangement belongs to the previous inputs. Generate the complete song again before playback.", true);
  const playbackToken = state.playbackToken + 1; state.playbackToken = playbackToken; clearScheduledPlayback();
  try {
    await validateEdits(); if (playbackToken !== state.playbackToken) return;
    const context = audioContext(); await context.resume(); if (playbackToken !== state.playbackToken) return;
    const events = await api("/playback/timeline", { method: "POST", body: JSON.stringify({ result: state.result }) });
    if (playbackToken !== state.playbackToken) return; state.events = events; state.paused = false;
    const master = context.createGain(); master.gain.value = Math.min(.8, Number($("#volume").value)); master.connect(context.destination);
    const secondsPerBeat = 60 / state.result.controls.tempo, offset = context.currentTime + .08; state.startTime = offset;
    state.events.forEach(event => {
      event.midi_notes.forEach(note => scheduleNote(context, master, note, offset + event.start_beat * secondsPerBeat, event.duration_beats * secondsPerBeat, $("#timbre").value, .12 / Math.sqrt(event.midi_notes.length)));
      (event.melody_midi_notes || []).forEach((note, index) => {
        const noteBeats = event.melody_note_beats || event.duration_beats;
        scheduleNote(context, master, note, offset + (event.start_beat + index * noteBeats) * secondsPerBeat, noteBeats * secondsPerBeat * .86, $("#timbre").value, .22, "melody");
      });
    });
    state.timer = setInterval(() => {
      if (playbackToken !== state.playbackToken) return;
      const beat = (context.currentTime - state.startTime) / secondsPerBeat;
      const current = [...state.events].reverse().find(event => event.start_beat <= beat);
      if (current) $("#now-playing").textContent = `${current.section_name}, line ${current.line_index + 1}: ${current.chord} · “${current.lyric_fragment}”${state.paused ? " (paused)" : ""}`;
      const last = state.events.at(-1); if (last && beat > last.start_beat + last.duration_beats) stopPlayback();
    }, 100); setStatus("Playback started with a lyric-shaped original melody.");
  } catch (error) { stopPlayback(); setStatus(error.message, true); }
}

function pause() {
  if (!state.timer) return;
  state.paused = true;
  if (!$("#now-playing").textContent.endsWith(" (paused)")) $("#now-playing").textContent += " (paused)";
  if (state.audio?.state === "running") state.audio.suspend().catch(error => setStatus(error.message, true));
}
function resume() {
  if (!state.timer) return;
  state.paused = false;
  if (state.audio?.state === "suspended") state.audio.resume().catch(error => setStatus(error.message, true));
}
async function restart() { stopPlayback(); await play(); }

async function download(kind) {
  if (!state.result) return setStatus("Generate a song first.", true);
  if (state.arrangementStale) return setStatus("Inputs changed. Generate the complete song again before exporting.", true);
  try {
    await validateEdits(); const response = await fetch(`/api/v1/exports/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.result) });
    if (!response.ok) throw new Error((await response.json()).detail || "Export failed");
    const blob = await response.blob(), url = URL.createObjectURL(blob), link = document.createElement("a");
    const disposition = response.headers.get("Content-Disposition") || ""; link.download = disposition.match(/filename="([^"]+)"/)?.[1] || `song.${kind}`;
    link.href = url; link.click(); URL.revokeObjectURL(url); setStatus(`${kind.toUpperCase()} downloaded.`);
  } catch (error) { setStatus(error.message, true); }
}

async function updateTechnical() {
  try {
    const model = await api("/model"), repairs = state.result?.repairs.flatMap(item => item.reasons) || [];
    const values = { Model: model.base_model, "Base-model status": model.base_model_status, "LoRA adapter": model.adapter_version,
      Device: model.device, "Generation source": state.result?.generation_source || "No generation yet", "Inference latency": state.result ? `${state.result.inference_latency_ms} ms` : "n/a",
      "Repairs performed": repairs.length ? repairs.join("; ") : "None" };
    const dl = $("#technical-details"); dl.replaceChildren(); Object.entries(values).forEach(([key, value]) => { const dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = key; dd.textContent = value; dl.append(dt, dd); });
  } catch (_) {}
}

$("#density").addEventListener("input", event => $("#density-value").value = event.target.value);
$("#variation").addEventListener("input", event => $("#variation-value").value = event.target.value);
$("#lyrics").addEventListener("input", markArrangementStale);
$(".controls-grid").addEventListener("input", markArrangementStale);
$(".controls-grid").addEventListener("change", markArrangementStale);
$("#section-editor").addEventListener("input", markArrangementStale);
$("#analyze").addEventListener("click", analyze); $("#generate").addEventListener("click", generate); $("#regenerate").addEventListener("click", regenerateSection); $("#transpose").addEventListener("click", transpose);
$("#save").addEventListener("click", saveProject); $("#open").addEventListener("click", openProject); $("#rename").addEventListener("click", renameProject); $("#duplicate").addEventListener("click", duplicateProject); $("#delete").addEventListener("click", deleteProject);
$("#play").addEventListener("click", play); $("#pause").addEventListener("click", pause); $("#resume").addEventListener("click", resume); $("#stop").addEventListener("click", stopPlayback); $("#restart").addEventListener("click", restart);
document.querySelectorAll("[data-export]").forEach(button => button.addEventListener("click", () => download(button.dataset.export)));
refreshProjects().catch(() => {}); updateTechnical();
