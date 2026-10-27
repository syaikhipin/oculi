const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const state = { watchlists: [], watchlist: "default", sources: [], lastEventId: 0, feed: [], paused: false, eventRows: [], workers: {} };

function getApiKey() { return localStorage.getItem("oculi_api_key") || ""; }

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = getApiKey();
  if (key) headers.set("Authorization", `Bearer ${key}`);
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401 && !state.authPrompted) {
    state.authPrompted = true;
    const entered = prompt("API key for Oculi");
    if (entered) {
      localStorage.setItem("oculi_api_key", entered.trim());
      state.authPrompted = false;
      return api(path, options);
    }
  }
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!response.ok) throw new Error((data && data.detail) || `${response.status} ${response.statusText}`);
  return data;
}
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (iso) => { if (!iso) return "—"; const d = new Date(iso); return isNaN(d) ? iso : d.toLocaleTimeString(); };
const fmtNum = (value) => (value ?? 0).toLocaleString();

function setHealth(text, ok) {
  const el = $("#health");
  el.innerHTML = `<span class="dot"></span>${esc(text)}`;
  el.classList.toggle("ok", !!ok);
}

function switchTab(name) {
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
}

function eventTitle(e) {
  if (e.type === "face_match") return e.subject?.name || e.subject?.subject_id || "match";
  if (e.type === "face") return "face detected";
  if (e.type === "spoof") return "spoof attempt";
  if (e.type === "plate") return e.plate?.text || "plate";
  if (e.type === "object") return `${e.category || "object"} · ${e.label || ""}`;
  return e.type;
}
function eventMeta(e) {
  const bits = [];
  if (e.source_id) bits.push(e.source_id);
  if (e.track_id) bits.push(`#${e.track_id}`);
  if (e.subject?.score != null) bits.push(`match ${(e.subject.score * 100).toFixed(1)}%`);
  if (e.liveness && e.liveness.is_live === false) bits.push(`spoof ${(e.liveness.score * 100).toFixed(1)}%`);
  if (e.attributes?.sex) bits.push(e.attributes.sex);
  if (e.attributes?.age) bits.push(`${e.attributes.age}y`);
  if (e.plate?.score != null) bits.push(`ocr ${(e.plate.score * 100).toFixed(1)}%`);
  if (e.score != null) bits.push(`det ${(e.score * 100).toFixed(1)}%`);
  return bits.join(" · ");
}
function eventRow(e) {
  const liveness = e.liveness ? `<span class="badge ${e.liveness.is_live ? "ok" : "bad"}">${e.liveness.is_live ? "live" : "spoof"}</span>` : "—";
  const subject = e.subject?.name || e.subject?.subject_id || e.plate?.text || e.label || "—";
  const score = e.subject?.score ?? e.plate?.score ?? e.score;
  return `<tr>
    <td>${esc(fmtTime(e.ts))}</td>
    <td>${e.thumbnail ? `<img class="shot" src="${e.thumbnail}" alt="">` : "—"}</td>
    <td><span class="badge ${e.type === "face_match" ? "ok" : e.type === "spoof" ? "bad" : e.type === "plate" ? "warn" : ""}">${esc(e.type)}</span></td>
    <td>${esc(e.source_id || "—")}</td>
    <td>${esc(e.track_id || "—")}</td>
    <td>${esc(e.label || e.category || "—")}</td>
    <td class="wrap">${esc(subject)}</td>
    <td>${score != null ? (score * 100).toFixed(1) + "%" : "—"}</td>
    <td>${liveness}</td>
  </tr>`;
}
function feedItem(e) {
  return `<div class="event">
    <span class="kind ${esc(e.type)}">${esc(e.type.replace("_", " "))}</span>
    <span class="meta">${esc(eventTitle(e))} · ${esc(eventMeta(e))}</span>
    <time>${esc(fmtTime(e.ts))}</time>
  </div>`;
}

async function loadHealth() {
  try {
    const h = await api("/health");
    state.health = h;
    const pipeline = h.pipeline;
    const bits = [];
    if (pipeline) {
      bits.push(`face ${pipeline.face?.engine || "?"}`);
      bits.push(pipeline.objects ? "objects" : "objects off");
      bits.push(pipeline.alpr?.enabled ? "ALPR" : "ALPR off");
      bits.push(pipeline.liveness?.enabled ? "liveness" : "liveness off");
    }
    setHealth(`online · ${bits.join(" · ") || "workers off"}`, true);
    $("#stat-sources").textContent = fmtNum(h.counts?.sources);
    $("#stat-events").textContent = fmtNum(h.counts?.events);
    $("#stat-subjects").textContent = fmtNum(h.counts?.subjects);
    $("#stat-templates").textContent = fmtNum(h.counts?.templates);
    $("#stat-pipeline").textContent = pipeline ? `${pipeline.face?.embedding_dim || "?"}D` : "off";
    renderMix(h.events?.by_type || {});
    renderWorkers(h.workers);
  } catch (err) { setHealth(`offline · ${err.message}`, false); }
}

function renderMix(counts) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((sum, [, n]) => sum + n, 0);
  $("#event-mix-total").textContent = `${fmtNum(total)} events`;
  $("#event-mix").className = `mix${entries.length ? "" : " empty"}`;
  $("#event-mix").innerHTML = entries.length
    ? entries.map(([type, n]) => `<div class="mixrow"><span>${esc(type)}</span><span class="bar"><i style="width:${Math.max(3, (n / total) * 100)}%"></i></span><span>${fmtNum(n)}</span></div>`).join("")
    : "No data";
}
function renderWorkers(workers) {
  const list = workers?.sources || [];
  state.workers = Object.fromEntries(list.map((w) => [w.id, w]));
  $("#worker-list").className = `list${list.length ? "" : " empty"}`;
  $("#worker-list").innerHTML = list.length ? list.map((w) => `<div class="rowitem"><div class="grow"><div class="title">${esc(w.id)}</div><div class="sub">${esc(w.state)} · ${fmtNum(w.frames)} frames · ${fmtNum(w.events)} events${w.errors ? ` · ${w.errors} errors` : ""}</div></div><span class="badge ${w.state === "running" ? "ok" : w.state === "error" ? "bad" : ""}">${esc(w.state)}</span></div>`).join("") : "Workers are not running in this process.";
}

async function loadFeed(reset = false) {
  if (state.paused) return;
  const path = `/v1/events?after_id=${reset ? 0 : state.lastEventId}&limit=${reset ? 60 : 100}&thumbnails=false`;
  try {
    const rows = await api(path);
    if (reset) state.feed = [];
    for (const e of rows) {
      state.feed.push(e);
      if (e.id != null) state.lastEventId = Math.max(state.lastEventId, e.id);
    }
    state.feed = state.feed.slice(-100);
    $("#live-events").className = `feed${state.feed.length ? "" : " empty"}`;
    $("#live-events").innerHTML = state.feed.length
      ? state.feed.slice().reverse().map(feedItem).join("")
      : "No events yet. Add a source or run a detection.";
  } catch (err) { console.warn(err); }
}

async function loadEventTable() {
  try {
    const type = $("#event-type").value;
    const source = $("#event-source").value;
    const params = new URLSearchParams({ after_id: "0", limit: "100" });
    if (type) params.set("type", type);
    if (source) params.set("source_id", source);
    const rows = await api(`/v1/events?${params}`);
    const tbody = $("#events-table tbody");
    tbody.innerHTML = rows.length ? rows.slice().reverse().map(eventRow).join("") : `<tr><td colspan="9" class="empty">No events</td></tr>`;
  } catch (err) { console.warn(err); }
}

async function loadWatchlists() {
  try {
    state.watchlists = await api("/v1/watchlists");
    if (!state.watchlists.find((w) => w.id === state.watchlist)) {
      state.watchlist = state.watchlists[0]?.id || "default";
    }
    $("#watchlist-list").className = `list${state.watchlists.length ? "" : " empty"}`;
    $("#watchlist-list").innerHTML = state.watchlists.map((w) => `
      <div class="rowitem ${w.id === state.watchlist ? "active" : ""}" data-watchlist="${esc(w.id)}">
        <div class="grow"><div class="title">${esc(w.name)}</div><div class="sub">${esc(w.id)} · ${fmtNum(w.subjects ?? w.subject_count)} subjects · ${fmtNum(w.templates ?? w.template_count)} templates</div></div>
      </div>`).join("");
    $$("#watchlist-list [data-watchlist]").forEach((el) => el.onclick = () => { state.watchlist = el.dataset.watchlist; loadWatchlists(); loadSubjects(); });
    [["#source-watchlist"], ["#detect-watchlist"]].forEach(([sel]) => {
      $(sel).innerHTML = state.watchlists.map((w) => `<option value="${esc(w.id)}" ${w.id === state.watchlist ? "selected" : ""}>${esc(w.name)}</option>`).join("");
    });
    $("#enroll-watchlist").textContent = state.watchlist;
    loadSubjects();
  } catch (err) { console.warn(err); }
}

async function loadSubjects() {
  try {
    const subjects = await api(`/v1/watchlists/${encodeURIComponent(state.watchlist)}/subjects`);
    $("#subject-list").className = `list${subjects.length ? "" : " empty"}`;
    $("#subject-list").innerHTML = subjects.length ? subjects.map((s) => `
      <div class="rowitem">
        <div class="grow"><div class="title">${esc(s.name)}</div><div class="sub">${esc(s.id)} · ${fmtNum(s.template_count)} templates</div></div>
        <div class="actions"><button class="danger small" data-delete-subject="${esc(s.id)}">delete</button></div>
      </div>`).join("") : "No subjects in this watchlist";
    $$("#subject-list [data-delete-subject]").forEach((btn) => btn.onclick = async () => {
      if (!confirm(`Delete ${btn.dataset.deleteSubject}?`)) return;
      await api(`/v1/subjects/${encodeURIComponent(btn.dataset.deleteSubject)}`, { method: "DELETE" });
      loadSubjects();
    });
  } catch (err) { $("#subject-list").textContent = err.message; }
}

async function loadSources() {
  try {
    state.sources = await api("/v1/sources");
    const selectedSource = $("#event-source").value;
    $("#event-source").innerHTML = `<option value="">all sources</option>` + state.sources.map((s) => `<option value="${esc(s.id)}">${esc(s.name)}</option>`).join("");
    if (selectedSource) $("#event-source").value = selectedSource;
    const live = state.workers;
    $("#source-list").className = `list${state.sources.length ? "" : " empty"}`;
    $("#source-list").innerHTML = state.sources.length ? state.sources.map((s) => {
      const w = live[s.id];
      return `<div class="rowitem">
        <div class="grow"><div class="title">${esc(s.name)} ${s.milestone ? '<span class="badge warn">Milestone</span>' : ""}</div>
        <div class="sub">${esc(s.kind)} · ${esc(s.url)} · ${esc(s.watchlist_id)}${w ? ` · ${esc(w.state)}` : ""}</div></div>
        <div class="actions">
          <button class="ghost small" data-toggle="${esc(s.id)}" data-enabled="${s.enabled ? "1" : "0"}">${s.enabled ? "disable" : "enable"}</button>
          <button class="ghost small" data-restart="${esc(s.id)}">restart</button>
          <button class="danger small" data-delete-source="${esc(s.id)}">delete</button>
        </div></div>`;
    }).join("") : "No sources configured";
    $$("#source-list [data-toggle]").forEach((b) => b.onclick = async () => {
      await api(`/v1/sources/${encodeURIComponent(b.dataset.toggle)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: b.dataset.enabled !== "1" }) });
      loadSources();
    });
    $$("#source-list [data-restart]").forEach((b) => b.onclick = async () => {
      try { await api(`/v1/sources/${encodeURIComponent(b.dataset.restart)}/restart`, { method: "POST" }); }
      catch (err) { alert(err.message); }
    });
    $$("#source-list [data-delete-source]").forEach((b) => b.onclick = async () => {
      if (!confirm("Delete source?")) return;
      await api(`/v1/sources/${encodeURIComponent(b.dataset.deleteSource)}`, { method: "DELETE" });
      loadSources();
    });
  } catch (err) { console.warn(err); }
}

function bind() {
  $$("#nav button").forEach((b) => b.onclick = () => switchTab(b.dataset.tab));
  $("#pause-feed").onclick = (e) => { state.paused = !state.paused; e.target.textContent = state.paused ? "resume" : "pause"; };
  $("#events-refresh").onclick = loadEventTable;
  $("#event-type").onchange = loadEventTable;
  $("#event-source").onchange = loadEventTable;
  $("#sources-refresh").onclick = loadSources;
  $("#auth-key").onclick = () => {
    const entered = prompt("Oculi API key (leave empty to clear)", getApiKey());
    if (entered !== null) {
      localStorage.setItem("oculi_api_key", entered.trim());
      state.authPrompted = false;
      location.reload();
    }
  };

  $("#watchlist-form").onsubmit = async (e) => {
    e.preventDefault();
    const name = e.target.name.value.trim();
    if (!name) return;
    try { await api("/v1/watchlists", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }); e.target.reset(); loadWatchlists(); }
    catch (err) { alert(err.message); }
  };
  $("#enroll-form").onsubmit = async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("name", e.target.name.value);
    if (e.target.subject_id.value) form.append("subject_id", e.target.subject_id.value);
    for (const file of e.target.images.files) form.append("images", file);
    form.append("require_quality", "true");
    try {
      const result = await api(`/v1/watchlists/${encodeURIComponent(state.watchlist)}/subjects`, { method: "POST", body: form });
      e.target.reset();
      alert(`Enrolled ${result.subject.name} (${result.templates.length} template${result.templates.length === 1 ? "" : "s"})`);
      loadSubjects(); loadWatchlists();
    } catch (err) { alert(err.message); }
  };
  $("#search-form").onsubmit = async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("image", e.target.image.files[0]);
    form.append("top_k", e.target.top_k.value || "5");
    $("#search-result").className = "result";
    $("#search-result").textContent = "searching…";
    try {
      const result = await api(`/v1/watchlists/${encodeURIComponent(state.watchlist)}/search`, { method: "POST", body: form });
      $("#search-result").textContent = JSON.stringify(result, null, 2);
    } catch (err) { $("#search-result").textContent = err.message; }
  };
  $("#source-form").onsubmit = async (e) => {
    e.preventDefault();
    const payload = {
      name: e.target.name.value, url: e.target.url.value, kind: e.target.kind.value,
      watchlist_id: e.target.watchlist_id.value, milestone: e.target.milestone.checked,
    };
    try { await api("/v1/sources", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); e.target.reset(); loadSources(); loadHealth(); }
    catch (err) { alert(err.message); }
  };
  $("#detect-form").onsubmit = async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("image", e.target.image.files[0]);
    form.append("watchlist_id", e.target.watchlist_id.value);
    $("#detect-json").textContent = "running…";
    try {
      const result = await api("/v1/detect", { method: "POST", body: form });
      $("#detect-json").textContent = JSON.stringify(result, null, 2);
      renderDetect(result, e.target.image.files[0]);
    } catch (err) { $("#detect-json").textContent = err.message; }
  };
}

function renderDetect(result, file) {
  const stage = $("#detect-stage");
  stage.className = "stage";
  stage.innerHTML = "";
  const img = new Image();
  img.onload = () => {
    const wrap = document.createElement("div");
    wrap.className = "stage-inner";
    wrap.appendChild(img);
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    wrap.appendChild(canvas);
    stage.appendChild(wrap);
    const ctx = canvas.getContext("2d");
    const draw = (box, color, label) => {
      const [x, y, w, h] = box;
      ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.strokeRect(x, y, w, h);
      ctx.font = `${Math.max(12, canvas.width / 60)}px system-ui`; ctx.fillStyle = color;
      ctx.fillText(label, x + 4, Math.max(14, y - 6));
    };
    result.faces.forEach((f) => draw(f.bbox, f.liveness && !f.liveness.is_live ? "#ff6b6b" : "#37d6a3", f.matches?.[0]?.name || "face"));
    result.objects.forEach((o) => draw(o.bbox, "#5aa9ff", o.label));
    result.plates.forEach((p) => draw(p.bbox, "#ffb454", p.text || "plate"));
  };
  img.src = URL.createObjectURL(file);
  const summary = [];
  for (const f of result.faces) {
    const live = f.liveness ? ` · ${f.liveness.is_live ? "live" : "SPOOF"} ${(f.liveness.score * 100).toFixed(1)}%` : "";
    const match = f.matches?.[0] ? ` · ${f.matches[0].name} ${(f.matches[0].score * 100).toFixed(1)}%` : "";
    summary.push(`<div class="rowitem"><div class="grow"><div class="title">face${match}</div><div class="sub">${f.attributes?.sex || "?"} · ${f.attributes?.age || "?"}y${live}</div></div></div>`);
  }
  for (const o of result.objects) summary.push(`<div class="rowitem"><div class="grow"><div class="title">${esc(o.label)}</div><div class="sub">${esc(o.category)} · ${(o.score * 100).toFixed(1)}%</div></div></div>`);
  for (const p of result.plates) summary.push(`<div class="rowitem"><div class="grow"><div class="title">${esc(p.text || "unread plate")}</div><div class="sub">ALPR · ${(p.score * 100).toFixed(1)}%</div></div></div>`);
  $("#detect-summary").className = `list${summary.length ? "" : " empty"}`;
  $("#detect-summary").innerHTML = summary.length ? summary.join("") : "No detections";
}

async function tick() {
  await loadFeed();
  await loadHealth();
}
async function tickSlow() { await loadSources(); await loadEventTable(); }

(function init() {
  bind();
  loadWatchlists();
  loadSources();
  loadFeed(true);
  loadHealth();
  loadEventTable();
  setInterval(tick, 2000);
  setInterval(tickSlow, 5000);
})();
