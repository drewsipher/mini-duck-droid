// Duck Dashboard: one page, hash-routed tabs, talks to server.py's /api/*.
// #progress #animations[/<clip path>] #training[/<runs dir>/<run>] #skills[/<name>/<vN>]
// #tools #jobs/<id> #telemetry #experiments #puppet
import * as A from './anim.js';
import { createDuckViewer } from './viewer.js';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const JOINTS = A.JOINTS;
const R2D = 180 / Math.PI;
const jointText = j => j.replace(/_/g, ' ');

async function api(path, opt) {
  const r = await fetch('/api/' + path, { cache: 'no-store', ...opt });
  if (!r.ok) { let msg = `${r.status} ${r.statusText}`; try { msg = (await r.json()).error || msg; } catch { /* not json */ } throw new Error(msg); }
  return r.json();
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const text = async p => { const r = await fetch(p, { cache: 'no-store' }); if (!r.ok) throw new Error(r.status); return r.text(); };

const fmtNum = (v, d = 2) => { if (v == null || Number.isNaN(+v)) return '–'; if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M'; if (Math.abs(v) >= 1e4) return Math.round(v).toLocaleString(); const s = (+v).toFixed(d); return s.includes('.') ? s.replace(/\.?0+$/, '') : s; };
const bytes = n => n == null ? '' : n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n >= 1e3 ? Math.round(n / 1e3) + ' kB' : n + ' B';
function ago(t) {
  const ms = typeof t === 'number' ? (t > 1e12 ? t : t * 1000) : Date.parse(t);
  if (!ms) return '';
  const s = Math.max(0, (Date.now() - ms) / 1000);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
function when(t) {
  const ms = typeof t === 'number' ? (t > 1e12 ? t : t * 1000) : Date.parse(t);
  if (!ms) return '–';
  const d = new Date(ms), today = new Date().toDateString() === d.toDateString();
  return (today ? '' : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ') + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
}
const chip = (t, cls = '') => `<span class="chip ${cls}">${esc(t)}</span>`;
const STATUS_CLASS = { done: 'good', in_progress: 'info', blocked: 'bad', todo: '', running: 'info', killing: 'warn', failed: 'bad', killed: 'warn', lost: 'warn', pass: 'good', fail: 'bad' };
const statusChip = s => chip(String(s ?? '–').replace(/_/g, ' '), STATUS_CLASS[s] || '');
const passChip = (ok, extra = '') => ok == null ? chip('not checked') : ok ? chip('pass' + extra, 'good') : chip('fail' + extra, 'bad');
let toastTimer = null;
function toast(msg, bad = false) {
  const t = $('#toast'); t.textContent = msg; t.className = bad ? 'bad' : ''; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), bad ? 5000 : 2500);
}
const pre = (obj, cls = '') => `<pre class="log json ${cls}">${esc(typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1))}</pre>`;
const kv = (pairs, cls = '') => `<dl class="kv ${cls}">${pairs.filter(p => p && p[1] != null && p[1] !== '').map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
const fileLink = (url, name) => `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(name || url.split('/').pop())}</a>`;
const mediaGrid = (items) => items.length ? `<div class="media-grid">${items.map(m => `<figure>${m.kind === 'mp4' || /\.mp4$/i.test(m.url)
  ? `<video controls playsinline preload="metadata" src="${esc(m.url)}"></video>` : `<a href="${esc(m.url)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(m.url)}" alt="${esc(m.name)}"></a>`}
  <figcaption>${fileLink(m.url, m.name)}${m.dir ? ` <span class="muted">· ${esc(m.dir)}</span>` : ''}${m.mtime ? ` <span class="muted">· ${esc(ago(m.mtime))}</span>` : ''}</figcaption></figure>`).join('')}</div>` : '';

// ---------------------------------------------------------------- running tools
async function runTool(tool_id, args, label) {
  try {
    const r = await post('jobs', { tool_id, args });
    toast(`started ${label || r.job.tool_name}`);
    location.hash = '#jobs/' + r.id;
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------- the line chart
const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function niceStep(range, n) { const raw = range / Math.max(1, n), mag = 10 ** Math.floor(Math.log10(raw || 1)), k = raw / mag; return (k < 1.5 ? 1 : k < 3 ? 2 : k < 7 ? 5 : 10) * mag; }
// series: [{name, points: [[x, y], ...]}]; one chart = one measure (never two y axes)
function chart(el, { title, series, xFmt = fmtNum, yFmt = v => fmtNum(v, 3), unit = '', height = 180, zero = false }) {
  const live = series.filter(s => s.points.length);
  el.className = 'chart';
  el.innerHTML = `<div class="ch-head"><b>${esc(title)}</b><span class="ch-val muted small">${unit ? esc(unit) : ''}</span></div>` +
    (live.length ? `<canvas style="height:${height}px"></canvas>` : '<div class="ch-empty">no data</div>') +
    (live.length >= 2 ? `<div class="ch-legend">${live.map((s, i) => `<span><i style="background:${cssVar(SERIES[(s.slot ?? i) % 8])}"></i>${esc(s.name)}</span>`).join('')}</div>` : '');
  if (!live.length) return;
  const cv = $('canvas', el), ctx = cv.getContext('2d'), val = $('.ch-val', el);
  const M = { l: 46, r: 10, t: 8, b: 20 };
  let xs = live.flatMap(s => s.points.map(p => p[0])), ys = live.flatMap(s => s.points.map(p => p[1])).filter(v => v != null && Number.isFinite(v));
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (zero) y0 = Math.min(0, y0);
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  let hover = null;
  const draw = () => {
    const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2), W = r.width, H = r.height;
    if (!W) return;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = x => M.l + (x - x0) / (x1 - x0) * (W - M.l - M.r), Y = y => M.t + (1 - (y - y0) / (y1 - y0)) * (H - M.t - M.b);
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'middle';
    const line = cssVar('--line'), muted = cssVar('--muted');
    const ys0 = niceStep(y1 - y0, 4), dec = Math.max(0, -Math.floor(Math.log10(ys0))), tick = v => Math.abs(v) >= 1e4 ? fmtNum(v) : (+v.toFixed(dec)).toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
    for (let y = Math.ceil(y0 / ys0) * ys0; y <= y1; y += ys0) { ctx.fillStyle = line; ctx.fillRect(M.l, Y(y), W - M.l - M.r, 1); ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.fillText(tick(y), M.l - 6, Y(y)); }
    const xs0 = niceStep(x1 - x0, W > 500 ? 6 : 4);
    ctx.textAlign = 'center';
    for (let x = Math.ceil(x0 / xs0) * xs0; x <= x1 + 1e-9; x += xs0) { ctx.fillStyle = line; ctx.fillRect(X(x), H - M.b, 1, 4); ctx.fillStyle = muted; ctx.fillText(xFmt(x), X(x), H - M.b / 2 + 1); }
    ctx.fillStyle = line; ctx.fillRect(M.l, H - M.b, W - M.l - M.r, 1);
    live.forEach((s, i) => {
      const col = cssVar(SERIES[(s.slot ?? i) % 8]);
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      let pen = false;
      for (const [x, y] of s.points) { if (y == null || !Number.isFinite(y)) { pen = false; continue; } if (pen) ctx.lineTo(X(x), Y(y)); else { ctx.moveTo(X(x), Y(y)); pen = true; } }
      ctx.stroke();
      if (s.points.length <= 40) { ctx.fillStyle = col; for (const [x, y] of s.points) if (y != null) { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, Math.PI * 2); ctx.fill(); } }
    });
    if (hover != null) {
      ctx.fillStyle = muted; ctx.fillRect(X(hover.x), M.t, 1, H - M.t - M.b);
      for (const h of hover.hits) { ctx.fillStyle = cssVar('--panel'); ctx.beginPath(); ctx.arc(X(h.x), Y(h.y), 6, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = h.col; ctx.beginPath(); ctx.arc(X(h.x), Y(h.y), 4, 0, Math.PI * 2); ctx.fill(); }
    }
  };
  const nearest = px => {
    const r = cv.getBoundingClientRect(), x = x0 + (px - M.l) / (r.width - M.l - M.r) * (x1 - x0);
    const hits = [];
    live.forEach((s, i) => { let best = null; for (const p of s.points) if (p[1] != null && (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p; if (best) hits.push({ x: best[0], y: best[1], name: s.name, col: cssVar(SERIES[(s.slot ?? i) % 8]) }); });
    if (!hits.length) return null;
    const hx = hits.reduce((a, h) => Math.abs(h.x - x) < Math.abs(a - x) ? h.x : a, hits[0].x);
    return { x: hx, hits: hits.filter(h => h.x === hx) };
  };
  cv.addEventListener('pointermove', e => { const r = cv.getBoundingClientRect(); hover = nearest(e.clientX - r.left); if (hover) val.innerHTML = `${esc(xFmt(hover.x))}: ` + hover.hits.map(h => `<span style="color:${h.col}">●</span> ${live.length > 1 ? esc(h.name) + ' ' : ''}${esc(yFmt(h.y))}`).join(' · '); draw(); });
  cv.addEventListener('pointerleave', () => { hover = null; val.textContent = unit; draw(); });
  new ResizeObserver(draw).observe(cv);
  draw();
}

// ---------------------------------------------------------------- a small markdown renderer (headings, lists, code, tables, links)
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/\[([^\]]+)\]\(((?:https?:|\/)[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
}
function md(src) {
  const lines = (src || '').replace(/\r/g, '').split('\n'); let out = '', i = 0, para = [], list = null;
  const flush = () => { if (para.length) { out += `<p>${inline(para.join(' '))}</p>`; para = []; } if (list) { out += `</${list}>`; list = null; } };
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) { flush(); let code = []; i++; while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]); i++; out += `<pre>${esc(code.join('\n'))}</pre>`; continue; }
    const h = /^(#{1,4})\s+(.*)/.exec(l);
    if (h) { flush(); out += `<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`; i++; continue; }
    if (/^\s*\|.*\|\s*$/.test(l)) {
      flush(); const rows = []; while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      const cells = r => r.trim().slice(1, -1).split('|').map(c => c.trim());
      const body = rows.filter(r => !/^\s*\|[\s:-]+\|\s*$/.test(r));
      out += '<table>' + body.map((r, k) => `<tr>${cells(r).map(c => `<${k ? 'td' : 'th'}>${inline(c)}</${k ? 'td' : 'th'}>`).join('')}</tr>`).join('') + '</table>'; continue;
    }
    const li = /^\s*([-*]|\d+\.)\s+(.*)/.exec(l);
    if (li) { const kind = /\d/.test(li[1]) ? 'ol' : 'ul'; if (list !== kind) { flush(); out += `<${kind}>`; list = kind; } out += `<li>${inline(li[2].replace(/^([a-z_ ]{2,20}):/i, '**$1:**'))}</li>`; i++; continue; }
    if (!l.trim()) { flush(); i++; continue; }
    if (list) flush();
    para.push(l); i++;
  }
  flush();
  return `<div class="md">${out}</div>`;
}

// ---------------------------------------------------------------- router
const TABS = ['progress', 'animations', 'training', 'skills', 'tools', 'telemetry', 'experiments', 'puppet'];
const timers = new Set();
const every = (ms, fn) => { fn(); const id = setInterval(fn, ms); timers.add(id); return id; };
let currentTab = null;
async function route() {
  const hash = location.hash.slice(1) || 'progress';
  const [tab, ...rest] = hash.split('/');
  const t = tab === 'jobs' ? 'tools' : TABS.includes(tab) ? tab : 'progress';
  for (const id of timers) clearInterval(id);
  timers.clear();
  player.active = false;
  for (const s of $$('.tab')) s.classList.toggle('on', s.id === 'tab-' + t);
  for (const a of $$('#tabs a')) a.classList.toggle('on', a.getAttribute('href') === '#' + t);
  if (currentTab !== t) window.scrollTo(0, 0);
  currentTab = t;
  try { await VIEWS[tab === 'jobs' ? 'jobs' : t](rest.map(x => { try { return decodeURIComponent(x); } catch { return x; } })); }
  catch (e) { console.error(e); $('#tab-' + t).insertAdjacentHTML('afterbegin', `<p class="err">Could not load: ${esc(e.message)}</p>`); }
}
window.addEventListener('hashchange', route);

async function headStatus() {
  const el = $('#hstatus');
  try {
    const s = await api('status');
    const g = s.gpu || {};
    el.innerHTML = [
      g.ok ? `<span class="dot ${g.util > 5 ? 'busy' : 'on'}"></span>GPU ${Math.round(g.util)}% · ${Math.round(g.mem_used)} / ${Math.round(g.mem_total)} MB` : `<span class="dot off"></span>no GPU`,
      s.jobs_running ? `<a href="#tools">${s.jobs_running} job${s.jobs_running > 1 ? 's' : ''} running</a>` : 'no jobs running',
      `<span class="dot ${s.agent ? 'on' : 'off'}"></span>duck ${s.agent ? 'online' : 'offline'}`,
    ].join(' · ');
  } catch { el.innerHTML = '<span class="dot off"></span>server unreachable'; }
}

// ================================================================ Progress
const PSTATUS = ['todo', 'in_progress', 'done', 'blocked'];
const statusSel = (v, cls) => `<select class="${cls}">${PSTATUS.map(s => `<option value="${s}"${s === v ? ' selected' : ''}>${s.replace('_', ' ')}</option>`).join('')}</select>`;
async function progress() {
  const el = $('#tab-progress');
  const p = await api('progress_skill_library');
  const ms = p.milestones || [], ws = p.workstreams || [];
  const teams = [...new Set(ws.map(w => w.team))];
  const done = ms.filter(m => m.status === 'done').length;
  el.innerHTML = `
    <div class="section-head"><h2>Skill library</h2><div class="links">
      <a href="${esc(p.flow_map || 'http://hermes-agent.local/apps/mermaid-flow-studio/?load=diagrams/skill-library-plan.mmd')}" target="_blank" rel="noopener">Plan flow map ↗</a>
      <a href="#" data-doc="plan">Plan</a><a href="#" data-doc="framework">Framework</a>
      <span class="muted">updated ${esc(p.updated ? ago(p.updated) : '–')}</span></div></div>
    <div class="grid wide">
      <div class="card"><div class="card-head"><h2>Milestones</h2><span class="muted small">${done} / ${ms.length} done</span></div>
        ${ms.map(m => `<div class="ms" data-id="${esc(m.id)}"><span class="id">${esc(m.id)}</span><div>
          <div class="t">${esc(m.title)} <span class="muted small">· weeks ${esc(m.weeks || '?')}</span></div>
          <div class="meta">${statusSel(m.status, 'st')} ${statusChip(m.status)} <span class="saved"></span></div>
          <input class="notes" value="${esc(m.notes || '')}" placeholder="notes"></div></div>`).join('') || '<p class="empty">No milestones in tracker/data/progress_skill_library.json yet.</p>'}
      </div>
      <div class="card"><div class="card-head"><h2>Workstreams</h2><span class="muted small">${ws.filter(w => w.status === 'done').length} / ${ws.length} done</span></div>
        ${teams.map(team => `<h3 style="margin-top:10px">${esc(team)}</h3>` + ws.filter(w => w.team === team).map(w => `<div class="ws" data-team="${esc(w.team)}" data-item="${esc(w.item)}">
          <div class="row-top"><span>${esc(w.item)}</span>${statusSel(w.status, 'st')}</div>
          <input class="notes" value="${esc(w.notes || '')}" placeholder="notes"> <span class="saved"></span></div>`).join('')).join('') || '<p class="empty">No workstreams yet.</p>'}
      </div>
    </div>
    <div id="doc-view" class="card" hidden></div>`;
  const save = async (row) => {
    const body = row.classList.contains('ms')
      ? { milestone: { id: row.dataset.id, status: $('.st', row).value, notes: $('.notes', row).value } }
      : { workstream: { team: row.dataset.team, item: row.dataset.item, status: $('.st', row).value, notes: $('.notes', row).value } };
    try { await post('progress_skill_library', body); $('.saved', row).textContent = 'saved'; setTimeout(() => ($('.saved', row).textContent = ''), 1500); if (row.classList.contains('ms')) $('.meta .chip', row).outerHTML = statusChip($('.st', row).value); }
    catch (e) { toast('not saved: ' + e.message, true); }
  };
  el.addEventListener('change', e => { const row = e.target.closest('.ms, .ws'); if (row && e.target.matches('.st, .notes')) save(row); });
  el.addEventListener('keydown', e => { if (e.key === 'Enter' && e.target.matches('.notes')) e.target.blur(); });
  el.addEventListener('click', async e => {
    const a = e.target.closest('[data-doc]'); if (!a) return;
    e.preventDefault();
    const v = $('#doc-view'); v.hidden = false; v.innerHTML = '<p class="muted">Loading…</p>';
    try { v.innerHTML = `<div class="card-head"><h2>${a.dataset.doc === 'plan' ? 'skill-library-plan.md' : 'skill-library-framework.md'}</h2><button class="ghost small" id="doc-close">Close</button></div>` + md(await text('/api/doc?name=' + a.dataset.doc)); }
    catch (err) { v.innerHTML = `<p class="err">${esc(err.message)}</p>`; }
    $('#doc-close')?.addEventListener('click', () => (v.hidden = true));
    v.scrollIntoView({ behavior: 'smooth' });
  });
}

// ================================================================ Animations
const player = { active: false, sampler: null, t: 0, playing: false, last: 0 };
let viewer = null, viewerPromise = null;
function getViewer() {
  if (!viewerPromise) viewerPromise = createDuckViewer($('#anim-canvas'), { loading: $('#anim-loading') }).then(v => (viewer = v)).catch(e => { $('#anim-loading').textContent = 'Model failed to load: ' + e.message; throw e; });
  return viewerPromise;
}
function samplerFor(clip) {
  const frames = Array.isArray(clip) ? clip : clip.frames;
  if (frames && frames.length) {
    const T = frames.map(f => +f[0]), dur = T[T.length - 1];
    return { duration: dur, poseAt(t) {
      if (t <= T[0]) return frames[0][1];
      if (t >= dur) return frames[frames.length - 1][1];
      let lo = 0, hi = T.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
      const u = (t - T[lo]) / ((T[hi] - T[lo]) || 1e-6), a = frames[lo][1], b = frames[hi][1], out = {};
      for (const j in a) out[j] = b[j] == null ? a[j] : a[j] + (b[j] - a[j]) * u;
      return out;
    } };
  }
  const k = A.normalise(clip);
  return { duration: A.duration(k), poseAt: t => A.poseAt(k, t) };
}
function playerTick(now) {
  if (!player.active) return;
  const s = player.sampler;
  if (s && player.playing) {
    player.t += (now - player.last) / 1000 * (+$('#anim-speed').value || 1);
    if (player.t > s.duration) { if ($('#anim-loop').checked) player.t = 0; else { player.t = s.duration; player.playing = false; $('#anim-play').textContent = '▶'; } }
    showFrame();
  }
  player.last = now;
  requestAnimationFrame(playerTick);
}
function showFrame() {
  const s = player.sampler; if (!s) return;
  if (viewer) viewer.pose(s.poseAt(player.t));
  $('#anim-time').textContent = `${player.t.toFixed(2)} / ${s.duration.toFixed(2)} s`;
  if (document.activeElement !== $('#anim-scrub')) $('#anim-scrub').value = Math.round(player.t / (s.duration || 1) * 1000);
}
let animWired = false;
function wireAnim() {
  if (animWired) return; animWired = true;
  $('#anim-play').addEventListener('click', () => { player.playing = !player.playing; $('#anim-play').textContent = player.playing ? '❚❚' : '▶'; if (player.playing && player.sampler && player.t >= player.sampler.duration) player.t = 0; });
  $('#anim-scrub').addEventListener('input', () => { if (!player.sampler) return; player.t = +$('#anim-scrub').value / 1000 * player.sampler.duration; player.playing = false; $('#anim-play').textContent = '▶'; showFrame(); });
  $('#anim-reset').addEventListener('click', () => viewer?.resetView());
  document.addEventListener('keydown', e => { if (e.key === ' ' && currentTab === 'animations' && !$('#anim-detail').hidden && !e.target.matches('input,select,textarea,button')) { e.preventDefault(); $('#anim-play').click(); } });
}
function reportCard(c) {
  const r = c.report;
  if (!r) return `<div class="card"><div class="card-head"><h2>Stage 2 check</h2>${passChip(null)}</div><p class="muted small">No report at <code>anim/reports/${esc(c.check_name)}/check.json</code>. Run the check below.</p></div>`;
  const peak = Object.entries(r.peak_vel_rad_s || {}).sort((a, b) => b[1] - a[1]);
  const over = r.over_speed || {};
  return `<div class="card"><div class="card-head"><h2>Stage 2 check</h2><span>${passChip(r.pass)} ${c.report_stale ? chip('older than the clip', 'warn') : ''}</span></div>
    ${r.fails?.length ? `<p class="err" style="margin:0 0 8px"><b>Fails:</b> ${esc(r.fails.join(', '))}</p>` : ''}
    ${kv([['floor penetration', `${fmtNum((r.floor_penetration_m ?? 0) * 1000, 1)} mm`], ['self-collisions', Object.keys(r.self_collisions_m || {}).length ? esc(Object.entries(r.self_collisions_m).map(([k, v]) => `${k} ${(v * 1000).toFixed(1)} mm`).join(', ')) : 'none'],
      ['joint limits', Object.keys(r.limit_breach_rad || {}).length ? esc(Object.entries(r.limit_breach_rad).map(([k, v]) => `${k} +${(v * R2D).toFixed(1)}°`).join(', ')) : 'inside'],
      ['start vs home', `${fmtNum(r.start_vs_home_rad * R2D, 1)}°`], ['end vs home', `${fmtNum(r.end_vs_home_rad * R2D, 1)}°`], ['end vs start', r.end_vs_start_rad != null ? `${fmtNum(r.end_vs_start_rad * R2D, 1)}°` : null],
      ['length', `${esc(r.seconds)} s · ${esc(r.frames)} frames`], ['checked', `${esc(ago(c.report_time))} · ${fileLink(c.report_url, 'check.json')}`]], 'tight')}
    <details style="margin-top:8px"><summary>peak joint speed (rad/s)${Object.keys(over).length ? ` · ${Object.keys(over).length} over the limit` : ''}</summary>
      <div class="jlist" style="margin-top:6px">${peak.map(([j, v]) => `<span>${esc(jointText(j))}</span><span class="v ${over[j] != null ? 'over' : ''}">${v.toFixed(2)}</span>`).join('')}</div></details>
  </div>`;
}
async function animations([id]) {
  wireAnim();
  const list = $('#anim-list'), detail = $('#anim-detail');
  const clips = await api('animations');
  if (!id) {
    detail.hidden = true; list.hidden = false;
    list.innerHTML = `<div class="section-head"><h2>Clips</h2><div class="links"><a href="/puppet/">Open the Duck Puppet ↗</a><a href="#" id="anim-blender">Open Blender rig</a></div></div>
      ${clips.length ? `<div class="card pad0 tbl-wrap"><table class="tbl"><thead><tr><th>Clip</th><th class="num">s</th><th class="num">frames</th><th>Kind</th><th>Check</th><th></th></tr></thead><tbody>
      ${clips.map(c => `<tr class="row-link" data-id="${esc(c.path)}"><td><a href="#animations/${encodeURIComponent(c.path)}">${esc(c.name)}</a><span class="sub">${c.source === 'skill' ? 'skill package' : 'tracker/data/clips'}${c.meta?.recorded_from_policy ? ' · recorded from a policy' : c.meta?.authored_in ? ' · ' + esc(c.meta.authored_in.tool || 'authored') : ''}${c.broken ? ' · broken file' : ''}</span></td>
        <td class="num">${(+c.seconds).toFixed(2)}</td><td class="num">${c.frames}${c.keys ? `<span class="sub">${c.keys} keys</span>` : ''}</td><td>${esc(c.kind || '')}${c.has_base ? chip('base', 'info') : ''}</td>
        <td>${passChip(c.report ? c.report.pass : null)}${c.report_stale ? ' ' + chip('stale', 'warn') : ''}${c.video_url ? ' ' + chip('video', 'info') : ''}</td>
        <td class="btns"><button class="ghost small" data-check="${esc(c.path)}">Check</button></td></tr>`).join('')}</tbody></table></div>`
      : '<p class="empty">No clips yet. Record or author one in the Duck Puppet (<code>tracker/data/clips/*.json</code>) or add a skill package with a <code>clip.json</code>.</p>'}`;
    list.onclick = e => {
      const b = e.target.closest('button[data-check]');
      if (b) { e.stopPropagation(); runTool('clip_check', { clip: b.dataset.check }, 'clip check'); return; }
      if (e.target.closest('#anim-blender')) { e.preventDefault(); runTool('blender', {}, 'Blender'); return; }
      const tr = e.target.closest('tr.row-link'); if (tr && !e.target.closest('a')) location.hash = '#animations/' + encodeURIComponent(tr.dataset.id);
    };
    return;
  }
  const c = clips.find(x => x.path === id);
  list.hidden = true; detail.hidden = false;
  if (!c) { $('#anim-info').innerHTML = `<p class="err">No clip at ${esc(id)}.</p>`; return; }
  $('#anim-title').textContent = c.name;
  $('#anim-badge').innerHTML = passChip(c.report ? c.report.pass : null) + (c.report_stale ? ' ' + chip('report older than clip', 'warn') : '');
  const info = $('#anim-info');
  const metaPairs = Object.entries(c.meta || {}).map(([k, v]) => [k.replace(/_/g, ' '), typeof v === 'object' ? `<code>${esc(JSON.stringify(v))}</code>` : esc(String(v))]);
  info.innerHTML = `
    <div class="card"><div class="card-head"><h2>Clip</h2><span class="muted small">${esc(ago(c.saved))}</span></div>
      ${kv([['file', fileLink(c.url, c.path)], ['length', `${(+c.seconds).toFixed(2)} s · ${c.frames} frames${c.keys ? ` · ${c.keys} keys` : ''}`], ['interp', c.interp], ['base track', c.has_base ? 'yes (trunk placed by the exporter)' : 'no (trunk held at home height)'], ['skill', c.skill ? `<a href="#skills/${esc(c.skill)}">${esc(c.skill)}</a>` : null], ...metaPairs], 'tight')}
      <div class="btns" style="margin-top:10px">
        <button class="ghost primary" id="act-check">Check</button>
        <label class="check small"><input type="checkbox" id="act-loop"> as a loop</label>
        <button class="ghost" id="act-blender">Open in Blender</button>
        ${c.source === 'puppet' ? `<a class="ghost btn" href="/puppet/#clip=${encodeURIComponent(c.name)}">Edit in Puppet</a>` : ''}
        <a class="ghost btn" href="${esc(c.url)}?download=1">Download</a>
      </div></div>
    ${reportCard(c)}
    ${c.video_url ? `<div class="card"><div class="card-head"><h2>Check video</h2>${fileLink(c.video_url, 'check.mp4')}</div><video class="inline" controls playsinline preload="metadata" src="${esc(c.video_url)}"></video></div>` : ''}`;
  $('#act-check').addEventListener('click', () => runTool('clip_check', { clip: c.path, loop: $('#act-loop').checked }, 'clip check'));
  $('#act-blender').addEventListener('click', () => runTool('blender', { clip: c.path }, 'Blender'));
  // the player
  const clip = await (await fetch(c.url, { cache: 'no-store' })).json();
  player.sampler = samplerFor(clip); player.t = 0; player.playing = true; player.last = performance.now(); player.active = true;
  $('#anim-play').textContent = '❚❚';
  requestAnimationFrame(playerTick);
  getViewer().then(showFrame).catch(() => {});
}

// ================================================================ Training
const PANELS = [  // (title, tag, per-step)  the same set as sim/make_curriculum_plots.py
  ['Eval reward per step', 'eval/episode_reward', true], ['Episode length (1000 = never fell)', 'eval/avg_episode_length', false],
  ['Foot lift while swinging', 'eval/episode_swing_peak', true], ['Imitation of the reference gait', 'eval/episode_reward/imitation', true],
  ['Speed tracking', 'eval/episode_reward/tracking_lin_vel', true], ['Turn tracking', 'eval/episode_reward/tracking_ang_vel', true],
  ['Action change per step (cost)', 'eval/episode_cost/action_rate', true], ['Both feet off the floor', 'eval/episode_scaffold/airborne', true],
  ['Policy loss', 'training/policy_loss', false], ['Value loss', 'training/v_loss', false],
];
const steps = v => v === 0 ? '0' : (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + 'M';
function gpuCard(g, running) {
  if (!g || !g.ok) return `<div class="card"><div class="card-head"><h2>GPU</h2>${chip('unavailable', 'warn')}</div><p class="muted small">${esc(g?.error || 'nvidia-smi not found')}</p></div>`;
  const memPct = g.mem_total ? g.mem_used / g.mem_total * 100 : 0;
  return `<div class="card"><div class="card-head"><h2>${esc(g.name || 'GPU')}</h2><span class="muted small">${g.temp != null ? Math.round(g.temp) + ' °C' : ''}</span></div>
    <div class="hero"><div class="n"><span class="v">${Math.round(g.util)}%</span><span class="l">utilisation</span><div class="bar ${g.util > 80 ? 'hot' : ''}" style="width:120px"><span style="width:${g.util}%"></span></div></div>
      <div class="n"><span class="v">${(g.mem_used / 1024).toFixed(1)} / ${(g.mem_total / 1024).toFixed(0)} GB</span><span class="l">memory</span><div class="bar ${memPct > 85 ? 'hot' : ''}" style="width:120px"><span style="width:${memPct}%"></span></div></div></div>
    <p class="muted small" style="margin:8px 0 0">${g.procs?.length ? 'on the GPU: ' + g.procs.map(p => `${esc(p.name)} (${esc(p.pid)}, ${typeof p.mem === 'number' ? Math.round(p.mem) + ' MB' : esc(p.mem)})`).join(', ') : 'nothing is using the GPU'}${running ? ` · ${chip('a GPU job is running', 'info')}` : ''}</p></div>`;
}
function jobRow(j) {
  return `<div class="job-row"><a href="#jobs/${esc(j.id)}">${esc(j.tool_name)}${j.gpu ? ' ' + chip('GPU', 'info') : ''}</a><span>${statusChip(j.status)}${j.exit != null && j.exit !== 0 ? ` <span class="muted small">exit ${j.exit}</span>` : ''}</span>
    <span class="sub">${esc(when(j.started))}${j.ended ? ` · ${esc(ago(j.started))}` : ' · running'} · ${esc(Object.entries(j.args || {}).filter(([k, v]) => v).map(([k, v]) => `${k}=${String(v).split('/').pop()}`).join(' '))}</span></div>`;
}
async function training([rd, run]) {
  const main = $('#training-main'), page = $('#run-page');
  if (rd && run) { main.hidden = true; page.hidden = false; return runPage(rd, run); }
  page.hidden = true; main.hidden = false;
  const [runs, media, jobs] = await Promise.all([api('runs'), api('reports'), api('jobs')]);
  const running = jobs.filter(j => !j.ended);
  main.innerHTML = `
    <div class="grid wide" id="train-top"><div id="gpu-card"></div>
      <div class="card"><div class="card-head"><h2>Running jobs</h2><a href="#tools" class="small">all jobs</a></div><div id="running-jobs">${running.length ? running.map(jobRow).join('') : '<p class="muted small" style="margin:0">Nothing running. Start a run from <a href="#tools">Tools</a>.</p>'}</div>
        <div id="running-logs"></div></div></div>
    <div class="section-head"><h2>Runs</h2><span class="muted small">${runs.length} in sim/train_pg/runs*</span></div>
    <div class="card pad0 runs">${runs.map(r => `<div class="run-row row-link" data-id="${esc(r.id)}">
        <div class="rn"><a href="#training/${esc(r.runs_dir)}/${esc(r.name)}">${esc(r.name)}</a> <span class="muted small">${esc(r.runs_dir)}</span>${r.running ? ' ' + chip('live', 'info') : ''}<span class="sub">${esc(when(r.mtime))} · ${esc(ago(r.mtime))}${r.onnx.length > 1 ? ` · ${r.onnx.length} checkpoints` : ''}</span></div>
        <div class="rs"><span class="v">${r.reward != null ? r.reward.toFixed(1) : '–'}</span><span class="sub">reward · ${r.steps != null ? steps(r.steps) : '–'}</span></div>
        <div class="rsc small">${r.scores.length ? r.scores.map(s => s.json ? chip('json score') : [s.pushes ? `pushes ${s.pushes}` : '', s.falls || '', s.speed_fwd != null ? `fwd ${s.speed_fwd.toFixed(2)} m/s` : ''].filter(Boolean).join(' · ') || chip('score')).join('<br>') : '<span class="muted">not scored</span>'}</div>
        <div class="ro">${r.onnx.length ? `<a class="ghost btn small" href="${esc(r.onnx[0].url)}?download=1" title="${esc(r.onnx[0].file)}">ONNX ${steps(r.onnx[0].step)} ↓</a>` : ''}</div></div>`).join('') || '<p class="empty">No runs under sim/train_pg/runs*/.</p>'}</div>
    <div class="section-head"><h2>Report media</h2><div class="btns"><input class="filter" id="media-filter" placeholder="filter by name" aria-label="filter media"><span class="muted small">${media.length} files in sim/report*</span></div></div>
    <div id="media-grid"></div>`;
  main.onclick = e => { const tr = e.target.closest('.row-link'); if (tr && !e.target.closest('a')) location.hash = '#training/' + tr.dataset.id; };
  const vis = media.filter(m => m.kind === 'mp4' || m.kind === 'png');
  const drawMedia = () => { const f = $('#media-filter').value.toLowerCase(); const items = vis.filter(m => m.name.toLowerCase().includes(f)); $('#media-grid').innerHTML = mediaGrid(items.slice(0, 24)) + (items.length > 24 ? `<p class="muted small">${items.length - 24} more match; narrow the filter.</p>` : '') || '<p class="empty">No mp4/png in sim/report*/.</p>'; };
  $('#media-filter').addEventListener('input', drawMedia); drawMedia();
  every(3000, async () => { try { const s = await api('status'); $('#gpu-card').innerHTML = gpuCard(s.gpu, s.gpu_busy); } catch { /* keep */ } });
  const tailLogs = async () => {
    const live = (await api('jobs')).filter(j => !j.ended);
    if (!live.length) { $('#running-logs').innerHTML = ''; return; }
    const parts = await Promise.all(live.map(async j => `<h3 style="margin-top:10px"><a href="#jobs/${esc(j.id)}">${esc(j.tool_name)}</a> <span class="muted small">${esc(j.id)}</span></h3><pre class="log short">${esc((await text(`/api/jobs/${j.id}/log?tail=3000`)).trimEnd())}</pre>`));
    $('#running-logs').innerHTML = parts.join('');
    for (const p of $$('#running-logs pre')) p.scrollTop = p.scrollHeight;
  };
  every(2500, tailLogs);
}
async function runPage(rd, run) {
  const page = $('#run-page');
  page.innerHTML = '<p class="muted">Loading…</p>';
  const r = await api(`runs/${encodeURIComponent(rd)}/${encodeURIComponent(run)}`);
  const newest = r.onnx[0];
  const curves = r.curves || {}, len = curves['eval/avg_episode_length'] || [];
  const lenAt = new Map(len.map(([s, v]) => [s, v]));
  const series = (tag, perStep) => { const pts = curves[tag] || []; return [{ name: tag, points: pts.map(([s, v]) => [s, perStep && $('#per-step')?.checked !== false ? v / (lenAt.get(s) || 1) : v]) }]; };
  const extra = Object.keys(curves).filter(t => !PANELS.some(p => p[1] === t)).sort();
  page.innerHTML = `
    <div class="detail-head"><a href="#training" class="back">‹ All runs</a><h2>${esc(r.name)}</h2><span class="muted">${esc(r.runs_dir)} · ${esc(when(r.mtime))}${r.running ? ' ' + chip('live', 'info') : ''}</span></div>
    <div class="grid wide">
      <div class="card"><div class="card-head"><h2>Summary</h2></div>
        <div class="hero"><div class="n"><span class="v">${r.steps != null ? steps(r.steps) : '–'}</span><span class="l">steps</span></div><div class="n"><span class="v">${r.reward != null ? r.reward.toFixed(1) : '–'}</span><span class="l">eval reward${r.reward_std != null ? ` ± ${r.reward_std.toFixed(1)}` : ''}</span></div><div class="n"><span class="v">${r.onnx.length}</span><span class="l">checkpoints</span></div></div>
        <div class="btns" style="margin-top:10px">${newest ? `<a class="ghost btn primary" href="${esc(newest.url)}?download=1">Newest ONNX ↓</a><button class="ghost" id="act-score">Score it</button><button class="ghost" id="act-render">Render video</button>` : ''}${r.log_url ? fileLink(r.log_url, 'full log') : ''}</div>
        ${r.onnx.length > 1 ? `<details style="margin-top:8px"><summary>all checkpoints</summary><div class="links" style="margin-top:6px">${r.onnx.map(o => `<a href="${esc(o.url)}?download=1">${steps(o.step)}</a>`).join('')}</div></details>` : ''}</div>
      ${r.scores.map(s => `<div class="card"><div class="card-head"><h2>${esc(s.file)}</h2>${fileLink(s.url, 'file')}</div>${s.json ? pre(s.json) : `<pre class="log short">${esc((s.lines || []).join('\n') || s.text)}</pre>`}${s.lines?.length && s.text ? `<details><summary>whole file</summary><pre class="log">${esc(s.text)}</pre></details>` : ''}</div>`).join('')}
    </div>
    <div class="section-head"><h2>Curves</h2><div class="btns"><label class="check small"><input type="checkbox" id="per-step" checked> episode sums ÷ episode length</label>
      ${extra.length ? `<select class="sel" id="extra-tag" style="width:auto"><option value="">+ another tag…</option>${extra.map(t => `<option>${esc(t)}</option>`).join('')}</select>` : ''}</div></div>
    ${Object.keys(curves).length ? '<div class="charts" id="run-charts"></div>' : '<p class="empty">No tfevents in this run folder.</p>'}
    <div class="section-head"><h2>Log tail</h2>${r.log_url ? fileLink(r.log_url, r.name + '.log') : ''}</div>
    <pre class="log" id="run-log">${esc(r.log_tail || 'no log')}</pre>
    ${r.media.length ? `<div class="section-head"><h2>Report media for “${esc(r.stage)}”</h2><span class="muted small">sim/report*</span></div>${mediaGrid(r.media)}` : ''}`;
  const drawCharts = () => {
    const grid = $('#run-charts'); if (!grid) return;
    grid.innerHTML = '';
    const shown = PANELS.filter(p => curves[p[1]]).concat(($('#extra-tag')?.value ? [[$('#extra-tag').value, $('#extra-tag').value, false]] : []));
    for (const [title, tag, perStep] of shown) { const el = document.createElement('div'); grid.appendChild(el); chart(el, { title, series: series(tag, perStep), xFmt: steps, yFmt: v => fmtNum(v, 3) }); }
  };
  drawCharts();
  $('#per-step')?.addEventListener('change', drawCharts);
  $('#extra-tag')?.addEventListener('change', drawCharts);
  $('#act-score')?.addEventListener('click', () => runTool('eval_policy', { onnx: `sim/train_pg/${r.runs_dir}/${r.name}/${newest.file}` }, 'scoring'));
  $('#act-render')?.addEventListener('click', () => runTool('render_video', { out: `report2/dash_${r.stage}.mp4`, spec: `${r.stage}=train_pg/${r.runs_dir}/${r.name}/${newest.file}:lpf=8:vel=4` }, 'render'));
  const log = $('#run-log'); log.scrollTop = log.scrollHeight;
  if (r.running) every(5000, async () => { try { const d = await api(`runs/${encodeURIComponent(rd)}/${encodeURIComponent(run)}`); log.textContent = d.log_tail; log.scrollTop = log.scrollHeight; } catch { /* keep */ } });
}

// ================================================================ Skills
const ratioChip = r => r == null ? chip('–') : chip(r.text, r.value >= 0.9 ? 'good' : r.value >= 0.5 ? 'warn' : 'bad');
async function skills([name, ver]) {
  const main = $('#skills-main'), page = $('#skill-page');
  if (name && ver) { main.hidden = true; page.hidden = false; return skillPage(name, ver); }
  page.hidden = true; main.hidden = false;
  const list = await api('skills');
  main.innerHTML = `<div class="section-head"><h2>Skill packages</h2><span class="muted small">skills/&lt;name&gt;/v&lt;N&gt;/</span></div>
    ${list.length ? `<div class="card pad0 tbl-wrap"><table class="tbl"><thead><tr><th>Skill</th><th>Type</th><th>Battery</th><th>Sim-to-sim</th><th>Hardware</th><th>Parts</th></tr></thead><tbody>
      ${list.map(s => `<tr class="row-link" data-id="${esc(s.id)}"><td><a href="#skills/${esc(s.id)}">${esc(s.name)}</a> <span class="muted">${esc(s.version)}</span><span class="sub">${esc(s.created ? when(s.created) : ago(s.mtime))}</span></td><td>${esc(s.type || '–')}</td>
        <td>${ratioChip(s.battery)}</td><td>${s.sim2sim ? statusChip(s.sim2sim) : chip('–')}</td><td>${ratioChip(s.hardware)}</td>
        <td>${['skill.json', 'clip.json', 'policy.onnx'].map(p => chip(p.split('.')[0], s.parts[p] ? 'good' : 'bad')).join(' ')}</td></tr>`).join('')}</tbody></table></div>`
      : `<p class="empty">No packages yet. A skill lands as <code>skills/&lt;name&gt;/v&lt;N&gt;/{skill.json, clip.json, policy.onnx, report/}</code>; the first one (<b>bow</b>) is being built by the Controls and Software teams. The Blender rig and the clip checker are ready in <a href="#animations">Animations</a>.</p>`}`;
  main.onclick = e => { const tr = e.target.closest('tr.row-link'); if (tr && !e.target.closest('a')) location.hash = '#skills/' + tr.dataset.id; };
}
async function skillPage(name, ver) {
  const page = $('#skill-page');
  page.innerHTML = '<p class="muted">Loading…</p>';
  const s = await api(`skills/${encodeURIComponent(name)}/${encodeURIComponent(ver)}`);
  const sj = s.skill || {};
  const known = ['name', 'type', 'version', 'commands', 'entry', 'exit', 'neighbours', 'timeout_s', 'envelope', 'robot_model_hash', 'recipe_version', 'seed', 'created', 'source'];
  const val = v => v == null ? null : typeof v === 'object' ? `<code>${esc(JSON.stringify(v))}</code>` : esc(String(v));
  page.innerHTML = `
    <div class="detail-head"><a href="#skills" class="back">‹ All skills</a><h2>${esc(s.name)} <span class="muted">${esc(s.version)}</span></h2>${s.type ? chip(s.type, 'info') : ''}${s.complete ? chip('complete', 'good') : chip('missing parts', 'bad')}</div>
    <div class="grid wide">
      <div class="card"><div class="card-head"><h2>skill.json</h2><span class="muted small">${esc(s.path)}</span></div>
        ${s.parts['skill.json'] ? kv(known.map(k => [k.replace(/_/g, ' '), val(sj[k])]).concat(Object.keys(sj).filter(k => !known.includes(k)).map(k => [k, val(sj[k])])), 'tight') : '<p class="muted small">missing</p>'}
        <div class="btns" style="margin-top:10px">${s.clip ? `<a class="ghost btn primary" href="#animations/${encodeURIComponent(s.clip.path)}">Play the clip</a><button class="ghost" id="act-check">Check the clip</button>` : ''}${s.onnx_url ? `<a class="ghost btn" href="${esc(s.onnx_url)}?download=1">policy.onnx ↓</a>` : ''}<button class="ghost" id="act-battery">Run the battery</button></div></div>
      <div class="card"><div class="card-head"><h2>Gates</h2></div>${kv([['battery (G1)', ratioChip(s.battery)], ['sim-to-sim (G2)', s.sim2sim ? statusChip(s.sim2sim) : chip('–')], ['hardware (G3/G4)', ratioChip(s.hardware)], ['clip check (stage 2)', s.clip ? passChip(s.clip.report ? s.clip.report.pass : null) : chip('–')]])}
        <details style="margin-top:8px" ${s.reports?.battery ? 'open' : ''}><summary>battery.json</summary>${s.reports?.battery ? pre(s.reports.battery) : '<p class="muted small">none</p>'}</details>
        <details><summary>sim2sim.json</summary>${s.reports?.sim2sim ? pre(s.reports.sim2sim) : '<p class="muted small">none</p>'}</details>
        <details><summary>hardware.json</summary>${s.reports?.hardware ? pre(s.reports.hardware) : '<p class="muted small">none</p>'}</details></div>
    </div>
    ${s.media?.length ? `<div class="section-head"><h2>Report media</h2></div>${mediaGrid(s.media)}` : ''}
    <details style="margin-top:12px"><summary>all files (${s.files.length})</summary><ul class="small">${s.files.map(f => `<li>${fileLink('/files/' + f, f)}</li>`).join('')}</ul></details>`;
  $('#act-check')?.addEventListener('click', () => runTool('clip_check', { clip: s.clip.path }, 'clip check'));
  $('#act-battery')?.addEventListener('click', () => runTool('battery', { skill_dir: s.path }, 'battery'));
}

// ================================================================ Tools and jobs
function field(tool, a) {
  const id = `f-${tool.id}-${a.name}`, label = a.label || a.name;
  if (a.type === 'derived') return '';
  if (a.type === 'bool') return `<label class="check"><input type="checkbox" id="${id}" name="${esc(a.name)}" ${a.default ? 'checked' : ''}> ${esc(label)}</label>`;
  if (a.type === 'choice') return `<label>${esc(label)}<select id="${id}" name="${esc(a.name)}">${(a.choices || []).map(c => `<option ${String(c) === String(a.default) ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></label>`;
  if (a.type === 'file') {
    const opts = a.options || [];
    return `<label>${esc(label)}<select id="${id}" name="${esc(a.name)}" ${!opts.length && !a.optional ? 'disabled' : ''}>${a.optional ? '<option value="">(none)</option>' : ''}${opts.map(o => `<option value="${esc(o)}">${esc(o)}</option>`).join('')}${!opts.length ? `<option value="">nothing matches ${esc(a.glob)}</option>` : ''}</select></label>`;
  }
  return `<label>${esc(label)}<input type="${a.type === 'number' ? 'number' : 'text'}" id="${id}" name="${esc(a.name)}" value="${esc(a.default ?? '')}" ${a.type === 'number' ? 'step="any"' : ''}></label>`;
}
async function tools() {
  const main = $('#tools-main'), page = $('#job-page');
  page.hidden = true; main.hidden = false;
  const [tls, jobs, status] = await Promise.all([api('tools'), api('jobs'), api('status')]);
  const groups = [...new Set(tls.map(t => t.group))];
  main.innerHTML = `
    <div class="section-head"><h2>Jobs</h2><span class="muted small">${jobs.filter(j => !j.ended).length} running · ${jobs.length} total · logs in tracker/data/jobs</span></div>
    <div class="card" id="jobs-list">${jobs.length ? jobs.slice(0, 15).map(jobRow).join('') : '<p class="muted small" style="margin:0">No jobs yet.</p>'}</div>
    ${groups.map(g => `<div class="section-head"><h2>${esc(g)}</h2></div><div class="grid wide">${tls.filter(t => t.group === g).map(t => `
      <form class="card form" data-tool="${esc(t.id)}"><div class="card-head"><h2>${esc(t.name)}</h2><span>${t.gpu ? chip('GPU', 'info') : ''} <span class="muted small mono">${esc(t.id)}</span></span></div>
        ${t.help ? `<p class="help">${esc(t.help)}</p>` : ''}
        ${(t.args || []).map(a => field(t, a)).join('')}
        <div class="actions"><button class="ghost primary" type="submit" ${t.gpu && status.gpu_busy ? 'disabled title="a GPU job is running"' : ''}>Run</button><span class="muted small mono">${esc(t.cwd || '.')}$ ${esc(t.command.join(' '))}</span></div></form>`).join('')}</div>`).join('')}`;
  main.onsubmit = async e => {
    e.preventDefault();
    const f = e.target, tool = tls.find(t => t.id === f.dataset.tool), args = {};
    for (const a of tool.args || []) { const inp = f.elements[a.name]; if (!inp) continue; args[a.name] = a.type === 'bool' ? inp.checked : inp.value; }
    if (tool.gpu && !confirm(`Start ${tool.name} on the GPU? It can take hours.`)) return;
    runTool(tool.id, args, tool.name);
  };
  every(4000, async () => { try { const js = await api('jobs'); $('#jobs-list').innerHTML = js.length ? js.slice(0, 15).map(jobRow).join('') : '<p class="muted small" style="margin:0">No jobs yet.</p>'; } catch { /* keep */ } });
}
async function jobs([id]) {
  const main = $('#tools-main'), page = $('#job-page');
  if (!id) return tools();
  main.hidden = true; page.hidden = false;
  page.innerHTML = '<p class="muted">Loading…</p>';
  let j = await api('jobs/' + encodeURIComponent(id));
  const render = () => {
    const running = !j.ended;
    page.innerHTML = `
      <div class="detail-head"><a href="#tools" class="back">‹ Tools</a><h2>${esc(j.tool_name)}</h2>${statusChip(j.status)}${j.gpu ? chip('GPU', 'info') : ''}${running ? '<button class="ghost danger small" id="job-kill">Kill</button>' : ''}</div>
      <div class="grid wide"><div class="card">${kv([['job', `<span class="mono">${esc(j.id)}</span>`], ['started', `${esc(when(j.started))} (${esc(ago(j.started))})`], ['ended', j.ended ? `${esc(when(j.ended))}${j.seconds != null ? ` · ${fmtNum(j.seconds, 0)} s` : ''}` : 'running'], ['exit code', j.exit], ['working dir', `<span class="mono small">${esc(j.cwd)}</span>`],
        ['command', `<span class="mono small">${esc(Object.entries(j.env || {}).map(([k, v]) => `${k}=${v} `).join(''))}${esc(j.command_str)}</span>`], ['arguments', Object.entries(j.args || {}).filter(([, v]) => v).map(([k, v]) => `<span class="chip">${esc(k)}=${esc(String(v).replace(/^\/home\/[^/]+\/src\/mini-duck-droid\//, ''))}</span>`).join(' ') || null]], 'tight')}</div>
      <div class="card"><div class="card-head"><h2>Output files</h2><span class="muted small">${(j.files || []).length}</span></div><div id="job-files">${(j.files || []).length ? `<ul class="small" style="margin:0;padding-left:18px">${j.files.map(f => `<li>${f.url ? fileLink(f.url, f.path) : esc(f.path)} <span class="muted">${bytes(f.size)}</span></li>`).join('')}</ul>${mediaGrid(j.files.filter(f => f.url && /\.(mp4|png|jpg)$/i.test(f.path)).map(f => ({ url: f.url, name: f.path.split('/').pop(), kind: f.path.split('.').pop().toLowerCase() })))}` : '<p class="muted small" style="margin:0">none yet</p>'}</div></div></div>
      <div class="section-head"><h2>Log</h2><label class="check small"><input type="checkbox" id="job-follow" checked> follow</label></div>
      <pre class="log" id="job-log"></pre>`;
    $('#job-kill')?.addEventListener('click', async () => { if (!confirm('Stop this job?')) return; try { toast((await post(`jobs/${j.id}/kill`, {})).msg); } catch (e) { toast(e.message, true); } });
  };
  render();
  const refresh = async () => {
    try {
      const log = $('#job-log'); if (!log) return;
      const t = await text(`/api/jobs/${encodeURIComponent(id)}/log?tail=65536`);
      if (log.textContent !== t) { log.textContent = t; if ($('#job-follow')?.checked) log.scrollTop = log.scrollHeight; }
      const was = j.ended, fresh = await api('jobs/' + encodeURIComponent(id));
      if (!was && fresh.ended) { j = fresh; render(); $('#job-log').textContent = t; $('#job-log').scrollTop = $('#job-log').scrollHeight; }
      else if ((fresh.files || []).length !== (j.files || []).length) { j = fresh; const keep = $('#job-log').textContent; render(); $('#job-log').textContent = keep; }
      if (j.ended) for (const id2 of timers) clearInterval(id2);
    } catch { /* keep */ }
  };
  every(1500, refresh);
}

// ================================================================ Telemetry
const GROUPS = { 'Left leg': JOINTS.slice(0, 5), 'Head': JOINTS.slice(5, 9), 'Right leg': JOINTS.slice(9) };
const METRICS = { voltage: ['Voltage', 'V'], temps: ['Temperature', '°C'], currents: ['Current', 'A'] };
async function telemetry() {
  const el = $('#tab-telemetry');
  const days = +(sessionStorage.getItem('tele-days') || 3), metric = sessionStorage.getItem('tele-metric') || 'temps';
  const d = await api('telemetry?days=' + days);
  const last = d.last;
  const lastCard = last ? (() => {
    const v = Object.values(last.voltage || {}), t = Object.entries(last.temps || {}), c = Object.entries(last.currents || {});
    const hot = t.sort((a, b) => b[1] - a[1])[0], amp = c.sort((a, b) => b[1] - a[1])[0];
    return `<div class="card"><div class="card-head"><h2>Last reading</h2><span class="muted small">${esc(when(last.t))} · ${esc(ago(last.t))} · ${esc(last.source || '')}</span></div>
      <div class="hero">${v.length ? `<div class="n"><span class="v">${Math.min(...v).toFixed(2)}–${Math.max(...v).toFixed(2)} V</span><span class="l">bus voltage across servos</span></div>` : ''}
        ${hot ? `<div class="n"><span class="v">${hot[1].toFixed(0)} °C</span><span class="l">hottest: ${esc(jointText(hot[0]))}</span></div>` : ''}
        ${amp ? `<div class="n"><span class="v">${hot ? amp[1].toFixed(2) : '–'} A</span><span class="l">highest current: ${esc(jointText(amp[0]))}</span></div>` : ''}</div>
      <details style="margin-top:8px"><summary>every joint</summary><div class="tbl-wrap"><table class="tbl"><thead><tr><th>joint</th><th class="num">V</th><th class="num">°C</th><th class="num">A</th><th class="num">pos °</th></tr></thead><tbody>
        ${JOINTS.map(j => `<tr><td>${esc(jointText(j))}</td><td class="num">${fmtNum(last.voltage?.[j])}</td><td class="num">${fmtNum(last.temps?.[j], 0)}</td><td class="num">${fmtNum(last.currents?.[j], 2)}</td><td class="num">${last.positions?.[j] != null ? (last.positions[j] * R2D).toFixed(1) : '–'}</td></tr>`).join('')}</tbody></table></div></details></div>`;
  })() : '';
  el.innerHTML = `<div class="section-head"><h2>Servo telemetry</h2><div class="btns">
      <select class="sel" id="tele-metric" style="width:auto">${Object.entries(METRICS).map(([k, [l]]) => `<option value="${k}" ${k === metric ? 'selected' : ''}>${l}</option>`).join('')}</select>
      <select class="sel" id="tele-days" style="width:auto">${[1, 3, 7, 30].map(n => `<option value="${n}" ${n === days ? 'selected' : ''}>last ${n} day${n > 1 ? 's' : ''}</option>`).join('')}</select>
      <button class="ghost small" id="tele-pull">Pull now</button></div></div>
    ${lastCard}
    ${d.count ? `<p class="muted small">${d.count} readings from ${d.files.map(f => f.split('/').pop()).join(', ')}</p><div class="charts" id="tele-charts"></div>`
      : `<p class="empty">No telemetry yet. Readings appear here once <code>mechatronics/telemetry/YYYY-MM-DD.jsonl</code> exists (one JSON per line with <code>voltage</code>, <code>temps</code>, <code>currents</code>, <code>positions</code> per joint). The Mechatronics team's <code>telemetry_pull.sh</code> writes it; “Pull now” runs it.</p>`}`;
  $('#tele-metric').addEventListener('change', e => { sessionStorage.setItem('tele-metric', e.target.value); telemetry(); });
  $('#tele-days').addEventListener('change', e => { sessionStorage.setItem('tele-days', e.target.value); telemetry(); });
  $('#tele-pull').addEventListener('click', () => runTool('servo_health', {}, 'telemetry pull'));
  if (!d.count) return;
  const T = d.series.t.map(t => Date.parse(t)), span = (T[T.length - 1] - T[0]) / 3.6e6;
  const xFmt = ms => { const dt = new Date(ms); return span > 30 ? dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + dt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : dt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }); };
  const grid = $('#tele-charts'), [label, unit] = METRICS[metric], data = d.series[metric] || {};
  for (const [g, joints] of Object.entries(GROUPS)) {
    const el2 = document.createElement('div'); grid.appendChild(el2);
    chart(el2, { title: `${label} · ${g}`, unit, xFmt, yFmt: v => fmtNum(v, metric === 'temps' ? 0 : 2), series: joints.map((j, i) => ({ name: jointText(j), slot: i, points: (data[j] || []).map((v, k) => [T[k], v]) })) });
  }
}

// ================================================================ Experiments
async function experiments() {
  const el = $('#tab-experiments');
  const d = await api('experiments');
  const which = sessionStorage.getItem('exp-tab') || 'log';
  el.innerHTML = `<div class="section-head"><div class="seg" id="exp-seg"><button data-w="log" class="${which === 'log' ? 'on' : ''}">Log (${d.entries.length})</button><button data-w="questions" class="${which === 'questions' ? 'on' : ''}">Questions</button></div>
      <span class="muted small">experiments/log.md · experiments/questions.md</span></div>
    <div class="card" id="exp-body"></div>`;
  const body = $('#exp-body');
  const draw = w => {
    for (const b of $$('#exp-seg button')) b.classList.toggle('on', b.dataset.w === w);
    if (w === 'log') body.innerHTML = d.entries.length ? (d.intro ? md(d.intro) : '') + d.entries.map(e => `<div class="entry"><h3>${inline(e.title)}</h3>${md(e.body)}</div>`).join('')
      : `<p class="empty">${d.log_exists ? 'The log has no entries yet.' : 'No <code>experiments/log.md</code> yet.'} Entries are <code>## &lt;ISO datetime&gt; &lt;title&gt;</code> followed by hypothesis, command, evidence and decision bullets; newest is shown first.</p>`;
    else body.innerHTML = d.questions.trim() ? md(d.questions) : '<p class="empty">No <code>experiments/questions.md</code> yet. Teams leave questions for each other there under a heading per team.</p>';
  };
  draw(which);
  $('#exp-seg').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; sessionStorage.setItem('exp-tab', b.dataset.w); draw(b.dataset.w); });
}

// ================================================================ Puppet
async function puppet() {
  const el = $('#tab-puppet');
  el.innerHTML = `<div class="grid wide">
    <div class="card"><div class="card-head"><h2>Duck Puppet</h2><span id="duck-state" class="muted small">checking…</span></div>
      <p class="small" style="margin:0 0 10px">Pose, hold, record and replay the real duck's 14 servos, and edit keyframe clips on the 3D model. It is its own full page so the timeline has room.</p>
      <div class="btns"><a class="ghost btn primary" href="/puppet/">Open the Duck Puppet</a><a class="ghost btn" href="/build/">Build tracker</a></div>
      <div id="duck-joints" style="margin-top:10px"></div></div>
    <div class="card"><div class="card-head"><h2>Links</h2></div><div class="links" style="flex-direction:column">
      <a href="http://hermes-agent.local/apps/mermaid-flow-studio/?load=diagrams/skill-library-plan.mmd" target="_blank" rel="noopener">Skill library plan flow map ↗</a>
      <a href="http://hermes-agent.local/" target="_blank" rel="noopener">Hermes Command Center ↗</a>
      <a href="/files/tracker/data/tools.json" target="_blank" rel="noopener">Tool registry (tools.json)</a>
      <a href="/files/tracker/data/progress_skill_library.json" target="_blank" rel="noopener">Progress JSON</a></div></div></div>`;
  every(2000, async () => {
    const st = $('#duck-state'), jl = $('#duck-joints'); if (!st) return;
    try {
      const r = await fetch('/duck/api/state', { cache: 'no-store' }); const j = await r.json();
      if (j.error) { st.innerHTML = '<span class="dot off"></span>duck agent not running (the Pi may be off)'; jl.innerHTML = ''; return; }
      const holding = j.joints.filter(x => x.torque).length, bad = j.joints.filter(x => !x.ok).length;
      st.innerHTML = `<span class="dot on"></span>${esc(j.volts ?? '–')} V · ${esc(j.amps)} A · ${holding}/14 holding${bad ? ` · ${bad} not answering` : ''}${j.mode && j.mode !== 'idle' ? ' · ' + esc(j.mode) : ''}`;
      jl.innerHTML = `<div class="jlist">${j.joints.map(x => `<span>${esc(jointText(x.name))}${x.torque ? ' ' + chip('hold', 'good') : ''}</span><span class="v">${x.rad != null ? (x.rad * R2D).toFixed(1) + '°' : '–'} · ${esc(x.temp)} °C</span>`).join('')}</div>`;
    } catch { st.innerHTML = '<span class="dot off"></span>no answer'; }
  });
}

const VIEWS = { progress, animations, training, skills, tools, jobs, telemetry, experiments, puppet };
headStatus(); setInterval(headStatus, 5000);
route();
