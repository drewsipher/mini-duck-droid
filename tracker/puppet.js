// Duck Puppet: pose, hold, record, edit and replay the real duck. Talks to tracker/server.py,
// which proxies /duck/api/* to duck_agent.py on the Pi and keeps clips in data/clips/.
import * as THREE from 'three';
import { STLLoader } from 'three/addons/STLLoader.js';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import * as A from './anim.js';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const R2D = 180 / Math.PI, D2R = Math.PI / 180;
const deg = r => r == null ? '–' : (r * R2D).toFixed(1) + '°';
const JOINTS = A.JOINTS;
const SERVO_IDS = { right_hip_yaw: 10, right_hip_roll: 11, right_hip_pitch: 12, right_knee: 13, right_ankle: 14,
  left_hip_yaw: 20, left_hip_roll: 21, left_hip_pitch: 22, left_knee: 23, left_ankle: 24,
  neck_pitch: 30, head_pitch: 31, head_yaw: 32, head_roll: 33 };
const SERVO_KEY = 'buy:servo_sts3215';
const SHAFT = [-0.0255, 0.0094, 0];  // output shaft in the STS3215 model's own frame
const LIMITS_DEFAULT = { hip_yaw: [-30, 30], hip_roll: [-25, 25], left_hip_pitch: [-70, 30], right_hip_pitch: [-30, 70],
  knee: [-90, 90], ankle: [-90, 90], neck_pitch: [-20, 65], head_pitch: [-45, 45], head_yaw: [-70, 70], head_roll: [-30, 30] };
const jointText = j => j.replace(/_/g, ' ');

let duck = null, agentOk = false;
let sel = null;                 // selected joint (side panel and timeline)
let source = 'live';            // what the 3D model shows
let clip = null, clipName = null, clipDirty = false;
let playhead = 0, playing = false, lastTick = 0;
const selection = new Set();    // selected key objects (the ones inside clip.keys)
let clipboard = [];             // [{joint, t, v}] with t relative to the earliest copied key
let box = null;                 // shift-drag rectangle while selecting
const history = [];
const jointOfKey = k => JOINTS.find(j => clip.keys[j].includes(k));
function oneSelected() { if (selection.size !== 1) return null; const k = [...selection][0], j = jointOfKey(k); return j ? { joint: j, i: clip.keys[j].indexOf(k), key: k } : null; }
function selectOnly(j, i) { selection.clear(); const k = clip?.keys[j]?.[i]; if (k) selection.add(k); }
let sliderBusy = false;
const model = { groups: {}, ready: false };

// ---------------------------------------------------------------- agent calls
async function cmd(body) {
  try {
    const r = await fetch('duck/api/cmd', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (j.error) note(j.error); else if (j.ok === false) note(j.msg);
    return j;
  } catch { note('no answer from the duck'); return null; }
}
let noteTimer = null;
function note(msg) {
  const s = $('#status'); s.textContent = msg; s.className = 'bad';
  clearTimeout(noteTimer); noteTimer = setTimeout(() => (noteTimer = null), 2500);
}
async function poll() {
  try {
    const r = await fetch('duck/api/state', { cache: 'no-store' });
    const j = await r.json();
    if (j.error) { agentOk = false; duck = null; } else { agentOk = true; duck = j; }
  } catch { agentOk = false; duck = null; }
  renderStatus(); renderJoints(); renderSelected(); renderRecord();
  if (source === 'live') poseModelFromDuck();
}
function limitsOf(j) {
  if (duck?.limits?.[j]) return duck.limits[j].map(r => r * R2D);
  for (const k in LIMITS_DEFAULT) if (j === k || j.endsWith(k)) return LIMITS_DEFAULT[k];
  return [-90, 90];
}
const duckRad = j => duck?.joints.find(x => x.name === j)?.rad ?? null;

// ---------------------------------------------------------------- side panel
function renderStatus() {
  const s = $('#status');
  if (noteTimer) return;
  if (!agentOk) { s.textContent = 'duck agent not running · press Connect duck'; s.className = 'bad'; return; }
  const holding = duck.joints.filter(j => j.torque).length, bad = duck.joints.filter(j => !j.ok).length;
  s.textContent = [`${duck.volts ?? '–'} V`, `${duck.amps} A`, `${holding}/14 holding`, bad ? `${bad} not answering` : '',
    duck.mode !== 'idle' ? duck.mode : '', duck.message].filter(Boolean).join(' · ');
  s.className = bad ? 'bad' : 'ok';
}
function renderJoints() {
  const list = $('#joint-list');
  if (!list.childElementCount) list.innerHTML = JOINTS.map(j => `<div class="j" data-j="${j}"><span class="dot"></span><span>${jointText(j)}</span><span class="deg mono"></span></div>`).join('');
  for (const el of $$('.j', list)) {
    const j = el.dataset.j, st = duck?.joints.find(x => x.name === j);
    el.classList.toggle('sel', j === sel);
    el.classList.toggle('hold', !!st?.torque && !st?.clay);
    el.classList.toggle('clay', !!st?.clay);
    el.classList.toggle('timer', st?.hold_in != null);
    el.classList.toggle('bad', !!duck && !st?.ok);
    $('.deg', el).textContent = st ? deg(st.rad) : '–';
  }
}
function renderSelected() {
  $('#joint-name').textContent = sel ? jointText(sel) : 'Click a servo';
  $('#joint-id').textContent = sel ? `ID ${SERVO_IDS[sel]}` : '';
  $('#joint-body').hidden = !sel;
  if (!sel) return;
  const st = duck?.joints.find(x => x.name === sel);
  const [lo, hi] = limitsOf(sel), slider = $('#joint-slider');
  slider.min = lo.toFixed(1); slider.max = hi.toFixed(1);
  $('#joint-angle').textContent = st ? deg(st.rad) : '–';
  const chip = $('#joint-torque');
  chip.textContent = !st ? '' : st.hold_in != null ? `free, holds in ${st.hold_in} s` : st.clay ? `clay · ${st.clay === 'follow' ? 'following your hand' : 'holding'}` : st.torque ? 'holding' : 'free';
  chip.className = 'chip' + (st?.hold_in != null ? ' timer' : st?.clay ? ' clay' : st?.torque ? ' on' : '');
  if (!sliderBusy && st) { const v = st.target ?? st.rad; if (v != null) { slider.value = (v * R2D).toFixed(1); $('#joint-target').textContent = deg(v); } }
  if (st && document.activeElement !== $('#joint-kp')) { $('#joint-kp').value = st.kp; $('#joint-kp-val').textContent = st.kp; }
  $('#joint-meta').textContent = st ? `load ${st.load} · ${st.ma} mA · ${st.volt} V · ${st.temp} °C` + (st.errors.length ? ' · ' + st.errors.join(', ') : '') + (st.ok ? '' : ' · not answering') : '';
}
function renderRecord() {
  const rec = duck?.mode === 'record';
  $('#rec-start').hidden = rec; $('#rec-stop').hidden = !rec;
  $('#rec-state').textContent = rec ? `recording ${duck.rec_live} s` : duck?.recorded ? `last take ${duck.recorded} s` : '';
  $('#rec-save').disabled = !duck?.recorded || rec;
  const onDuck = duck?.mode === 'play';
  $('#clip-duck').hidden = onDuck; $('#clip-duck-stop').hidden = !onDuck;
}
async function renderClips() {
  const clips = await (await fetch('api/clips')).json();
  const list = $('#clip-list');
  list.innerHTML = clips.length ? clips.map(c => `<div class="c${clipName === c.name ? ' sel' : ''}" data-name="${esc(c.name)}">
    <span><span class="n">${esc(c.name)}</span><br><span class="m">${(+c.seconds).toFixed(1)} s · ${c.frames} frames${c.broken ? ' · broken' : ''}</span></span>
    <button class="ghost small" data-act="load">Edit</button>
    <button class="ghost small" data-act="duck">Duck</button>
    <button class="ghost small" data-act="del" title="Delete">×</button></div>`).join('') : '<p class="muted small">No clips yet.</p>';
  const sel2 = $('#t-append-sel');
  sel2.innerHTML = clips.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join('');
}

// ---------------------------------------------------------------- model posing
function poseModelFromDuck() {
  if (!model.ready || !duck) return;
  for (const j of duck.joints) if (j.rad != null && model.groups[j.name]) model.groups[j.name].rotation.z = j.rad;
}
function poseModelFromClip() {
  if (!model.ready || !clip) return;
  const pose = A.poseAt(clip, playhead);
  for (const j in pose) if (model.groups[j]) model.groups[j].rotation.z = pose[j];
}
function setSource(s) {
  source = s;
  for (const b of $$('#model-source button')) b.classList.toggle('on', b.dataset.src === s);
  if (s === 'live') playing = false;
  if (s === 'clip') poseModelFromClip();
  drawTimeline();
}

// ---------------------------------------------------------------- the clip editor
function snapshot() { history.push(JSON.stringify(clip)); if (history.length > 60) history.shift(); }
function changed() { clipDirty = true; $('#clip-save').disabled = !clipName; drawTimeline(); poseModelFromClip(); renderKeyBar(); }
function undo() { if (!history.length) return; clip = JSON.parse(history.pop()); selection.clear(); changed(); }
function openClip(c, name) {
  clip = A.normalise(c); clipName = name; clipDirty = false; history.length = 0;
  playhead = 0; playing = false; selection.clear(); $('#clip-play').textContent = '▶';
  $('#editor').hidden = false; $('#t-smooth').checked = clip.interp !== 'linear';
  $('#clip-title').textContent = name || 'unsaved clip';
  $('#clip-save').disabled = !name;
  setSource('clip'); renderClips(); renderKeyBar();
}
async function loadClip(name) { openClip(await (await fetch('api/clips/' + encodeURIComponent(name))).json(), name); }
async function saveClip(name) {
  if (!clip) return;
  const r = await fetch('api/clips/' + encodeURIComponent(name), { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(A.toSaved(clip)) });
  if (!r.ok) return note('could not save (name must be letters, digits, space, _ - .)');
  clipName = name; clipDirty = false; $('#clip-title').textContent = name; $('#clip-save').disabled = false; renderClips();
}
function playOnDuck(frames) { cmd({ action: 'play', frames, loop: $('#clip-loop').checked, speed: +$('#clip-speed').value || 1 }); }

function renderKeyBar() {
  const kt = $('#k-time'), kd = $('#k-deg'), one = clip && oneSelected();
  kt.disabled = kd.disabled = !one; $('#k-del').disabled = !selection.size;
  if (one) { if (document.activeElement !== kt) kt.value = one.key.t.toFixed(2); if (document.activeElement !== kd) kd.value = (one.key.v * R2D).toFixed(1); }
  else { kt.value = ''; kd.value = ''; }
  $('#k-count').textContent = selection.size > 1 ? `${selection.size} keys selected` : '';
  $('#k-copy').disabled = !selection.size; $('#k-paste').disabled = $('#k-paste-m').disabled = !clipboard.length || !clip;
  const v = sel && clip ? A.valueAt(clip.keys[sel], playhead, clip.interp) : null, s = $('#tl-val');
  if (sel) { const [lo, hi] = limitsOf(sel); s.min = lo.toFixed(1); s.max = hi.toFixed(1); }
  s.disabled = !sel || !clip;
  if (v != null && document.activeElement !== s) s.value = (v * R2D).toFixed(1);
  $('#tl-val-txt').textContent = sel ? `${jointText(sel)} ${deg(v)}` : 'select a joint';
  $('#t-len').value = clip ? A.duration(clip).toFixed(2) : '';
  const d = clip ? A.duration(clip) : 0;
  $('#clip-time').textContent = `${playhead.toFixed(2)} / ${d.toFixed(2)} s`;
}

// --- the timeline canvas
const TL = { left: 118, ruler: 22, row: 17, lane: 140, pad: 12, bottom: 8 };
const tl = { canvas: null, ctx: null, w: 0, h: 0, drag: null };
function tMax() { return Math.max(1, A.duration(clip || { keys: {} }) * 1.06); }
function xOf(t) { return TL.left + (t / tMax()) * (tl.w - TL.left - TL.pad); }
function tOf(x) { return Math.max(0, Math.min(tMax(), (x - TL.left) / (tl.w - TL.left - TL.pad) * tMax())); }
function laneTop() { return TL.ruler + JOINTS.length * TL.row + 10; }
function laneRange() {
  const [lo, hi] = sel ? limitsOf(sel) : [-90, 90];
  const ks = sel && clip ? clip.keys[sel] : [];
  let a = Math.min(lo, ...ks.map(k => k.v * R2D)) - 5, b = Math.max(hi, ...ks.map(k => k.v * R2D)) + 5;
  return [a, b];
}
function yOfDeg(d) { const [a, b] = laneRange(); return laneTop() + (1 - (d - a) / (b - a)) * TL.lane; }
function degOfY(y) { const [a, b] = laneRange(); return a + (1 - (y - laneTop()) / TL.lane) * (b - a); }

function drawTimeline() {
  if (!tl.ctx) return;
  const c = tl.ctx, W = tl.w, H = tl.h, css = getComputedStyle(document.documentElement);
  const col = n => css.getPropertyValue(n).trim();
  c.clearRect(0, 0, W, H);
  c.font = '12px system-ui, sans-serif'; c.textBaseline = 'middle';
  if (!clip) { c.fillStyle = col('--muted'); c.fillText('Load a clip (Edit) or press New clip.', TL.left, 30); return; }
  const dur = A.duration(clip), tm = tMax();
  // ruler
  const stepS = tm > 30 ? 5 : tm > 12 ? 2 : tm > 5 ? 1 : 0.5;
  c.fillStyle = col('--muted');
  for (let t = 0; t <= tm; t += stepS) { const x = xOf(t); c.fillRect(x, TL.ruler - 6, 1, 6); c.fillText(t.toFixed(stepS < 1 ? 1 : 0) + 's', x + 3, 8); }
  c.fillStyle = col('--line'); c.fillRect(TL.left, TL.ruler, W - TL.left - TL.pad, 1);
  // end of clip
  c.fillStyle = col('--accent-soft'); c.fillRect(xOf(dur), TL.ruler, W - TL.pad - xOf(dur), laneTop() + TL.lane - TL.ruler);
  // joint rows
  JOINTS.forEach((j, i) => {
    const y = TL.ruler + i * TL.row + TL.row / 2;
    if (j === sel) { c.fillStyle = col('--accent-soft'); c.fillRect(0, y - TL.row / 2, W, TL.row); }
    c.fillStyle = j === sel ? col('--text') : col('--muted'); c.fillText(jointText(j), 6, y);
    c.fillStyle = col('--line'); c.fillRect(TL.left, y, W - TL.left - TL.pad, 1);
    for (let k = 0; k < clip.keys[j].length; k++) {
      const key = clip.keys[j][k], x = xOf(key.t), isSel = selection.has(key);
      c.fillStyle = isSel ? col('--accent') : j === sel ? col('--text') : col('--muted');
      c.beginPath(); c.moveTo(x, y - 5); c.lineTo(x + 5, y); c.lineTo(x, y + 5); c.lineTo(x - 5, y); c.closePath(); c.fill();
    }
  });
  // curve lane for the selected joint
  const top = laneTop();
  c.fillStyle = col('--panel'); c.fillRect(TL.left, top, W - TL.left - TL.pad, TL.lane);
  c.strokeStyle = col('--line'); c.strokeRect(TL.left + 0.5, top + 0.5, W - TL.left - TL.pad - 1, TL.lane - 1);
  if (sel) {
    const [a, b] = laneRange(), stepD = b - a > 100 ? 30 : b - a > 40 ? 15 : 5;
    c.fillStyle = col('--muted');
    for (let d = Math.ceil(a / stepD) * stepD; d <= b; d += stepD) { const y = yOfDeg(d); c.fillStyle = col('--line'); c.fillRect(TL.left, y, W - TL.left - TL.pad, 1); c.fillStyle = col('--muted'); c.fillText(d + '°', 70, y); }
    c.fillStyle = col('--text'); c.fillText(jointText(sel), 6, top + 12);
    const ks = clip.keys[sel];
    if (ks.length) {
      c.strokeStyle = col('--info'); c.lineWidth = 2; c.beginPath();
      for (let x = TL.left; x <= W - TL.pad; x += 2) { const v = A.valueAt(ks, tOf(x), clip.interp) * R2D, y = yOfDeg(v); x === TL.left ? c.moveTo(x, y) : c.lineTo(x, y); }
      c.stroke(); c.lineWidth = 1;
      ks.forEach((k, i) => {
        const isSel = selection.has(k);
        c.fillStyle = isSel ? col('--accent') : col('--info');
        c.beginPath(); c.arc(xOf(k.t), yOfDeg(k.v * R2D), isSel ? 6 : 4.5, 0, Math.PI * 2); c.fill();
      });
    }
    // live duck value, for keying from it
    const live = duckRad(sel);
    if (live != null && agentOk) { c.fillStyle = col('--good'); c.fillRect(TL.left, yOfDeg(live * R2D), W - TL.left - TL.pad, 1); c.fillText('duck now ' + deg(live), W - TL.pad - 110, yOfDeg(live * R2D) - 8); }
  } else { c.fillStyle = col('--muted'); c.fillText('Select a joint to edit its curve: drag the dots, double-click to add a key, Delete removes it.', TL.left + 10, top + TL.lane / 2); }
  if (box) { c.strokeStyle = col('--info'); c.setLineDash([4, 3]); c.strokeRect(Math.min(box.x0, box.x1) + 0.5, Math.min(box.y0, box.y1) + 0.5, Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0)); c.setLineDash([]); }
  // playhead
  const px = xOf(playhead);
  c.fillStyle = col('--accent'); c.fillRect(px, 0, 2, top + TL.lane);
  c.fillRect(px - 5, 0, 12, 10);
}

function hitKey(x, y) {
  if (!clip) return null;
  if (sel && y >= laneTop() && y <= laneTop() + TL.lane) {
    const ks = clip.keys[sel];
    for (let i = ks.length - 1; i >= 0; i--) if (Math.hypot(xOf(ks[i].t) - x, yOfDeg(ks[i].v * R2D) - y) <= 8) return { joint: sel, i, key: ks[i], lane: true };
    return null;
  }
  const row = Math.floor((y - TL.ruler) / TL.row);
  if (row >= 0 && row < JOINTS.length) {
    const j = JOINTS[row], ks = clip.keys[j];
    for (let i = ks.length - 1; i >= 0; i--) if (Math.abs(xOf(ks[i].t) - x) <= 6) return { joint: j, i, key: ks[i], lane: false };
    return { joint: j, i: -1, lane: false };
  }
  return null;
}
function setupTimeline() {
  const cv = $('#tl'); tl.canvas = cv; tl.ctx = cv.getContext('2d');
  const fit = () => {
    const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
    tl.w = r.width; tl.h = laneTop() + TL.lane + TL.bottom; cv.style.height = tl.h + 'px';
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(tl.h * dpr); tl.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTimeline();
  };
  new ResizeObserver(fit).observe(cv); fit();
  const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  cv.addEventListener('pointerdown', e => {
    if (!clip) return;
    const [x, y] = pos(e), hit = hitKey(x, y);
    cv.setPointerCapture(e.pointerId);
    if (e.shiftKey) {                                   // box select
      if (!e.ctrlKey && !e.metaKey) selection.clear();
      box = { x0: x, y0: y, x1: x, y1: y }; tl.drag = { kind: 'box' };
    } else if (hit && hit.i >= 0) {
      if (e.ctrlKey || e.metaKey) { if (selection.has(hit.key)) selection.delete(hit.key); else selection.add(hit.key); tl.drag = null; }
      else {
        if (hit.joint !== sel) select(hit.joint);
        if (!selection.has(hit.key)) { selection.clear(); selection.add(hit.key); }
        snapshot();
        const start = [...selection].map(k => ({ k, j: jointOfKey(k), t0: k.t, v0: k.v }));
        tl.drag = { kind: 'key', lane: hit.lane, moved: false, x0: x, y0: y, start };
      }
    } else {
      if (hit && hit.joint !== sel) select(hit.joint);
      if (!(e.ctrlKey || e.metaKey)) selection.clear();
      playing = false; $('#clip-play').textContent = '▶';
      playhead = Math.min(A.duration(clip), tOf(x)); tl.drag = { kind: 'scrub' };
      poseModelFromClip();
    }
    drawTimeline(); renderKeyBar();
  });
  cv.addEventListener('pointermove', e => {
    if (!tl.drag || !clip) return;
    const [x, y] = pos(e);
    if (tl.drag.kind === 'scrub') { playhead = Math.min(A.duration(clip), tOf(x)); poseModelFromClip(); drawTimeline(); renderKeyBar(); return; }
    if (tl.drag.kind === 'box') { box.x1 = x; box.y1 = y; drawTimeline(); return; }
    // move every selected key by the same time shift (and angle shift inside the curve lane)
    const dt = tOf(x) - tOf(tl.drag.x0), dv = tl.drag.lane ? (degOfY(y) - degOfY(tl.drag.y0)) * D2R : 0;
    for (const s of tl.drag.start) { s.k.t = Math.round(Math.max(0, s.t0 + dt) * 1000) / 1000; if (tl.drag.lane && s.j === sel) s.k.v = Math.round((s.v0 + dv) * 10000) / 10000; }
    for (const j of new Set(tl.drag.start.map(s => s.j))) clip.keys[j].sort((a, b) => a.t - b.t);
    tl.drag.moved = true; clipDirty = true; drawTimeline(); renderKeyBar(); poseModelFromClip();
  });
  cv.addEventListener('pointerup', () => {
    if (tl.drag?.kind === 'key' && !tl.drag.moved) history.pop();
    if (tl.drag?.kind === 'box' && box) {                 // everything inside the rectangle
      const [xa, xb] = [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1)], [ya, yb] = [Math.min(box.y0, box.y1), Math.max(box.y0, box.y1)];
      JOINTS.forEach((j, r) => {
        const y = TL.ruler + r * TL.row + TL.row / 2;
        for (const k of clip.keys[j]) {
          const inRow = y >= ya && y <= yb && xOf(k.t) >= xa && xOf(k.t) <= xb;
          const inLane = j === sel && ya >= laneTop() - 2 && yOfDeg(k.v * R2D) >= ya && yOfDeg(k.v * R2D) <= yb && xOf(k.t) >= xa && xOf(k.t) <= xb;
          if (inRow || inLane) selection.add(k);
        }
      });
      box = null; drawTimeline(); renderKeyBar();
    }
    tl.drag = null;
  });
  cv.addEventListener('dblclick', e => {
    if (!clip || !sel) return;
    const [x, y] = pos(e);
    if (y < laneTop() || y > laneTop() + TL.lane) return;
    snapshot(); const i = A.setKey(clip, sel, tOf(x), degOfY(y) * D2R); selectOnly(sel, i); changed();
  });
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,select,textarea')) return;
    if ((e.key === 'Delete' || e.key === 'Backspace') && selection.size && clip) { deleteSelected(); }
    if (e.code === 'KeyZ' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undo(); }
    if (e.code === 'KeyC' && (e.ctrlKey || e.metaKey)) { copySelected(); }
    if (e.code === 'KeyV' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); pasteAtPlayhead(e.shiftKey); }
    if (e.code === 'KeyA' && (e.ctrlKey || e.metaKey) && clip) { e.preventDefault(); for (const j of JOINTS) for (const k of clip.keys[j]) selection.add(k); drawTimeline(); renderKeyBar(); }
    if (e.key === ' ' && clip && source === 'clip') { e.preventDefault(); togglePlay(); }
  });
}
function deleteSelected() {
  if (!clip || !selection.size) return;
  snapshot();
  for (const j of JOINTS) { const keep = clip.keys[j].filter(k => !selection.has(k)); clip.keys[j] = keep.length ? keep : clip.keys[j].slice(0, 1); }
  selection.clear(); changed();
}
function copySelected() {
  if (!clip || !selection.size) return;
  const items = [...selection].map(k => ({ joint: jointOfKey(k), t: k.t, v: k.v })).filter(i => i.joint);
  const t0 = Math.min(...items.map(i => i.t));
  clipboard = items.map(i => ({ joint: i.joint, t: Math.round((i.t - t0) * 1000) / 1000, v: i.v }));
  note(`copied ${clipboard.length} key${clipboard.length > 1 ? 's' : ''} from ${new Set(clipboard.map(i => i.joint)).size} joint(s)`);
  renderKeyBar();
}
function pasteAtPlayhead(mirrored) {   // keys land at the playhead, keeping their spacing
  if (!clip || !clipboard.length) return;
  snapshot(); selection.clear();
  for (const item of clipboard) {
    const [j, v] = mirrored ? A.mirrorKey(item.joint, item.v) : [item.joint, item.v];
    const i = A.setKey(clip, j, playhead + item.t, v); selection.add(clip.keys[j][i]);
  }
  changed(); note(mirrored ? 'pasted on the other side' : 'pasted');
}
function togglePlay() {
  if (!clip) return;
  playing = !playing; $('#clip-play').textContent = playing ? '❚❚' : '▶';
  if (playing && playhead >= A.duration(clip)) playhead = 0;
}
function tick(now) {
  if (playing && clip && source === 'clip') {
    playhead += (now - lastTick) / 1000 * (+$('#clip-speed').value || 1);
    const end = A.duration(clip);
    if (playhead > end) { if ($('#clip-loop').checked) playhead = 0; else { playhead = end; playing = false; $('#clip-play').textContent = '▶'; } }
    poseModelFromClip(); drawTimeline(); renderKeyBar();
  }
  lastTick = now; requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- wiring
function wire() {
  $('#stop').addEventListener('click', () => cmd({ action: 'stop' }));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') cmd({ action: 'stop' }); });
  for (const b of $$('[data-cmd]')) b.addEventListener('click', () => {
    const c = b.dataset.cmd;
    cmd(c === 'home' ? { action: 'home', seconds: 2.5 } : c === 'clay' ? { action: 'clay', kp: +$('#clay-kp').value } : { action: c });
  });
  $('#clay-kp').addEventListener('input', () => ($('#clay-kp-val').textContent = $('#clay-kp').value));
  $('#clay-kp').addEventListener('change', () => { const inClay = duck?.joints.filter(j => j.clay).map(j => j.name) || []; if (inClay.length) cmd({ action: 'clay', joints: inClay, kp: +$('#clay-kp').value }); });
  $('#joint-clay').addEventListener('click', () => sel && cmd({ action: 'clay', joints: [sel], kp: +$('#clay-kp').value }));
  $('#connect').addEventListener('click', async () => {
    note('starting the agent on the duck…');
    const r = await (await fetch('api/agent', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'deploy' }) })).json();
    note(r.msg);
  });
  $('#joint-list').addEventListener('click', e => { const el = e.target.closest('.j'); if (el) select(el.dataset.j); });
  $('#joint-free').addEventListener('click', () => sel && cmd({ action: 'free', joints: [sel] }));
  $('#joint-hold').addEventListener('click', () => sel && cmd({ action: 'hold', joints: [sel] }));
  $('#joint-free-timer').addEventListener('click', e => { if (e.target.tagName === 'INPUT') return; sel && cmd({ action: 'free', joints: [sel], seconds: +$('#joint-secs').value || 5 }); });
  const slider = $('#joint-slider'); let sendTimer = null;
  slider.addEventListener('pointerdown', () => (sliderBusy = true));
  slider.addEventListener('input', () => { sliderBusy = true; $('#joint-target').textContent = deg(+slider.value * D2R); clearTimeout(sendTimer); sendTimer = setTimeout(() => sel && cmd({ action: 'set', joint: sel, rad: +slider.value * D2R, seconds: 0.25 }), 60); });
  slider.addEventListener('change', () => { sliderBusy = false; sel && cmd({ action: 'set', joint: sel, rad: +slider.value * D2R, seconds: 0.25 }); });
  $('#joint-kp').addEventListener('input', () => ($('#joint-kp-val').textContent = $('#joint-kp').value));
  $('#joint-kp').addEventListener('change', () => sel && cmd({ action: 'kp', joints: [sel], kp: +$('#joint-kp').value }));
  // record
  $('#rec-start').addEventListener('click', () => cmd({ action: 'record' }));
  $('#rec-stop').addEventListener('click', () => cmd({ action: 'record_stop' }));
  $('#rec-save').addEventListener('click', async () => {
    const name = $('#clip-name').value.trim() || `take ${new Date().toISOString().slice(5, 16).replace('T', ' ')}`;
    const frames = await (await fetch('duck/api/clip')).json();
    if (!Array.isArray(frames) || !frames.length) return note('nothing recorded');
    const r = await fetch('api/clips/' + encodeURIComponent(name), { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(frames) });
    if (!r.ok) return note('could not save (name must be letters, digits, space, _ - .)');
    $('#clip-name').value = ''; await renderClips(); loadClip(name);
  });
  // clips
  $('#clip-new').addEventListener('click', () => {
    const keys = {}; for (const j of JOINTS) { const v = duckRad(j) ?? 0; keys[j] = [{ t: 0, v }, { t: 2, v }]; }
    openClip({ keys, interp: 'smooth' }, null); note('new 2 s clip from the duck\'s current pose; Save as… to keep it');
  });
  $('#clip-list').addEventListener('click', async e => {
    const b = e.target.closest('button'); if (!b) return;
    const name = b.closest('.c').dataset.name;
    if (b.dataset.act === 'load') loadClip(name);
    else if (b.dataset.act === 'duck') { const c = await (await fetch('api/clips/' + encodeURIComponent(name))).json(); playOnDuck(Array.isArray(c) ? c : c.frames || A.sample(A.normalise(c))); }
    else if (b.dataset.act === 'del') { if (confirm(`Delete clip "${name}"?`)) { await fetch('api/clips/' + encodeURIComponent(name), { method: 'DELETE' }); if (clipName === name) clipName = null; renderClips(); } }
  });
  for (const b of $$('#model-source button')) b.addEventListener('click', () => { if (b.dataset.src === 'clip' && !clip) return note('load a clip first'); setSource(b.dataset.src); });
  // transport
  $('#clip-play').addEventListener('click', togglePlay);
  $('#clip-duck').addEventListener('click', () => clip && playOnDuck(A.sample(clip)));
  $('#clip-duck-stop').addEventListener('click', () => cmd({ action: 'play_stop' }));
  $('#clip-save').addEventListener('click', () => clipName && saveClip(clipName));
  $('#clip-saveas').addEventListener('click', () => { const n = prompt('Clip name', clipName ? clipName + ' v2' : 'clip'); if (n) saveClip(n.trim()); });
  // keys
  const tlVal = $('#tl-val');
  tlVal.addEventListener('input', () => { if (!sel || !clip) return; if (!tl.valSnap) { snapshot(); tl.valSnap = true; } const i = A.setKey(clip, sel, playhead, +tlVal.value * D2R); selectOnly(sel, i); changed(); });
  tlVal.addEventListener('change', () => (tl.valSnap = false));
  $('#k-set').addEventListener('click', () => { if (!sel || !clip) return; snapshot(); const i = A.setKey(clip, sel, playhead, +tlVal.value * D2R); selectOnly(sel, i); changed(); });
  $('#k-duck').addEventListener('click', () => { if (!sel || !clip) return; const v = duckRad(sel); if (v == null) return note('no live value from the duck'); snapshot(); const i = A.setKey(clip, sel, playhead, v); selectOnly(sel, i); changed(); });
  $('#k-duck-all').addEventListener('click', () => { if (!clip || !duck) return note('no live values from the duck'); snapshot(); for (const j of JOINTS) { const v = duckRad(j); if (v != null) A.setKey(clip, j, playhead, v); } changed(); });
  $('#k-del').addEventListener('click', deleteSelected);
  $('#k-copy').addEventListener('click', copySelected);
  $('#k-paste').addEventListener('click', () => pasteAtPlayhead(false));
  $('#k-paste-m').addEventListener('click', () => pasteAtPlayhead(true));
  $('#k-time').addEventListener('change', () => { const o = clip && oneSelected(); if (!o) return; snapshot(); A.moveKey(clip, o.joint, o.i, +$('#k-time').value, o.key.v); changed(); });
  $('#k-deg').addEventListener('change', () => { const o = clip && oneSelected(); if (!o) return; snapshot(); A.moveKey(clip, o.joint, o.i, o.key.t, +$('#k-deg').value * D2R); changed(); });
  $('#undo').addEventListener('click', undo);
  // whole-clip tools
  const tool = (id, fn) => $(id).addEventListener('click', () => { if (!clip) return; snapshot(); fn(); selection.clear(); changed(); });
  tool('#t-faster', () => { A.scaleTime(clip, 0.5); playhead *= 0.5; });
  tool('#t-slower', () => { A.scaleTime(clip, 2); playhead *= 2; });
  tool('#t-len-apply', () => { const d = A.duration(clip), want = +$('#t-len').value; if (d > 0 && want > 0) { A.scaleTime(clip, want / d); playhead *= want / d; } });
  tool('#t-cut-before', () => { A.trim(clip, playhead, A.duration(clip)); playhead = 0; });
  tool('#t-cut-after', () => A.trim(clip, 0, playhead));
  tool('#t-nudge-up', () => sel && A.offsetJoint(clip, sel, 2 * D2R));
  tool('#t-nudge-dn', () => sel && A.offsetJoint(clip, sel, -2 * D2R));
  $('#t-loop').addEventListener('click', () => {
    if (!clip) return;
    const d = A.duration(clip), ans = prompt(`Keep from .. to (seconds), then blend the end into the start.\nClip is 0 .. ${d.toFixed(2)} s`, `0, ${d.toFixed(2)}`);
    if (!ans) return;
    const [a, b] = ans.split(/[,\s]+/).map(Number);
    if (!(b > a && a >= 0 && b <= d + 1e-6)) return note('bad range');
    snapshot(); clip = A.makeLoop(clip, a, b, 0.15); playhead = 0; selection.clear(); changed(); note('looped: end blended into the start');
  });
  $('#t-mirror').addEventListener('click', () => { if (!clip) return; openClip(A.mirror(clip), null); $('#clip-title').textContent = (clipName || 'clip') + ' mirrored (unsaved)'; });
  $('#t-append').addEventListener('click', async () => {
    const name = $('#t-append-sel').value; if (!clip || !name) return;
    const other = A.normalise(await (await fetch('api/clips/' + encodeURIComponent(name))).json());
    snapshot(); A.append(clip, other, +$('#t-gap').value || 0); changed();
  });
  $('#t-smooth').addEventListener('change', () => { if (!clip) return; snapshot(); clip.interp = $('#t-smooth').checked ? 'smooth' : 'linear'; changed(); });
  requestAnimationFrame(tick);
}
function select(j) { sel = j; sliderBusy = false; renderJoints(); renderSelected(); highlight(); drawTimeline(); renderKeyBar(); }

// ---------------------------------------------------------------- 3D model
let highlight = () => {};
async function initViewer() {
  const wrap = $('#canvas-wrap');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  wrap.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 1.8));
  const key = new THREE.DirectionalLight(0xffffff, 1.8); key.position.set(1.2, 2, 1.5); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7); fill.position.set(-1.5, 0.6, -1); scene.add(fill);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  const catalog = await (await fetch('data/catalog.json')).json();
  const xml = new DOMParser().parseFromString(await (await fetch('robot/robot.urdf')).text(), 'application/xml');
  const applyOrigin = (obj, el) => {
    if (!el) return;
    const xyz = (el.getAttribute('xyz') || '0 0 0').trim().split(/\s+/).map(Number), rpy = (el.getAttribute('rpy') || '0 0 0').trim().split(/\s+/).map(Number);
    obj.position.set(xyz[0], xyz[1], xyz[2]); obj.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');
  };
  const links = {}, children = new Set(), visuals = [];
  for (const l of xml.querySelectorAll('robot > link')) links[l.getAttribute('name')] = new THREE.Group();
  for (const j of xml.querySelectorAll('robot > joint')) {
    const name = j.getAttribute('name'), parent = j.querySelector('parent').getAttribute('link'), child = j.querySelector('child').getAttribute('link');
    const jg = new THREE.Group(); applyOrigin(jg, j.querySelector(':scope > origin'));
    const axis = new THREE.Group();                 // every revolute joint turns about its own z
    links[parent].add(jg); jg.add(axis); axis.add(links[child]); children.add(child);
    if (name in SERVO_IDS) model.groups[name] = axis;
  }
  const root = new THREE.Group(); root.rotation.x = -Math.PI / 2;
  for (const n in links) if (!children.has(n)) root.add(links[n]);
  for (const l of xml.querySelectorAll('robot > link')) for (const v of l.querySelectorAll(':scope > visual')) {
    const m = v.querySelector('geometry > mesh'); if (!m) continue;
    const holder = new THREE.Group(); applyOrigin(holder, v.querySelector(':scope > origin'));
    links[l.getAttribute('name')].add(holder);
    const rgba = v.querySelector('material > color')?.getAttribute('rgba')?.trim().split(/\s+/).map(Number);
    visuals.push({ file: m.getAttribute('filename').replace(/^.*\//, ''), rgba, holder });
  }
  const files = [...new Set(visuals.map(v => v.file))], geoms = {}, loader = new STLLoader();
  let n = 0;
  await Promise.all(files.map(async f => { geoms[f] = await loader.loadAsync('robot/' + f); $('#loading').textContent = `Loading model… ${++n}/${files.length}`; }));
  const meshes = [];
  for (const v of visuals) {
    const base = v.file.replace(/\.stl$/i, ''), k = catalog.mesh_map[base] || 'other:' + base;
    const cad = v.rgba ? new THREE.Color().setRGB(v.rgba[0], v.rgba[1], v.rgba[2], THREE.SRGBColorSpace) : new THREE.Color(0x999999);
    const mesh = new THREE.Mesh(geoms[v.file], new THREE.MeshStandardMaterial({ color: cad, roughness: 0.6, metalness: 0.05 }));
    mesh.userData = { key: k, cad, file: v.file };
    v.holder.add(mesh); meshes.push(mesh);
  }
  scene.add(root); root.updateMatrixWorld(true);
  {   // which servo mesh drives which joint: the joint sits on the output shaft in the servo's frame
    const shaft = new THREE.Vector3(...SHAFT), jp = Object.entries(model.groups).map(([nm, g]) => [nm, g.getWorldPosition(new THREE.Vector3())]);
    const servos = meshes.filter(m => m.userData.key === SERVO_KEY), bodies = [];
    for (const m of servos) {
      if (m.userData.file === 'passive_palonier.stl') continue;
      let best = null;
      for (const [nm, p] of jp) { const d = m.parent.worldToLocal(p.clone()).distanceTo(shaft); if (!best || d < best[0]) best = [d, nm]; }
      if (best && best[0] < 0.002) { m.userData.joint = best[1]; bodies.push(m); }
    }
    for (const m of servos) if (m.userData.file === 'passive_palonier.stl') {
      const p = m.getWorldPosition(new THREE.Vector3()); let best = null;
      for (const b of bodies) if (b.parent.parent === m.parent.parent) { const d = b.getWorldPosition(new THREE.Vector3()).distanceTo(p); if (!best || d < best[0]) best = [d, b.userData.joint]; }
      if (best) m.userData.joint = best[1];
    }
  }
  const box = new THREE.Box3().setFromObject(root), c = box.getCenter(new THREE.Vector3());
  root.position.set(-c.x, -box.min.y, -c.z);
  const h = box.max.y - box.min.y;
  const ground = new THREE.Mesh(new THREE.CircleGeometry(h * 0.6, 48), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.06 }));
  ground.rotation.x = -Math.PI / 2; scene.add(ground);
  $('#loading').remove();
  const resetView = () => { controls.target.set(0, h * 0.5, 0); camera.position.set(h * 2.0, h * 0.9, h * 0.7); controls.update(); };
  resetView(); $('#reset-view').addEventListener('click', resetView);
  new ResizeObserver(() => { const w = wrap.clientWidth, ht = wrap.clientHeight; if (!w || !ht) return; renderer.setSize(w, ht, false); camera.aspect = w / ht; camera.updateProjectionMatrix(); }).observe(wrap);
  highlight = () => { for (const m of meshes) { const on = m.userData.joint && m.userData.joint === sel; m.material.emissive.set(on ? 0xd99a00 : 0x000000); m.material.emissiveIntensity = on ? 0.55 : 0; } };
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(), tip = $('#hover-tip');
  const pickHit = ev => { const r = renderer.domElement.getBoundingClientRect(); ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ptr, camera); return ray.intersectObjects(meshes, false)[0]; };
  let down = null;
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => { if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return; const j = pickHit(e)?.object.userData.joint; if (j) select(j); });
  renderer.domElement.addEventListener('pointermove', e => {
    if (e.buttons || e.pointerType !== 'mouse') { tip.style.display = 'none'; return; }
    const j = pickHit(e)?.object.userData.joint;
    if (!j) { tip.style.display = 'none'; return; }
    const r = wrap.getBoundingClientRect();
    tip.textContent = `${jointText(j)} · ID ${SERVO_IDS[j]}`; tip.style.display = 'block'; tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
  });
  renderer.domElement.addEventListener('pointerleave', () => (tip.style.display = 'none'));
  model.ready = true;
  (function animate() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); })();
}

wire(); setupTimeline(); renderJoints(); renderKeyBar();
initViewer().catch(e => { $('#loading').textContent = 'Model failed to load: ' + e.message; });
renderClips();
poll(); setInterval(poll, 100);
