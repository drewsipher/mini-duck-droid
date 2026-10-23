import * as THREE from 'three';
import { STLLoader } from 'three/addons/STLLoader.js';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const money = n => n == null ? '–' : '$' + Math.round(n).toLocaleString();

let catalog, state, viewer = null, selected = null, focused = null, pendingFocus = null;
const printById = {}, buyById = {};
let modelKeys = new Set();  // tracker keys that have geometry in the 3D model

// ---------------------------------------------------------------- state -----
function normalise(s) {
  s = s && typeof s === 'object' ? s : {};
  s.version = 1;
  s.print ??= {}; s.buy ??= {}; s.ui ??= {}; s.ui.layers ??= {};
  for (const p of catalog.print) s.print[p.id] = { done: 0, printing: 0, notes: '', ...(s.print[p.id] || {}) };
  for (const b of catalog.buy) s.buy[b.id] = { status: 'need', notes: '', ...(s.buy[b.id] || {}) };
  s.palette ??= structuredClone(catalog.palette);
  s.colours ??= { ...catalog.designer_colours };
  return s;
}

let saveTimer;
function changed({ render = true } = {}) {
  if (render) renderAll();
  else renderStats();
  viewer?.recolour();
  $('#save-state').textContent = 'Saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 400);
}
async function save() {
  state.updated = new Date().toISOString();
  try {
    const r = await fetch('api/progress', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state) });
    if (!r.ok) throw new Error(r.status);
    $('#save-state').textContent = 'Saved';
  } catch {
    $('#save-state').textContent = 'Not saved — is the tracker server running?';
  }
}

const included = item => !item.optional || state.ui.includeOptional;
const printIncluded = p => !p.section.startsWith('Expression') || state.ui.includeOptional;
const colourOf = id => state.palette.find(c => c.id === state.colours[id]);

// ---------------------------------------------------------------- stats -----
function renderStats() {
  let total = 0, done = 0, printing = 0;
  for (const p of catalog.print.filter(printIncluded)) {
    const s = state.print[p.id];
    total += p.qty; done += Math.min(s.done, p.qty); printing += s.printing;
  }
  $('#stat-print').textContent = `${done} / ${total} pieces` + (printing ? ` · ${printing} printing` : '');
  $('#bar-print').style.width = `${(100 * done / total).toFixed(1)}%`;

  const items = catalog.buy.filter(included);
  const have = items.filter(b => state.buy[b.id].status === 'have').length;
  const ordered = items.filter(b => state.buy[b.id].status === 'ordered');
  $('#stat-buy').textContent = `${have} / ${items.length} items` + (ordered.length ? ` · ${ordered.length} ordered` : '');
  $('#bar-buy').style.width = `${(100 * have / items.length).toFixed(1)}%`;

  const need = items.filter(b => state.buy[b.id].status === 'need');
  const spend = need.reduce((a, b) => a + (b.est_total_cad || 0), 0);
  const unpriced = need.filter(b => b.est_total_cad == null).length;
  $('#stat-spend').textContent = `C${money(spend)}`;
  $('#stat-spend-sub').textContent = [
    ordered.length ? `C${money(ordered.reduce((a, b) => a + (b.est_total_cad || 0), 0))} on order` : '',
    unpriced ? `${unpriced} item${unpriced > 1 ? 's' : ''} unpriced` : '',
    state.ui.includeOptional ? 'incl. optional' : 'excl. optional parts',
  ].filter(Boolean).join(' · ');
}

// ------------------------------------------------- show a part in the 3D -----
function nameLink(key, name) {
  if (!modelKeys.has(key)) return esc(name);
  return `<button class="link3d" data-view="focus" data-key="${esc(key)}" title="Show only this part in the 3D model">${esc(name)}<span class="ico" aria-hidden="true">◎ 3D</span></button>`;
}
function focusPart(key) {
  selected = focused = key;
  showTab('model');
  renderSelection();
  if (viewer) { viewer.recolour(); viewer.frame(key); } else pendingFocus = key;
}
function clearFocus() {
  if (!focused) return;
  focused = null;
  renderSelection();
  viewer?.recolour();
}
function onViewAction(e) {
  const b = e.target.closest('[data-view]'); if (!b) return;
  e.preventDefault();
  if (b.dataset.view === 'focus') focusPart(b.dataset.key || selected);
  else if (b.dataset.view === 'unfocus') clearFocus();
  else if (b.dataset.view === 'frame') viewer?.frame(selected);
  else if (b.dataset.view === 'ghost' || b.dataset.view === 'hidden') setLayer(selected, b.dataset.view);
}

// ----------------------------------------- per-part solid / ghost / off -----
// Ghosted parts are see-through and clicks pass through them; "off" parts are not drawn at all.
const layerOf = key => state.ui.layers[key] || 'solid';
function setLayer(key, layer) {
  if (!key) return;
  if (layer === 'solid') delete state.ui.layers[key]; else state.ui.layers[key] = layer;
  if (layer !== 'solid' && selected === key) selected = null;  // it can't be clicked any more
  renderSelection(); renderLayers();
  changed({ render: false });
}
let layersOpen = false;
function renderLayers() {
  const el = $('#layers'); if (!el) return;
  const kindOrder = { print: 0, buy: 1, other: 2 };
  const keys = [...modelKeys].sort((a, b) =>
    (kindOrder[a.split(':')[0]] - kindOrder[b.split(':')[0]]) || labelOf(a).localeCompare(labelOf(b)));
  const changedKeys = keys.filter(k => layerOf(k) !== 'solid');
  const row = k => {
    const l = layerOf(k);
    const seg = [['solid', 'Solid'], ['ghost', 'Ghost'], ['hidden', 'Off']]
      .map(([v, t]) => `<button data-layer="${v}" class="${l === v ? 'on' : ''}">${t}</button>`).join('');
    return `<div class="layer-row" data-key="${esc(k)}"><button class="layer-name" data-layer-select title="Select and zoom to it">${esc(labelOf(k))}</button><div class="seg layer-seg" data-l="${l}">${seg}</div></div>`;
  };
  const group = (title, kind) => {
    const ks = keys.filter(k => k.startsWith(kind + ':'));
    const all = [['solid', 'Solid'], ['ghost', 'Ghost'], ['hidden', 'Off']].map(([v, t]) => `<button data-layer-group="${kind}" data-v="${v}">${t}</button>`).join(' · ');
    return ks.length ? `<div class="layer-group"><span>${title}</span><span class="layer-group-all">all: ${all}</span></div>${ks.map(row).join('')}` : '';
  };
  const nGhost = changedKeys.filter(k => layerOf(k) === 'ghost').length, nOff = changedKeys.length - nGhost;
  el.innerHTML = `<div class="card-head"><h2>Show / hide parts</h2>${changedKeys.length ? '<button class="ghost" data-layer-all>Show all</button>' : ''}</div>
    <p class="muted small layer-help">Ghost = see-through, and clicks go straight through it to the parts behind. Off = not drawn.
      Shortcuts: click a part, then <kbd>T</kbd> to ghost it or <kbd>H</kbd> to turn it off.</p>
    ${changedKeys.length ? `<div class="small muted">${[nGhost && `${nGhost} ghosted`, nOff && `${nOff} off`].filter(Boolean).join(' · ')}</div>` : ''}
    ${changedKeys.length && !layersOpen ? `<div class="layer-list">${changedKeys.map(row).join('')}</div>` : ''}
    <details id="layers-all" ${layersOpen ? 'open' : ''}><summary>All parts (${keys.length})</summary>
      <div class="layer-list">${group('Printed', 'print')}${group('Bought', 'buy')}${group('Other', 'other')}</div></details>`;
}
function onLayerAction(e) {
  if (e.target.closest('[data-layer-all]')) {
    state.ui.layers = {};
    renderSelection(); renderLayers(); return changed({ render: false });
  }
  const g = e.target.closest('[data-layer-group]');
  if (g) {
    for (const k of modelKeys) if (k.startsWith(g.dataset.layerGroup + ':')) {
      if (g.dataset.v === 'solid') delete state.ui.layers[k]; else state.ui.layers[k] = g.dataset.v;
    }
    if (selected && layerOf(selected) !== 'solid') selected = null;
    renderSelection(); renderLayers(); return changed({ render: false });
  }
  const r = e.target.closest('[data-key]'); if (!r) return;
  const key = r.dataset.key, b = e.target.closest('[data-layer]');
  if (b) return setLayer(key, b.dataset.layer);
  if (e.target.closest('[data-layer-select]')) {
    if (layerOf(key) !== 'solid') setLayer(key, 'solid');
    selected = key; showTab('model'); renderSelection();
    viewer?.recolour(); viewer?.frame(key);
  }
}

// ----------------------------------------------------------- print list -----
function counter(act, value, label) {
  return `<div class="counter ${act === 'printing' ? 'printing-ctl' : ''}" data-act="${act}">
    <button data-d="-1" aria-label="minus">−</button><span>${label}</span><button data-d="1" aria-label="plus">+</button></div>`;
}
function colourSelect(id) {
  const c = colourOf(id);
  const opts = [`<option value="">No colour</option>`].concat(state.palette.map(p =>
    `<option value="${esc(p.id)}" ${p.id === c?.id ? 'selected' : ''}>${esc(p.name)}</option>`));
  return `<span class="sw-select" title="${esc(c?.name || 'No colour')}">
    <span class="sw ${c ? '' : 'none'}" style="${c ? `background:${esc(c.hex)}` : ''}; display:block"></span>
    <select data-act="colour" aria-label="Colour">${opts.join('')}</select></span>`;
}
function printRow(p) {
  const s = state.print[p.id];
  const isDone = s.done >= p.qty;
  return `<div class="item print-item ${isDone ? 'done' : ''}" data-kind="print" data-id="${p.id}">
    ${colourSelect(p.id)}
    <div><div class="name">${nameLink('print:' + p.id, p.name)}${isDone ? ' <span class="tag good">done</span>' : ''}</div>
      <div class="spec"><span class="tag">${p.material} ${p.infill}</span>
      ${p.in_model ? '' : '<span class="tag">not in 3D</span>'}
      <a href="stl/${esc(p.file)}" download>${esc(p.file)}</a></div></div>
    ${counter('done', s.done, `${s.done}/${p.qty} done`)}
    ${counter('printing', s.printing, `${s.printing} printing`)}
    <input class="notes" data-act="notes" placeholder="Notes" value="${esc(s.notes)}">
  </div>`;
}
function renderPrint() {
  const f = state.ui.printFilter || 'all', g = state.ui.printGroup || 'section';
  $$('#print-filter button').forEach(b => b.classList.toggle('on', b.dataset.f === f));
  $$('#print-group button').forEach(b => b.classList.toggle('on', b.dataset.g === g));
  const parts = catalog.print.filter(p => {
    const d = state.print[p.id].done >= p.qty;
    return f === 'all' || (f === 'done' ? d : !d);
  });
  const groups = new Map();
  for (const p of parts) {
    const key = g === 'section' ? p.section : (colourOf(p.id)?.id || '');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }
  if (g === 'colour') {  // palette order, uncoloured last
    const order = state.palette.map(c => c.id).concat('');
    [...groups.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b)).forEach(k => { const v = groups.get(k); groups.delete(k); groups.set(k, v); });
  }
  let html = '';
  for (const [key, ps] of groups) {
    const all = catalog.print.filter(p => g === 'section' ? p.section === key : (colourOf(p.id)?.id || '') === key);
    const tot = all.reduce((a, p) => a + p.qty, 0), dn = all.reduce((a, p) => a + Math.min(state.print[p.id].done, p.qty), 0);
    let title = esc(key);
    if (g === 'colour') {
      const c = state.palette.find(x => x.id === key);
      title = c ? `<span class="sw" style="background:${esc(c.hex)};display:inline-block;width:14px;height:14px;vertical-align:-2px;margin-right:6px"></span>${esc(c.name)}` : 'No colour assigned';
    }
    html += `<div class="group"><div class="group-head"><span>${title}</span><span class="muted small">${dn}/${tot} pieces</span></div>${ps.map(printRow).join('')}</div>`;
  }
  $('#print-list').innerHTML = html || '<p class="muted">Nothing here.</p>';
}

// ------------------------------------------------------------- buy list -----
function source(label, s) {
  if (!s?.url) return '';
  const pack = s.pack_size > 1 ? ` (pack of ${s.pack_size})` : '';
  const price = s.price_cad != null ? ` — C$${Number(s.price_cad).toFixed(2)}${pack}` : pack;
  return `<div class="source"><span class="lbl">${label}</span><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.store || 'Link')}</a>${esc(price)}${s.ships_from ? `<span class="v">${esc(s.ships_from)}</span>` : ''}</div>`;
}
function aliSource(b) {
  const a = b.ali; if (!a?.url) return '';
  const pack = a.pack_size > 1 ? ` (pack of ${a.pack_size})` : '';
  const price = a.price_cad != null ? ` — C$${Number(a.price_cad).toFixed(2)}${pack}` : ' — price not readable';
  const tag = a.cheaper === true ? '<span class="tag good">cheaper</span>' : a.cheaper === false ? '<span class="tag">not cheaper</span>' : '<span class="tag">unclear</span>';
  return `<div class="source ali"><span class="lbl">Ali</span><a href="${esc(a.url)}" target="_blank" rel="noopener noreferrer">AliExpress</a>${esc(price)} ${tag}${a.note ? `<span class="v">${esc(a.note)}</span>` : ''}</div>`;
}
function details(b) {
  const rows = [b.spec_notes && `<p>${esc(b.spec_notes)}</p>`,
    b.best?.verified && `<p><b>Best pick checked:</b> ${esc(b.best.verified)}</p>`,
    b.alt?.verified && `<p><b>Alt checked:</b> ${esc(b.alt.verified)}</p>`,
    b.ali?.verified && `<p><b>AliExpress checked:</b> ${esc(b.ali.verified)}</p>`].filter(Boolean);
  return rows.length ? `<details><summary>Details</summary><div class="spec">${rows.join('')}</div></details>` : '';
}
function buyRow(b) {
  const s = state.buy[b.id];
  const seg = ['need', 'ordered', 'have'].map(k =>
    `<button data-s="${k}" class="${s.status === k ? 'on' : ''}">${k[0].toUpperCase() + k.slice(1)}</button>`).join('');
  const sources = source('Best', b.best) + source('Alt', b.alt) + aliSource(b) || '<span class="muted small">Sourcing pending</span>';
  return `<div class="item buy-item ${s.status}" data-kind="buy" data-id="${b.id}">
    <div class="seg status-seg" data-act="status">${seg}</div>
    <div><div class="name">${nameLink('buy:' + b.id, b.name)} <span class="tag">×${b.qty}</span>${b.optional ? ' <span class="tag">optional</span>' : ''}${b.flag ? ` <span class="tag warn">${esc(b.flag)}</span>` : ''}</div>
      ${b.summary ? `<div class="summary">${esc(b.summary)}</div>` : ''}
      ${details(b)}
      <input class="notes" data-act="notes" placeholder="Notes (order #, where it's stored…)" value="${esc(s.notes)}" style="margin-top:6px"></div>
    <div class="sources">${sources}</div>
    <div class="price">${b.est_total_cad != null ? 'C' + money(b.est_total_cad) : '–'}<span class="muted">est. total</span></div>
  </div>`;
}
function renderBuy() {
  const f = state.ui.buyFilter || 'all';
  $$('#buy-filter button').forEach(b => b.classList.toggle('on', b.dataset.f === f));
  $('#include-optional').checked = !!state.ui.includeOptional;
  const items = catalog.buy.filter(b => included(b) && (f === 'all' || state.buy[b.id].status === f));
  const groups = new Map();
  for (const b of items) { if (!groups.has(b.category)) groups.set(b.category, []); groups.get(b.category).push(b); }
  let html = '';
  for (const [cat, bs] of groups) {
    const left = bs.filter(b => state.buy[b.id].status === 'need').reduce((a, b) => a + (b.est_total_cad || 0), 0);
    html += `<div class="group"><div class="group-head"><span>${esc(cat)}</span><span class="muted small">C${money(left)} left to buy</span></div>${bs.map(buyRow).join('')}</div>`;
  }
  $('#buy-list').innerHTML = html || '<p class="muted">Nothing here.</p>';
}

// ------------------------------------------------------ model side panel -----
function renderPalette() {
  const count = {};
  for (const p of catalog.print) {
    const c = state.colours[p.id]; if (!c) continue;
    count[c] ??= { n: 0, left: 0 };
    count[c].n += p.qty; count[c].left += Math.max(0, p.qty - state.print[p.id].done);
  }
  $('#palette').innerHTML = state.palette.map(c => `<div class="pal-row" data-cid="${esc(c.id)}">
    <input type="color" value="${esc(c.hex)}" data-pal="hex" aria-label="Colour">
    <input type="text" value="${esc(c.name)}" data-pal="name" aria-label="Name">
    <span class="muted small">${count[c.id] ? `${count[c.id].n} pcs · ${count[c.id].left} left` : 'unused'}</span>
    <button class="x" data-pal="del" title="Remove colour">✕</button></div>`).join('');
}
function focusButtons() {
  const layer = modelKeys.has(selected) && !focused
    ? `<div class="row-line"><button class="ghost" data-view="ghost" title="See-through; clicks pass through it">Ghost it <kbd>T</kbd></button><button class="ghost" data-view="hidden">Turn off <kbd>H</kbd></button></div>`
    : '';
  return (focused === selected
    ? `<div class="row-line"><button class="ghost" data-view="unfocus">Show all parts</button><button class="ghost" data-view="frame">Zoom to it</button></div>`
    : `<div class="row-line"><button class="ghost" data-view="focus">Show only this part</button><button class="ghost" data-view="frame">Zoom to it</button></div>`) + layer;
}
function renderSelection() {
  const el = $('#selection');
  $('#show-all').hidden = !focused;
  if (!selected) { el.innerHTML = '<p class="muted">Click a part in the model to select it. Double-click to show only that part, or use the part names in the Print and Shopping lists.</p>'; return; }
  const [kind, id] = selected.split(/:(.*)/s);
  if (kind === 'print') {
    const p = printById[id], s = state.print[id], cur = state.colours[id] || '';
    const sw = [`<button class="sw none ${cur ? '' : 'on'}" data-act="swatch" data-c="" title="No colour"></button>`]
      .concat(state.palette.map(c => `<button class="sw ${c.id === cur ? 'on' : ''}" style="background:${esc(c.hex)}" data-act="swatch" data-c="${esc(c.id)}" title="${esc(c.name)}"></button>`));
    el.innerHTML = `<div data-kind="print" data-id="${id}">
      <div class="sel-title">${esc(p.name)}</div>
      <div class="sel-meta">Printed part · ×${p.qty} · ${p.material} ${p.infill} · <a href="stl/${esc(p.file)}" download>${esc(p.file)}</a></div>
      <div class="row-line">${counter('done', s.done, `${s.done}/${p.qty} done`)} ${counter('printing', s.printing, `${s.printing} printing`)}</div>
      <div class="small muted" style="margin-top:10px">Colour</div><div class="swatches" style="margin-top:4px">${sw.join('')}</div>${focusButtons()}</div>`;
  } else if (kind === 'buy') {
    const b = buyById[id], s = state.buy[id];
    const seg = ['need', 'ordered', 'have'].map(k => `<button data-s="${k}" class="${s.status === k ? 'on' : ''}">${k[0].toUpperCase() + k.slice(1)}</button>`).join('');
    el.innerHTML = `<div data-kind="buy" data-id="${id}">
      <div class="sel-title">${esc(b.name)}</div><div class="sel-meta">Bought part · ×${b.qty}</div>
      <div class="seg status-seg" data-act="status">${seg}</div>
      ${b.summary ? `<div class="summary" style="margin-top:8px">${esc(b.summary)}</div>` : ''}
      <div style="margin-top:8px">${source('Best', b.best) || '<span class="muted small">Sourcing pending</span>'}</div>${focusButtons()}</div>`;
  } else {
    el.innerHTML = `<div class="sel-title">${esc(id)}</div><div class="sel-meta">Comes with another part — not tracked separately.</div>${focusButtons()}`;
  }
}
function renderAll() { renderStats(); renderPrint(); renderBuy(); renderPalette(); renderSelection(); renderLayers(); }

// ------------------------------------------------------------- actions -----
function onAction(e) {
  const t = e.target;
  const row = t.closest('[data-kind][data-id]'); if (!row) return;
  const { kind, id } = row.dataset;
  const actEl = t.closest('[data-act]'); if (!actEl) return;
  const act = actEl.dataset.act;
  if (kind === 'print') {
    const p = printById[id], s = state.print[id];
    if ((act === 'done' || act === 'printing') && e.type === 'click' && t.dataset.d) {
      const d = Number(t.dataset.d);
      if (act === 'done') {
        s.done = Math.max(0, Math.min(p.qty, s.done + d));
        if (d > 0 && s.printing > 0) s.printing--;          // a print came off the bed
      } else s.printing = Math.max(0, Math.min(p.qty - s.done, s.printing + d));
      s.printing = Math.min(s.printing, p.qty - s.done);
      return changed();
    }
    if (act === 'colour' && e.type === 'change') { setColour(id, t.value); return changed(); }
    if (act === 'swatch' && e.type === 'click') { setColour(id, actEl.dataset.c); return changed(); }
    if (act === 'notes' && e.type === 'input') { s.notes = t.value; return changed({ render: false }); }
  } else if (kind === 'buy') {
    const s = state.buy[id];
    if (act === 'status' && e.type === 'click' && t.dataset.s) { s.status = t.dataset.s; return changed(); }
    if (act === 'notes' && e.type === 'input') { s.notes = t.value; return changed({ render: false }); }
  }
}
function setColour(id, cid) { if (cid) state.colours[id] = cid; else delete state.colours[id]; }

function setupControls() {
  for (const sel of ['#print-list', '#buy-list', '#selection'])
    for (const ev of ['click', 'change', 'input']) $(sel).addEventListener(ev, onAction);

  for (const sel of ['#print-list', '#buy-list', '#selection']) $(sel).addEventListener('click', onViewAction);
  $('#show-all').addEventListener('click', clearFocus);
  $('#layers').addEventListener('click', onLayerAction);
  $('#layers').addEventListener('toggle', e => {
    if (e.target.id !== 'layers-all' || layersOpen === e.target.open) return;
    layersOpen = e.target.open; renderLayers();   // the short "changed" list only shows while this is closed
  }, true);
  document.addEventListener('keydown', e => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === 'Escape') { viewer?.closePop(); return clearFocus(); }
    if (e.ctrlKey || e.metaKey || e.altKey || !selected || focused || !modelKeys.has(selected)) return;
    if (e.key === 't' || e.key === 'T') setLayer(selected, 'ghost');
    else if (e.key === 'h' || e.key === 'H') setLayer(selected, 'hidden');
  });
  $$('nav button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
  $('#print-filter').addEventListener('click', e => { if (e.target.dataset.f) { state.ui.printFilter = e.target.dataset.f; changed(); } });
  $('#print-group').addEventListener('click', e => { if (e.target.dataset.g) { state.ui.printGroup = e.target.dataset.g; changed(); } });
  $('#buy-filter').addEventListener('click', e => { if (e.target.dataset.f) { state.ui.buyFilter = e.target.dataset.f; changed(); } });
  $('#include-optional').addEventListener('change', e => { state.ui.includeOptional = e.target.checked; changed(); });
  $('#view-mode').addEventListener('click', e => { if (e.target.dataset.mode) { state.ui.viewMode = e.target.dataset.mode; syncViewControls(); changed({ render: false }); } });
  $('#show-bought').addEventListener('change', e => { state.ui.showBought = e.target.checked; changed({ render: false }); });
  $('#reset-view').addEventListener('click', () => viewer?.resetView());

  const pal = $('#palette');
  pal.addEventListener('input', e => {
    const row = e.target.closest('[data-cid]'); if (!row) return;
    const c = state.palette.find(x => x.id === row.dataset.cid);
    if (e.target.dataset.pal === 'hex') c.hex = e.target.value;
    if (e.target.dataset.pal === 'name') c.name = e.target.value;
    renderPrint(); renderSelection(); changed({ render: false });
  });
  pal.addEventListener('click', e => {
    if (e.target.dataset.pal !== 'del') return;
    const cid = e.target.closest('[data-cid]').dataset.cid;
    const used = Object.values(state.colours).filter(v => v === cid).length;
    if (used && !confirm(`${used} part type(s) use this colour. Remove it anyway?`)) return;
    state.palette = state.palette.filter(x => x.id !== cid);
    for (const k in state.colours) if (state.colours[k] === cid) delete state.colours[k];
    changed();
  });
  $('#add-colour').addEventListener('click', () => {
    state.palette.push({ id: 'c' + Date.now().toString(36), name: 'New colour', hex: '#7f8c8d' });
    changed();
  });
  $('#designer-colours').addEventListener('click', () => {
    if (!confirm("Replace your colour choices with the designer's scheme?")) return;
    const ids = new Set(state.palette.map(c => c.id));
    for (const c of catalog.palette) if (!ids.has(c.id)) state.palette.push(structuredClone(c));
    state.colours = { ...catalog.designer_colours };
    changed();
  });
  syncViewControls();
}
function syncViewControls() {
  const mode = state.ui.viewMode || 'colour';
  $$('#view-mode button').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  $('#show-bought').checked = state.ui.showBought !== false;
}
function showTab(tab) {
  $$('nav button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  $$('.tab').forEach(s => s.classList.toggle('on', s.id === 'tab-' + tab));
  try { localStorage.setItem('duck-tab', tab); } catch { /* storage unavailable */ }
  history.replaceState(null, '', '#' + tab);
}

// -------------------------------------------------------------- 3D view -----
const SERVO_KEY = 'buy:servo_sts3215';
const SERVO_IDS = {  // the IDs written by setup_all_servos.py (the runtime's joint order)
  right_hip_yaw: 10, right_hip_roll: 11, right_hip_pitch: 12, right_knee: 13, right_ankle: 14,
  left_hip_yaw: 20, left_hip_roll: 21, left_hip_pitch: 22, left_knee: 23, left_ankle: 24,
  neck_pitch: 30, head_pitch: 31, head_yaw: 32, head_roll: 33,
};
const SHAFT = [-0.0255, 0.0094, 0];  // output shaft in the STS3215 model's own frame, measured from the CAD
const jointText = j => j.replace(/_/g, ' ');

function applyOrigin(obj, el) {
  if (!el) return;
  const xyz = (el.getAttribute('xyz') || '0 0 0').trim().split(/\s+/).map(Number);
  const rpy = (el.getAttribute('rpy') || '0 0 0').trim().split(/\s+/).map(Number);
  obj.position.set(...xyz);
  obj.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');  // URDF fixed-axis roll-pitch-yaw
}

function labelOf(key) {
  const [kind, id] = key.split(/:(.*)/s);
  if (kind === 'print') return printById[id].name;
  if (kind === 'buy') return buyById[id]?.name || id;
  return id;
}

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
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Build the robot from the URDF in the official repo (zero pose).
  const xml = new DOMParser().parseFromString(await (await fetch('robot/robot.urdf')).text(), 'application/xml');
  const links = {}, children = new Set(), visuals = [], jointGroups = {};
  for (const l of xml.querySelectorAll('robot > link')) links[l.getAttribute('name')] = new THREE.Group();
  for (const j of xml.querySelectorAll('robot > joint')) {
    const parent = j.querySelector('parent').getAttribute('link'), child = j.querySelector('child').getAttribute('link');
    const jg = new THREE.Group(); applyOrigin(jg, j.querySelector(':scope > origin'));
    links[parent].add(jg); jg.add(links[child]); children.add(child);
    if (j.getAttribute('name') in SERVO_IDS) jointGroups[j.getAttribute('name')] = jg;
  }
  const root = new THREE.Group(); root.rotation.x = -Math.PI / 2;  // URDF is Z-up
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
  await Promise.all(files.map(async f => {
    geoms[f] = await loader.loadAsync('robot/' + f);
    $('#loading').textContent = `Loading model… ${++n}/${files.length}`;
  }));

  const meshes = [], inst = {};
  for (const v of visuals) {
    const base = v.file.replace(/\.stl$/i, '');
    const key = catalog.mesh_map[base] || 'other:' + base;
    inst[key] = (inst[key] ?? -1) + 1;
    const cad = v.rgba ? new THREE.Color().setRGB(v.rgba[0], v.rgba[1], v.rgba[2], THREE.SRGBColorSpace) : new THREE.Color(0x999999);
    const mesh = new THREE.Mesh(geoms[v.file], new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.05 }));
    mesh.userData = { key, idx: inst[key], cad, file: v.file };
    v.holder.add(mesh); meshes.push(mesh);
  }
  scene.add(root);
  modelKeys = new Set(meshes.map(m => m.userData.key));  // exactly what the model contains
  renderPrint(); renderBuy(); renderLayers();
  root.updateMatrixWorld(true);

  // Which motor is which: each servo's 4 body/horn meshes share one frame, and the joint it drives sits
  // exactly on the output shaft in that frame. The passive horn has its own frame, so it takes the ID of
  // the nearest servo body in the same link.
  {
    const shaft = new THREE.Vector3(...SHAFT), jp = Object.entries(jointGroups).map(([n, g]) => [n, g.getWorldPosition(new THREE.Vector3())]);
    const servos = meshes.filter(m => m.userData.key === SERVO_KEY), bodies = [];
    for (const m of servos) {
      if (m.userData.file === 'passive_palonier.stl') continue;
      let best = null;
      for (const [n, p] of jp) { const d = m.parent.worldToLocal(p.clone()).distanceTo(shaft); if (!best || d < best[0]) best = [d, n]; }
      if (best && best[0] < 0.002) { m.userData.joint = best[1]; bodies.push(m); }
    }
    for (const m of servos) if (m.userData.file === 'passive_palonier.stl') {
      const p = m.getWorldPosition(new THREE.Vector3()); let best = null;
      for (const b of bodies) if (b.parent.parent === m.parent.parent) {
        const d = b.getWorldPosition(new THREE.Vector3()).distanceTo(p); if (!best || d < best[0]) best = [d, b.userData.joint];
      }
      if (best) m.userData.joint = best[1];
    }
  }
  const box = new THREE.Box3().setFromObject(root), c = box.getCenter(new THREE.Vector3());
  root.position.set(-c.x, -box.min.y, -c.z);
  const h = box.max.y - box.min.y;
  const ground = new THREE.Mesh(new THREE.CircleGeometry(h * 0.6, 48), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.06 }));
  ground.rotation.x = -Math.PI / 2; scene.add(ground);
  $('#loading').remove();

  const resetView = () => {
    controls.target.set(0, h * 0.5, 0);
    camera.position.set(h * 2.0, h * 0.9, h * 0.7);
    controls.update();
  };
  resetView();

  new ResizeObserver(() => {
    const w = wrap.clientWidth, ht = wrap.clientHeight; if (!w || !ht) return;
    renderer.setSize(w, ht, false); camera.aspect = w / ht; camera.updateProjectionMatrix();
  }).observe(wrap);

  // Picking: click (not drag) selects; hover shows the part name.
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(), tip = $('#hover-tip');
  const pickHit = ev => {
    const r = renderer.domElement.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    // ghosted parts are click-through, so you can reach whatever is behind them
    const pickable = meshes.filter(m => m.visible && (layerOf(m.userData.key) === 'solid' || m.userData.key === focused));
    return ray.intersectObjects(pickable, false)[0];
  };
  const pick = ev => pickHit(ev)?.object;
  const motorText = m => m.userData.joint ? `Motor ID ${SERVO_IDS[m.userData.joint]} · ${jointText(m.userData.joint)}` : '';

  // Popup pinned to the clicked spot on the part; follows it as the view moves.
  const pop = $('#part-pop');
  let popAt = null, popKey = null;   // anchor in the clicked mesh's own frame, so it stays on the part
  const showPop = hit => {
    const m = hit.object, [kind, id] = m.userData.key.split(/:(.*)/s);
    popAt = { mesh: m, local: m.worldToLocal(hit.point.clone()) }; popKey = m.userData.key;
    $('.pop-name', pop).textContent = labelOf(popKey);
    const motor = motorText(m);
    const qty = kind === 'print' ? printById[id]?.qty : kind === 'buy' ? buyById[id]?.qty : null;
    $('.pop-sub', pop).textContent = motor || [kind === 'print' ? 'Printed part' : kind === 'buy' ? 'Bought part' : 'Comes with another part', qty > 1 ? `×${qty} in the robot` : ''].filter(Boolean).join(' · ');
    $('.pop-sub', pop).classList.toggle('motor', !!motor);
    pop.hidden = false; placePop();
  };
  const closePop = () => { popAt = null; popKey = null; pop.hidden = true; };
  $('.pop-x', pop).addEventListener('click', closePop);
  const v3 = new THREE.Vector3();
  function placePop() {
    if (!popAt) return;
    // close it when its part is no longer the selection (deselected, ghosted, turned off, Esc...)
    if (popKey !== selected || !popAt.mesh.visible) return closePop();
    v3.copy(popAt.local); popAt.mesh.localToWorld(v3); v3.project(camera);
    const off = v3.z > 1 || Math.abs(v3.x) > 1 || Math.abs(v3.y) > 1;
    pop.style.visibility = off ? 'hidden' : '';
    const W = wrap.clientWidth, x = (v3.x + 1) / 2 * W, y = (1 - v3.y) / 2 * wrap.clientHeight;
    const pw = pop.offsetWidth, below = y - pop.offsetHeight - 16 < 0;   // no room above: open underneath
    const cx = Math.min(Math.max(x, pw / 2 + 6), W - pw / 2 - 6);       // keep it inside the viewer
    pop.classList.toggle('below', below);
    pop.style.left = `${Math.round(cx)}px`;   // whole pixels: crisp, no jitter
    pop.style.top = `${Math.round(y)}px`;
    pop.style.setProperty('--arrow', `${Math.round(x - cx)}px`);         // arrow still points at the spot
  }

  let down = null, lastHover = 0;
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const hit = pickHit(e);
    selected = hit ? hit.object.userData.key : null;
    if (focused) focused = selected;   // stay in "show only" mode, switch to the clicked part
    renderSelection(); recolour();
    if (hit) showPop(hit); else closePop();
  });
  renderer.domElement.addEventListener('dblclick', e => {
    const hit = pick(e);
    if (hit) focusPart(hit.userData.key);
  });
  renderer.domElement.addEventListener('pointermove', e => {
    if (e.buttons || e.pointerType !== 'mouse') { tip.style.display = 'none'; return; }
    const now = performance.now(); if (now - lastHover < 80) return; lastHover = now;
    const hit = pick(e);
    if (!hit) { tip.style.display = 'none'; renderer.domElement.style.cursor = ''; return; }
    const r = wrap.getBoundingClientRect();
    tip.textContent = labelOf(hit.userData.key) + (hit.userData.joint ? ` · ID ${SERVO_IDS[hit.userData.joint]}` : '');
    tip.style.display = 'block'; tip.style.left = `${e.clientX - r.left + 12}px`; tip.style.top = `${e.clientY - r.top + 12}px`;
    renderer.domElement.style.cursor = 'pointer';
  });
  renderer.domElement.addEventListener('pointerleave', () => { tip.style.display = 'none'; });

  const GHOST = 0.12, FOCUS_GHOST = 0.07, LAYER_GHOST = 0.1;
  function recolour() {
    const mode = state.ui.viewMode || 'colour', showBought = state.ui.showBought !== false;
    for (const m of meshes) {
      const { key, idx, cad } = m.userData, [kind, id] = key.split(/:(.*)/s), mat = m.material;
      const layer = key === focused ? 'solid' : layerOf(key);
      let colour = cad, opacity = 1, emissive = 0x000000;
      m.visible = (kind === 'print' || showBought || key === focused) && layer !== 'hidden';
      if (kind === 'print') {
        const pc = colourOf(id);
        colour = pc ? new THREE.Color(pc.hex) : new THREE.Color(0xcfcfcf);
        if (mode === 'progress') {
          const s = state.print[id];
          if (idx >= s.done) { if (idx < s.done + s.printing) emissive = 0x7a4300; else opacity = GHOST; }
        }
      } else if (mode === 'progress') {
        const st = kind === 'buy' ? state.buy[id]?.status : 'need';
        if (st === 'need') opacity = GHOST; else if (st === 'ordered') opacity = 0.45;
      }
      if (layer === 'ghost') { opacity = Math.min(opacity, LAYER_GHOST); emissive = 0x000000; }
      if (focused) {
        if (key === focused) { opacity = 1; emissive = 0x000000; }
        else { opacity = FOCUS_GHOST; emissive = 0x000000; }
      }
      mat.color.copy(colour);
      mat.emissive.setHex(key === selected && !focused ? 0x1f4fa8 : emissive);
      mat.transparent = opacity < 1; mat.opacity = opacity; mat.depthWrite = opacity === 1;
      mat.needsUpdate = true;
    }
    const lg = $('#legend');
    if (focused) {
      const perFile = {};
      for (const m of meshes) if (m.userData.key === focused) perFile[m.userData.file] = (perFile[m.userData.file] || 0) + 1;
      const n = Math.max(0, ...Object.values(perFile));  // a servo is several meshes; count copies, not meshes
      lg.innerHTML = `<span><b>Showing only:</b> ${esc(labelOf(focused))}${n > 1 ? ` (all ${n} in the model)` : ''}</span><span class="muted">Click another part to switch · Esc or “Show all parts” to go back</span>`;
    } else if (mode === 'progress') {
      lg.innerHTML = `<span>Solid = printed / have</span><span><i style="background:#c77a1a"></i>Glowing = printing</span><span>Faded = ordered</span><span>Ghost = still to print / buy</span>`;
    } else {
      const used = new Set(Object.values(state.colours));
      lg.innerHTML = state.palette.filter(c => used.has(c.id)).map(c => `<span><i style="background:${esc(c.hex)}"></i>${esc(c.name)}</span>`).join('')
        + '<span><i style="background:#cfcfcf"></i>No colour</span><span class="muted">Click a part to select it</span>';
    }
    const nLayer = Object.keys(state.ui.layers).filter(k => modelKeys.has(k)).length;
    if (!focused && nLayer) lg.innerHTML += `<span class="muted">· ${nLayer} part type${nLayer > 1 ? 's' : ''} ghosted or off (clicks pass through ghosts)</span>`;
  }

  let tween = null;
  function frame(key) {
    const ms = meshes.filter(m => m.userData.key === key && m.visible);
    if (!ms.length) return;
    const box = new THREE.Box3();
    for (const m of ms) box.expandByObject(m);
    const centre = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length();
    const dir = camera.position.clone().sub(controls.target).normalize();
    const dist = Math.max(size * 1.6, h * 0.35) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    tween = { t0: performance.now(), from: [controls.target.clone(), camera.position.clone()],
              to: [centre, centre.clone().add(dir.multiplyScalar(dist))] };
  }
  (function loop() {
    if (tween) {
      const k = Math.min(1, (performance.now() - tween.t0) / 450), e = 1 - Math.pow(1 - k, 3);
      controls.target.lerpVectors(tween.from[0], tween.to[0], e);
      camera.position.lerpVectors(tween.from[1], tween.to[1], e);
      if (k === 1) tween = null;
    }
    controls.update(); renderer.render(scene, camera); placePop(); requestAnimationFrame(loop);
  })();
  viewer = { recolour, resetView, frame, closePop };
  recolour();
  if (pendingFocus) { const k = pendingFocus; pendingFocus = null; frame(k); }
}

// ---------------------------------------------------------------- boot ------
(async () => {
  catalog = await (await fetch('data/catalog.json')).json();
  catalog.print.forEach(p => { printById[p.id] = p; });
  catalog.buy.forEach(b => { buyById[b.id] = b; });
  modelKeys = new Set([...Object.values(catalog.mesh_map).filter(k => !k.startsWith('print:')),
                       ...catalog.print.filter(p => p.in_model).map(p => 'print:' + p.id)]);
  let saved = {};
  try { saved = await (await fetch('api/progress')).json(); } catch { /* first run */ }
  state = normalise(saved);
  setupControls();
  let tab = location.hash.slice(1);
  if (!['model', 'print', 'buy', 'docs'].includes(tab)) {
    tab = 'model';
    try { tab = localStorage.getItem('duck-tab') || tab; } catch { /* storage unavailable */ }
  }
  showTab(tab);
  renderAll();
  $('#save-state').textContent = state.updated ? `Last saved ${new Date(state.updated).toLocaleString()}` : '';
  initViewer().catch(err => { $('#loading').textContent = 'Could not load the 3D model: ' + err.message; });
})();
