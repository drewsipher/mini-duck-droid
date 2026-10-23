// The 3D duck: the URDF from official_repo posed by joint angles. Shared by the Duck Puppet
// (puppet.js) and the dashboard's animation player (dash.js). No DOM outside `wrap`.
//
//   const v = await createDuckViewer(wrap, { onPick: joint => ..., loading: el, tipText: j => '...' });
//   v.pose({ left_knee: 0.4, ... })   radians per joint (missing joints keep their angle)
//   v.highlight(joint | null)         glow the servo that drives this joint
//   v.resetView()                     camera back to the three-quarter view
import * as THREE from 'three';
import { STLLoader } from 'three/addons/STLLoader.js';
import { OrbitControls } from 'three/addons/OrbitControls.js';

export const SERVO_IDS = { right_hip_yaw: 10, right_hip_roll: 11, right_hip_pitch: 12, right_knee: 13, right_ankle: 14,
  left_hip_yaw: 20, left_hip_roll: 21, left_hip_pitch: 22, left_knee: 23, left_ankle: 24,
  neck_pitch: 30, head_pitch: 31, head_yaw: 32, head_roll: 33 };
const SERVO_KEY = 'buy:servo_sts3215';
const SHAFT = [-0.0255, 0.0094, 0];  // output shaft in the STS3215 model's own frame

export async function createDuckViewer(wrap, { onPick, loading, tipText } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  wrap.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 1.8));
  const key = new THREE.DirectionalLight(0xffffff, 1.8); key.position.set(1.2, 2, 1.5); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7); fill.position.set(-1.5, 0.6, -1); scene.add(fill);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  const catalog = await (await fetch('/data/catalog.json')).json();
  const xml = new DOMParser().parseFromString(await (await fetch('/robot/robot.urdf')).text(), 'application/xml');
  const applyOrigin = (obj, el) => {
    if (!el) return;
    const xyz = (el.getAttribute('xyz') || '0 0 0').trim().split(/\s+/).map(Number), rpy = (el.getAttribute('rpy') || '0 0 0').trim().split(/\s+/).map(Number);
    obj.position.set(xyz[0], xyz[1], xyz[2]); obj.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');
  };
  const links = {}, children = new Set(), visuals = [], groups = {};
  for (const l of xml.querySelectorAll('robot > link')) links[l.getAttribute('name')] = new THREE.Group();
  for (const j of xml.querySelectorAll('robot > joint')) {
    const name = j.getAttribute('name'), parent = j.querySelector('parent').getAttribute('link'), child = j.querySelector('child').getAttribute('link');
    const jg = new THREE.Group(); applyOrigin(jg, j.querySelector(':scope > origin'));
    const axis = new THREE.Group();                 // every revolute joint turns about its own z
    links[parent].add(jg); jg.add(axis); axis.add(links[child]); children.add(child);
    if (name in SERVO_IDS) groups[name] = axis;
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
  await Promise.all(files.map(async f => { geoms[f] = await loader.loadAsync('/robot/' + f); if (loading) loading.textContent = `Loading model… ${++n}/${files.length}`; }));
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
    const shaft = new THREE.Vector3(...SHAFT), jp = Object.entries(groups).map(([nm, g]) => [nm, g.getWorldPosition(new THREE.Vector3())]);
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
  if (loading) loading.remove();
  const resetView = () => { controls.target.set(0, h * 0.5, 0); camera.position.set(h * 2.0, h * 0.9, h * 0.7); controls.update(); };
  resetView();
  new ResizeObserver(() => { const w = wrap.clientWidth, ht = wrap.clientHeight; if (!w || !ht) return; renderer.setSize(w, ht, false); camera.aspect = w / ht; camera.updateProjectionMatrix(); }).observe(wrap);
  let lit = null;
  const highlight = joint => { lit = joint; for (const m of meshes) { const on = m.userData.joint && m.userData.joint === joint; m.material.emissive.set(on ? 0xd99a00 : 0x000000); m.material.emissiveIntensity = on ? 0.55 : 0; } };
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  let tip = wrap.querySelector('#hover-tip, .hover-tip');
  if (!tip) { tip = document.createElement('div'); tip.className = 'hover-tip'; wrap.appendChild(tip); }
  const pickHit = ev => { const r = renderer.domElement.getBoundingClientRect(); ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ptr, camera); return ray.intersectObjects(meshes, false)[0]; };
  let down = null;
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => { if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return; const j = pickHit(e)?.object.userData.joint; if (j && onPick) onPick(j); });
  renderer.domElement.addEventListener('pointermove', e => {
    if (e.buttons || e.pointerType !== 'mouse') { tip.style.display = 'none'; return; }
    const j = pickHit(e)?.object.userData.joint;
    if (!j) { tip.style.display = 'none'; return; }
    const r = wrap.getBoundingClientRect();
    tip.textContent = tipText ? tipText(j) : `${j.replace(/_/g, ' ')} · ID ${SERVO_IDS[j]}`; tip.style.display = 'block'; tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
  });
  renderer.domElement.addEventListener('pointerleave', () => (tip.style.display = 'none'));
  let alive = true;
  (function animate() { if (!alive) return; controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); })();
  return {
    groups, meshes, scene, camera, renderer, controls, resetView, highlight, height: h,
    get highlighted() { return lit; },
    pose(p) { for (const j in p) if (p[j] != null && groups[j]) groups[j].rotation.z = p[j]; },
    dispose() { alive = false; renderer.dispose(); renderer.domElement.remove(); },
  };
}
