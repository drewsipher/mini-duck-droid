// Keyframe animation for the duck: pure functions, no DOM. Used by puppet.js and testable in node.
// A clip is { keys: { joint: [{t, v}, ...] sorted by t }, interp: 'smooth' | 'linear' }.
// Dense frames (what the agent plays and what a recording is) are [[t, {joint: rad}], ...].

export const JOINTS = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
  'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
  'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle'];

// Which joints swap and which change sign when the duck is mirrored left <-> right. The runtime
// frame has opposite signs for the two hip pitches (left -0.63 / right +0.635 when standing) and
// for the two hip rolls and yaws; knees and ankles share a sign.
const MIRROR = {
  left_hip_yaw: ['right_hip_yaw', -1], right_hip_yaw: ['left_hip_yaw', -1],
  left_hip_roll: ['right_hip_roll', -1], right_hip_roll: ['left_hip_roll', -1],
  left_hip_pitch: ['right_hip_pitch', -1], right_hip_pitch: ['left_hip_pitch', -1],
  left_knee: ['right_knee', 1], right_knee: ['left_knee', 1],
  left_ankle: ['right_ankle', 1], right_ankle: ['left_ankle', 1],
  neck_pitch: ['neck_pitch', 1], head_pitch: ['head_pitch', 1],
  head_yaw: ['head_yaw', -1], head_roll: ['head_roll', -1],
};

// The other side's joint and the sign a value takes there.
export function mirrorKey(joint, v) { const [j, s] = MIRROR[joint]; return [j, v * s]; }

export function duration(clip) {
  let d = 0;
  for (const j in clip.keys) for (const k of clip.keys[j]) if (k.t > d) d = k.t;
  return d;
}

// Ramer-Douglas-Peucker on one joint's recording: keep the points a straight line misses by > tol.
function rdp(pts, tol) {
  if (pts.length <= 2) return pts.slice();
  const [a, b] = [pts[0], pts[pts.length - 1]];
  let worst = 0, wi = -1;
  for (let i = 1; i < pts.length - 1; i++) {
    const u = (pts[i].t - a.t) / (b.t - a.t || 1), d = Math.abs(pts[i].v - (a.v + (b.v - a.v) * u));
    if (d > worst) { worst = d; wi = i; }
  }
  if (worst <= tol) return [a, b];
  return rdp(pts.slice(0, wi + 1), tol).slice(0, -1).concat(rdp(pts.slice(wi), tol));
}

// A dense recording -> keyframes. tol in radians (0.025 is about 1.4 degrees).
export function framesToClip(frames, tol = 0.025) {
  const keys = {};
  for (const j of JOINTS) {
    const pts = frames.filter(f => f[1][j] != null).map(f => ({ t: +f[0], v: +f[1][j] }));
    keys[j] = pts.length ? rdp(pts, tol).map(k => ({ t: round3(k.t), v: round4(k.v) })) : [];
  }
  return { keys, interp: 'smooth' };
}

export function normalise(clip) {   // accept a saved dict or a plain recording
  if (Array.isArray(clip)) return framesToClip(clip);
  const out = { keys: {}, interp: clip.interp === 'linear' ? 'linear' : 'smooth' };
  for (const j of JOINTS) out.keys[j] = (clip.keys?.[j] || []).map(k => ({ t: +k.t, v: +k.v })).sort((a, b) => a.t - b.t);
  if (!Object.values(out.keys).some(k => k.length) && clip.frames) return framesToClip(clip.frames);
  return out;
}

const round3 = x => Math.round(x * 1000) / 1000, round4 = x => Math.round(x * 10000) / 10000;

// Value of one joint at time t. Catmull-Rom through the keys (continuous velocity, the way a
// walk should be) or straight lines; flat before the first key and after the last.
export function valueAt(keys, t, interp = 'smooth') {
  const n = keys.length;
  if (!n) return null;
  if (t <= keys[0].t) return keys[0].v;
  if (t >= keys[n - 1].t) return keys[n - 1].v;
  let i = 0;
  while (i < n - 2 && keys[i + 1].t <= t) i++;
  const k1 = keys[i], k2 = keys[i + 1], h = k2.t - k1.t || 1e-6, u = (t - k1.t) / h;
  if (interp === 'linear' || n < 3) return k1.v + (k2.v - k1.v) * u;
  const k0 = keys[Math.max(0, i - 1)], k3 = keys[Math.min(n - 1, i + 2)];
  // tangents scaled by the segment's own length, so uneven key spacing does not kink
  const m1 = i === 0 ? (k2.v - k1.v) / h : (k2.v - k0.v) / (k2.t - k0.t);
  const m2 = i + 2 > n - 1 ? (k2.v - k1.v) / h : (k3.v - k1.v) / (k3.t - k1.t);
  const u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * k1.v + (u3 - 2 * u2 + u) * h * m1 + (-2 * u3 + 3 * u2) * k2.v + (u3 - u2) * h * m2;
}

export function poseAt(clip, t) {
  const pose = {};
  for (const j of JOINTS) { const v = valueAt(clip.keys[j], t, clip.interp); if (v != null) pose[j] = v; }
  return pose;
}

// Dense frames at `hz` for the agent and for saving.
export function sample(clip, hz = 50) {
  const d = duration(clip), frames = [];
  for (let i = 0, n = Math.max(1, Math.round(d * hz)); i <= n; i++) {
    const t = round3(Math.min(d, i / hz)), pose = poseAt(clip, t);
    for (const j in pose) pose[j] = round4(pose[j]);
    frames.push([t, pose]);
  }
  return frames;
}

export function setKey(clip, joint, t, v, snap = 0.02) {
  const ks = clip.keys[joint];
  const i = ks.findIndex(k => Math.abs(k.t - t) <= snap);
  if (i >= 0) { ks[i] = { t: ks[i].t, v: round4(v) }; return i; }
  ks.push({ t: round3(t), v: round4(v) });
  ks.sort((a, b) => a.t - b.t);
  return ks.findIndex(k => k.t === round3(t));
}

export function moveKey(clip, joint, i, t, v) {
  const ks = clip.keys[joint], k = ks[i];
  if (!k) return i;
  k.t = round3(Math.max(0, t)); k.v = round4(v);
  ks.sort((a, b) => a.t - b.t);
  return ks.indexOf(k);
}

export function deleteKey(clip, joint, i) {
  if (clip.keys[joint].length > 1) clip.keys[joint].splice(i, 1);
}

export function scaleTime(clip, factor) {   // factor 0.5 = twice as fast
  for (const j in clip.keys) for (const k of clip.keys[j]) k.t = round3(k.t * factor);
}

export function trim(clip, from, to) {      // keep [from, to], re-anchor to 0, keep the values at the cuts
  for (const j of JOINTS) {
    const ks = clip.keys[j]; if (!ks.length) continue;
    const vFrom = valueAt(ks, from, clip.interp), vTo = valueAt(ks, to, clip.interp);
    const inside = ks.filter(k => k.t > from + 1e-6 && k.t < to - 1e-6).map(k => ({ t: round3(k.t - from), v: k.v }));
    clip.keys[j] = [{ t: 0, v: round4(vFrom) }, ...inside, { t: round3(to - from), v: round4(vTo) }];
  }
}

export function mirror(clip) {              // the same motion done by the other side
  const keys = {};
  for (const j of JOINTS) { const [src, sgn] = MIRROR[j]; keys[j] = (clip.keys[src] || []).map(k => ({ t: k.t, v: round4(k.v * sgn) })); }
  return { keys, interp: clip.interp };
}

export function append(clip, other, gap = 0) {   // `other` starts where this one ends (plus gap)
  const d = duration(clip) + gap;
  for (const j of JOINTS) {
    const more = (other.keys[j] || []).map(k => ({ t: round3(k.t + d), v: k.v }));
    clip.keys[j] = [...clip.keys[j], ...more].sort((a, b) => a.t - b.t);
  }
}

export function offsetJoint(clip, joint, dv) {
  for (const k of clip.keys[joint]) k.v = round4(k.v + dv);
}

// Make a clip loop: cut `from`..`to` out of it, then cross-fade the last `blend` seconds into
// the first frame so the end flows into the start. Keys are rebuilt from the result.
export function makeLoop(clip, from, to, blend = 0.15, hz = 50) {
  const cut = JSON.parse(JSON.stringify(clip));
  trim(cut, from, to);
  const frames = sample(cut, hz), n = frames.length - 1, nb = Math.min(Math.round(blend * hz), Math.floor(n / 2));
  const first = frames[0][1];
  for (let k = 1; k <= nb; k++) {              // the last nb frames bend toward the first frame
    const u = k / (nb + 1), f = frames[n - nb + k - 1][1];
    for (const j in f) if (first[j] != null) f[j] = (1 - u) * f[j] + u * first[j];
  }
  frames[n] = [frames[n][0], { ...first }];    // the last frame is the first again
  const out = framesToClip(frames, 0.015);
  out.interp = clip.interp;
  return out;
}

export function toSaved(clip) {
  return { keys: clip.keys, interp: clip.interp, frames: sample(clip) };
}
