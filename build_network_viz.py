"""
Build an interactive 3D visualization of the trained Hebbian network,
using YOUR REAL FILES (not the synthetic generators).

Run this after confirming train_on_real_files.py works. It retrains a
fresh network on your real photos/recordings, figures out which neurons
learned which class, extracts the strongest synapses, and writes a
self-contained HTML file you can open directly in a browser.

Expects the same six files as train_on_real_files.py, in the same folder:
    can.jpeg   can.m4a
    cup.jpeg   cup.m4a
    controller.jpeg   controller.m4a

Output: network_3d_map.html  (open it directly, no server needed)
"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hebbian_binding import *   # noqa

FILES = {
    "can":        ("can.jpeg", "can.m4a"),
    "cup":        ("cup.jpeg", "cup.m4a"),
    "controller": ("controller.jpeg", "controller.m4a"),
}

print("Training on real files (this retrains fresh, ~10-30s)...")
net = Net()
base = train_on_real_files(net, FILES, n_per_class=120, shuffle_pairs=False)

# ---- association-layer class assignment (vision-based, matches eval script) ----
def assign_readout_real(net, base, n=40):
    resp = np.zeros((ASSOC, N_CLASS), np.float32)
    for c in range(N_CLASS):
        for _ in range(n):
            a = net.present(augment(base[c][0]), base[c][1], learn=False,
                             vis_on=True, aud_on=False)
            resp[:, c] += a
    assign = resp.argmax(1)
    active = resp.sum(1) > 0
    return assign, active

assoc_assign, assoc_active = assign_readout_real(net, base)

# ---- per-layer (vis_feat / aud_feat) class preference ----
def layer_class_pref_real(run_fn, get_sample_fn, n_units, n=30):
    resp = np.zeros((n_units, N_CLASS), np.float32)
    for c in range(N_CLASS):
        for _ in range(n):
            counts = run_fn(get_sample_fn(c))
            resp[:, c] += counts
    assign = resp.argmax(1)
    active = resp.sum(1) > 0
    return assign, active

vis_assign, vis_active = layer_class_pref_real(
    lambda vec: net.vis.run(rate_to_spikes(vec), learn=False),
    lambda c: augment(base[c][0]), VIS_FEAT)

aud_assign, aud_active = layer_class_pref_real(
    lambda vec: net.aud.run(rate_to_spikes(vec), learn=False),
    lambda c: augment(base[c][1]), AUD_FEAT)

print(f"vis active: {vis_active.sum()}/{VIS_FEAT}   "
      f"aud active: {aud_active.sum()}/{AUD_FEAT}   "
      f"assoc active: {assoc_active.sum()}/{ASSOC}")

# ---- colors: auto-generate one per class (works for any number/names of classes) ----
PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
CLASS_COLORS = [PALETTE[i % len(PALETTE)] for i in range(N_CLASS)]
GRAY = "#7f8c8d"

def color_for(assign, active, idx):
    return CLASS_COLORS[assign[idx]] if active[idx] else GRAY

# ---- layout ----
def ring(n, cx, cy, cz, r, y_scale=0.55):
    pts = []
    for i in range(n):
        a = 2*math.pi*i/n
        pts.append((cx + r*math.cos(a), cy + r*math.sin(a)*y_scale, cz))
    return pts

vis_pos   = ring(VIS_FEAT, cx=-7, cy=0, cz=-4, r=4.2)
aud_pos   = ring(AUD_FEAT, cx= 7, cy=0, cz=-4, r=3.2)
assoc_pos = ring(ASSOC,    cx= 0, cy=0, cz= 4, r=3.0)

def f3(v): return float(v)

nodes = []
for i in range(VIS_FEAT):
    x,y,z = vis_pos[i]
    nodes.append(dict(id=f"v{i}", x=f3(x),y=f3(y),z=f3(z), r=0.12,
                       color=color_for(vis_assign, vis_active, i), layer="vision"))
for i in range(AUD_FEAT):
    x,y,z = aud_pos[i]
    nodes.append(dict(id=f"a{i}", x=f3(x),y=f3(y),z=f3(z), r=0.14,
                       color=color_for(aud_assign, aud_active, i), layer="audio"))
for i in range(ASSOC):
    x,y,z = assoc_pos[i]
    nodes.append(dict(id=f"s{i}", x=f3(x),y=f3(y),z=f3(z), r=0.22,
                       color=color_for(assoc_assign, assoc_active, i), layer="assoc"))

# ---- edges: top-K strongest connections into each assoc neuron ----
W_va = net.assoc.W[:, :VIS_FEAT]
W_aa = net.assoc.W[:, VIS_FEAT:VIS_FEAT+AUD_FEAT]

edges = []
K = 4
wmax = max(W_va.max(), W_aa.max(), 1e-6)
for j in range(ASSOC):
    if not assoc_active[j]:
        continue
    tgt_class = assoc_assign[j]
    top_v = np.argsort(-W_va[j])[:K]
    for i in top_v:
        w = float(W_va[j, i]) / wmax
        if w < 0.05: continue
        same = vis_active[i] and vis_assign[i] == tgt_class
        edges.append(dict(a=f"v{i}", b=f"s{j}", w=w,
                           color=CLASS_COLORS[tgt_class] if same else GRAY,
                           same=bool(same)))
    top_a = np.argsort(-W_aa[j])[:K]
    for i in top_a:
        w = float(W_aa[j, i]) / wmax
        if w < 0.05: continue
        same = aud_active[i] and aud_assign[i] == tgt_class
        edges.append(dict(a=f"a{i}", b=f"s{j}", w=w,
                           color=CLASS_COLORS[tgt_class] if same else GRAY,
                           same=bool(same)))

same_class_edges = sum(1 for e in edges if e["same"])
print(f"edges: {len(edges)}  same-class: {same_class_edges} "
      f"({same_class_edges/max(len(edges),1)*100:.0f}%)")

data = dict(nodes=nodes, edges=edges, classes=CLASSES, colors=CLASS_COLORS)

def _np_default(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")

data_json = json.dumps(data, default=_np_default)

# ---- embed into the HTML template and write output ----
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hebbian Network — 3D Map</title>
<style>
  html, body { margin:0; padding:0; overflow:hidden; background:#0b0d12; font-family: -apple-system, Helvetica, Arial, sans-serif; }
  #canvas { display:block; width:100vw; height:100vh; }
  #ui {
    position:absolute; top:14px; left:14px; color:#e8e8ec; z-index:10;
    background:rgba(15,17,23,0.78); padding:14px 16px; border-radius:10px;
    max-width:300px; font-size:13px; line-height:1.5; backdrop-filter: blur(4px);
    border:1px solid rgba(255,255,255,0.08);
  }
  #ui h1 { font-size:15px; margin:0 0 8px 0; font-weight:600; }
  .legend-item { display:flex; align-items:center; gap:8px; margin:3px 0; }
  .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  #hint { position:absolute; bottom:14px; left:14px; color:#8a8f9c; font-size:12px; z-index:10; }
  #tooltip {
    position:absolute; pointer-events:none; z-index:20; display:none;
    background:rgba(20,22,28,0.95); color:#fff; padding:6px 10px; border-radius:6px;
    font-size:12px; border:1px solid rgba(255,255,255,0.15);
  }
  #stats { position:absolute; top:14px; right:14px; color:#8a8f9c; font-size:12px; z-index:10; text-align:right; }
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="ui">
  <h1>Hebbian Cross-Modal Network</h1>
  <div class="legend-item"><div class="dot" style="background:#e74c3c"></div>can — visual + auditory neurons</div>
  <div class="legend-item"><div class="dot" style="background:#3498db"></div>pen — visual + auditory neurons</div>
  <div class="legend-item"><div class="dot" style="background:#2ecc71"></div>cup — visual + auditory neurons</div>
  <div class="legend-item"><div class="dot" style="background:#7f8c8d"></div>unassigned / weak</div>
  <div style="margin-top:10px; opacity:0.85;">
    Back rings: vision (left) &amp; audio (right) feature neurons, self-organized by STDP.<br>
    Front ring: cross-modal association neurons — bound by co-occurrence, not labels.<br>
    Lines = the strongest learned synapses feeding each association neuron.
  </div>
</div>
<div id="stats"></div>
<div id="hint">drag to rotate · scroll to zoom · hover a neuron</div>
<div id="tooltip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = __DATA_JSON__;

// ---------------- Scene setup ----------------
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 200);
camera.position.set(0, 6, 22);
camera.lookAt(0,0,0);

const ambient = new THREE.AmbientLight(0xffffff, 0.9);
scene.add(ambient);
const light1 = new THREE.PointLight(0xffffff, 0.6);
light1.position.set(10,10,10); scene.add(light1);

const group = new THREE.Group();
scene.add(group);

// ---------------- Build nodes ----------------
const nodeMeshes = {};
const geomSphere = new THREE.SphereGeometry(1, 10, 10);
DATA.nodes.forEach(n => {
  const mat = new THREE.MeshStandardMaterial({
    color: n.color, emissive: n.color, emissiveIntensity: 0.35, roughness:0.5
  });
  const mesh = new THREE.Mesh(geomSphere, mat);
  mesh.position.set(n.x, n.y, n.z);
  mesh.scale.setScalar(n.r);
  mesh.userData = n;
  group.add(mesh);
  nodeMeshes[n.id] = mesh;
});

// ---------------- Build edges ----------------
DATA.edges.forEach(e => {
  const a = nodeMeshes[e.a], b = nodeMeshes[e.b];
  if (!a || !b) return;
  const pts = [a.position.clone(), b.position.clone()];
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({
    color: e.color, transparent:true,
    opacity: e.same ? Math.max(0.15, e.w*0.85) : Math.max(0.04, e.w*0.18)
  });
  const line = new THREE.Line(geom, mat);
  group.add(line);
});

// ---------------- Layer labels (simple sprites via canvas texture) ----------------
function makeLabel(text, x, y, z, color) {
  const cnv = document.createElement('canvas');
  cnv.width = 512; cnv.height = 96;
  const ctx = cnv.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,0)'; ctx.fillRect(0,0,512,96);
  ctx.font = 'bold 48px -apple-system, Arial'; ctx.fillStyle = color;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 256, 48);
  const tex = new THREE.CanvasTexture(cnv);
  const mat = new THREE.SpriteMaterial({map:tex, transparent:true});
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(6,1.1,1);
  sprite.position.set(x,y,z);
  group.add(sprite);
}
makeLabel('VISION', -7, 5.2, -4, '#cfd3da');
makeLabel('AUDIO', 7, 4.2, -4, '#cfd3da');
makeLabel('ASSOCIATION (bound concepts)', 0, 4.6, 4, '#f0f0f4');

// ---------------- Stats ----------------
document.getElementById('stats').innerHTML =
  `${DATA.nodes.length} neurons &nbsp;·&nbsp; ${DATA.edges.length} shown synapses`;

// ---------------- Mouse drag rotate + zoom ----------------
let isDown = false, lastX = 0, lastY = 0;
let rotX = 0.15, rotY = 0.3;
canvas.addEventListener('mousedown', e => { isDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => isDown = false);
window.addEventListener('mousemove', e => {
  if (!isDown) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  rotY += dx * 0.006; rotX += dy * 0.006;
  rotX = Math.max(-1.4, Math.min(1.4, rotX));
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', e => {
  camera.position.z = Math.max(6, Math.min(60, camera.position.z + e.deltaY*0.02));
}, {passive:true});

// touch support
let touchLast = null;
canvas.addEventListener('touchstart', e => { touchLast = e.touches[0]; }, {passive:true});
canvas.addEventListener('touchmove', e => {
  if (!touchLast) return;
  const t = e.touches[0];
  const dx = t.clientX - touchLast.clientX, dy = t.clientY - touchLast.clientY;
  rotY += dx * 0.006; rotX += dy * 0.006;
  rotX = Math.max(-1.4, Math.min(1.4, rotX));
  touchLast = t;
}, {passive:true});
canvas.addEventListener('touchend', () => touchLast = null);

// ---------------- Hover tooltip via raycasting ----------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');
canvas.addEventListener('mousemove', e => {
  mouse.x = (e.clientX/window.innerWidth)*2 - 1;
  mouse.y = -(e.clientY/window.innerHeight)*2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(group.children.filter(c => c.geometry && c.geometry.type==='SphereGeometry'));
  if (hits.length) {
    const n = hits[0].object.userData;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX+14)+'px';
    tooltip.style.top = (e.clientY+10)+'px';
    tooltip.innerHTML = `<b>${n.id}</b> · ${n.layer} layer`;
  } else {
    tooltip.style.display = 'none';
  }
});

// slow auto-rotate when idle, resumes after user interaction pause
let idleTimer = 0;

function animate() {
  requestAnimationFrame(animate);
  if (!isDown) { idleTimer += 1; if (idleTimer > 60) rotY += 0.0015; }
  else idleTimer = 0;
  group.rotation.y = rotY;
  group.rotation.x = rotX;
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""

out_html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network_3d_map.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out_html)

print(f"\nWrote {out_path}")
print("Open it directly in your browser (double-click it) -- no server needed.")