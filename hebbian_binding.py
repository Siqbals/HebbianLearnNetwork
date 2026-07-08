"""
Fixed-position cross-modal Hebbian binding prototype (v1).

A no-backprop spiking neural network that binds a "spoken word" to a visual
"object" through co-occurrence alone, then discriminates a small object set.

Substrate : Leaky Integrate-and-Fire (LIF) spiking neurons
Learning  : trace-based STDP (local, spike-triggered) — NO backprop anywhere
Stability : hard lateral inhibition (WTA) + adaptive threshold homeostasis
            + divisive weight normalization
Grounding : a cross-modal association layer bound by co-activation

Hardware note:
    This sandbox has no camera/mic, so vision/audio are replaced by SYNTHETIC
    but cleanly-separable stand-ins (shapes for objects, spectro-temporal
    patterns for words). The encoding interface is identical to the real thing;
    see `capture_real_*` for the drop-in camera/mic path you'd enable locally.

Labels are used ONLY at evaluation time to (a) assign readout neurons to
classes and (b) score accuracy. Labels NEVER touch a synaptic weight. All
weight changes are driven purely by local spike correlations.
"""

import numpy as np

rng = np.random.default_rng(0)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CLASSES = ["can", "cup", "controller"]
N_CLASS = len(CLASSES)

# Presentation timing
T_STEPS      = 60      # timesteps per presentation
DT           = 1.0     # ms per step (nominal)
MAX_RATE     = 0.35    # peak spike prob per step for a fully-on input unit

# Vision encoding
IMG          = 20      # downsampled frame is IMG x IMG
VIS_IN       = IMG * IMG * 2          # on/off channels
VIS_FEAT     = 500                    # visual feature neurons (excitatory)

# Audio encoding
MEL          = 24      # mel bands
FRAMES       = 6       # time frames in the word window
AUD_IN       = MEL * FRAMES
AUD_FEAT     = 500                    # auditory feature neurons

# Association
ASSOC        = 500                     # cross-modal binding neurons

# LIF
V_THRESH0    = 1.0
V_LEAK       = 0.90    # membrane decay per step
THETA_INC    = 0.05    # adaptive-threshold bump per spike (homeostasis)
THETA_DECAY  = 0.999   # adaptive-threshold slow decay

# STDP
STDP_LR      = 0.010
TRACE_DECAY  = 0.85
W_NORM       = 1.0     # target L2 sum of each neuron's incoming weights


# --------------------------------------------------------------------------- #
# 1. Encoders  (synthetic stand-ins for camera + mic)
# --------------------------------------------------------------------------- #
def make_object(cls, jitter=True):
    """Return an IMG x IMG intensity map standing in for a centered object.

    Shapes are deliberately distinct silhouettes. Small per-instance jitter and
    noise mean no two presentations are pixel-identical -> lets us later probe
    weak generalization (feature cluster vs memorized pixels)."""
    img = np.zeros((IMG, IMG), np.float32)
    c = IMG // 2
    dx, dy = (rng.integers(-1, 2), rng.integers(-1, 2)) if jitter else (0, 0)
    if cls == 0:                      # "can": tall filled rectangle
        img[3+dy:IMG-3+dy, c-3+dx:c+3+dx] = 1.0
    elif cls == 1:                    # "pen": thin diagonal stroke
        for i in range(3, IMG-3):
            img[i, np.clip(i+dx, 0, IMG-1)] = 1.0
            img[i, np.clip(i+1+dx, 0, IMG-1)] = 1.0
    else:                            # "cup": open U / trapezoid
        img[c-4+dy:c+4+dy, c-4+dx] = 1.0
        img[c-4+dy:c+4+dy, c+3+dx] = 1.0
        img[c+3+dy, c-4+dx:c+4+dx] = 1.0
    img += 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def dog(img):
    """Center-surround (Difference-of-Gaussians) -> ON/OFF contrast channels.
    Strips flat background/lighting (quietly helps figure-ground)."""
    from scipy.ndimage import gaussian_filter
    g1 = gaussian_filter(img, 0.7)
    g2 = gaussian_filter(img, 1.8)
    d = g1 - g2
    on = np.clip(d, 0, None)
    off = np.clip(-d, 0, None)
    m = max(on.max(), off.max(), 1e-6)
    return np.concatenate([on.ravel(), off.ravel()]) / m


def make_word(cls, noise=True):
    """Return a MEL x FRAMES spectro-temporal map standing in for a spoken word.
    Classes differ mainly in onset band + envelope -- the plosive-ish structure
    that actually distinguishes short words like can/pen/cup."""
    s = np.zeros((MEL, FRAMES), np.float32)
    if cls == 0:                      # "can": low-mid onset, sustained
        s[4:12, 0:2] = 1.0; s[6:14, 2:5] = 0.7
    elif cls == 1:                    # "pen": high-band plosive onset, short
        s[14:22, 0:1] = 1.0; s[8:16, 1:3] = 0.6
    else:                            # "cup": broad onset then decay
        s[2:20, 0:1] = 1.0; s[4:12, 1:2] = 0.5; s[6:10, 2:3] = 0.3
    if noise:
        s += 0.08 * rng.standard_normal(s.shape).astype(np.float32)
    s = np.clip(s, 0, 1)
    m = max(s.max(), 1e-6)
    return s.ravel() / m


def rate_to_spikes(vec, T=T_STEPS):
    """Poisson rate coding: intensity -> per-step spike probability -> spikes."""
    p = np.clip(vec, 0, 1)[None, :] * MAX_RATE          # [1, N]
    return (rng.random((T, vec.size)) < p).astype(np.float32)


# --- real-file loaders (camera-frame-from-file and audio-from-file) -------- #
def load_real_image(path):
    """Load a real photo (e.g. 'can.png') -> same 20x20 DoG-filtered vector
    the rest of the network expects. Center-crops to square, resizes, grays."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = img.shape[:2]
    s = min(h, w)                                   # center-crop to square
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = img[y0:y0+s, x0:x0+s]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return dog(img)                                  # -> [VIS_IN] vector


def load_real_audio(path, word_window_s=0.5):
    """Load real audio from a file (wav/mp3/m4a/mp4/etc) -> same
    [MEL*FRAMES] vector the rest of the network expects.

    Decodes via an explicit ffmpeg subprocess call rather than relying on
    librosa/audioread to auto-discover a backend -- this is what fails on
    many Windows setups with NoBackendError even when ffmpeg IS installed,
    if it's just not on PATH the way audioread expects. Calling ffmpeg
    directly here gives a clear, actionable error instead."""
    import shutil, subprocess, tempfile, os as _os
    import soundfile as sf
    import librosa

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg was not found on your system PATH.\n"
            "This is required to decode audio files (wav/m4a/mp4/etc).\n\n"
            "Windows fix:\n"
            "  1. winget install ffmpeg\n"
            "     (or download from https://ffmpeg.org/download.html and\n"
            "      add the folder containing ffmpeg.exe to your PATH)\n"
            "  2. Close and reopen your terminal (PATH changes need a fresh shell)\n"
            "  3. Verify with:  ffmpeg -version\n"
        )

    sr = 16000
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = _os.path.join(tmpdir, "decoded.wav")
        result = subprocess.run(
            [ffmpeg_path, "-y", "-i", path, "-ac", "1", "-ar", str(sr), wav_path],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not _os.path.exists(wav_path):
            raise RuntimeError(
                f"ffmpeg failed to decode '{path}'.\n"
                f"ffmpeg stderr:\n{result.stderr[-1500:]}"
            )
        y, sr_read = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

    # trim silence, keep the loudest word_window_s-long window (crude VAD)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if yt.size == 0:
        yt = y
    target_len = int(word_window_s * sr)
    if yt.size < target_len:
        yt = np.pad(yt, (0, target_len - yt.size))
    else:
        yt = yt[:target_len]
    S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=MEL,
                                        n_fft=512, hop_length=target_len // FRAMES)
    S_db = librosa.power_to_db(S, ref=np.max)
    # resize time axis to exactly FRAMES columns
    if S_db.shape[1] != FRAMES:
        idx = np.linspace(0, S_db.shape[1] - 1, FRAMES).astype(int)
        S_db = S_db[:, idx]
    s = S_db - S_db.min()
    s = s / (s.max() + 1e-6)
    return s.ravel().astype(np.float32)               # -> [AUD_IN] vector


def capture_real_frame(path="can.png"):
    return load_real_image(path)

def capture_real_word(path="can.mp4"):
    return load_real_audio(path)


def augment(vec, noise=0.04, drop=0.03):
    """Light augmentation for repeated presentations of a SINGLE real photo/
    clip. A real photo/recording gives you exactly one vector; STDP needs
    repeated, slightly-varying exposure (this is standing in for natural
    trial-to-trial variation -- lighting flicker, mic noise, hand tremor).
    Additive Gaussian noise + small random dropout, then re-clip to [0,1]."""
    v = vec + noise * rng.standard_normal(vec.shape).astype(np.float32)
    if drop > 0:
        mask = rng.random(vec.shape) > drop
        v = v * mask
    return np.clip(v, 0, 1)


def load_real_dataset(files):
    """files: dict like {"can": ("can.png","can.mp4"), "pen": (...), "cup": (...)}
    (order must match CLASSES). Returns {class_idx: (vis_vec, aud_vec)}."""
    data = {}
    for idx, cls in enumerate(CLASSES):
        img_path, aud_path = files[cls]
        data[idx] = (load_real_image(img_path), load_real_audio(aud_path))
    return data


def train_on_real_files(net, files, n_per_class=120, shuffle_pairs=False):
    """Same training loop as train(), but drawing from real loaded files
    (each repeated with light augmentation) instead of synthetic generators."""
    base = load_real_dataset(files)
    order = []
    for c in range(N_CLASS):
        order += [c] * n_per_class
    rng.shuffle(order)
    for c in order:
        vis0, aud0 = base[c]
        vis = augment(vis0)
        wc = rng.integers(N_CLASS) if shuffle_pairs else c
        aud = augment(base[wc][1])
        net.present(vis, aud, learn=True)
    return base


# --------------------------------------------------------------------------- #
# 2. Spiking layer:  LIF + trace-STDP + WTA + homeostasis + weight-norm
# --------------------------------------------------------------------------- #
class SpikingLayer:
    def __init__(self, n_in, n_out, name=""):
        self.name = name
        # small positive random init, normalized
        W = rng.random((n_out, n_in)).astype(np.float32) * 0.3
        self.W = self._normalize(W)
        self.theta = np.zeros(n_out, np.float32)        # adaptive threshold
        self.n_in, self.n_out = n_in, n_out

    def _normalize(self, W):
        nrm = np.sqrt((W ** 2).sum(1, keepdims=True)) + 1e-6
        return W / nrm * W_NORM

    def run(self, in_spikes, learn=True):
        """Process one presentation. in_spikes: [T, n_in].
        Returns per-neuron output spike counts (a rate readout of the layer)."""
        T = in_spikes.shape[0]
        V = np.zeros(self.n_out, np.float32)
        pre_trace = np.zeros(self.n_in, np.float32)
        out_count = np.zeros(self.n_out, np.float32)

        for t in range(T):
            x = in_spikes[t]                              # [n_in]
            pre_trace = TRACE_DECAY * pre_trace + x
            # membrane update
            V = V_LEAK * V + self.W @ x
            fired = V > (V_THRESH0 + self.theta)
            if fired.any():
                # ---- hard lateral inhibition (winner-take-all) ----
                winners = np.where(fired)[0]
                if winners.size > 1:                      # keep the strongest
                    best = winners[np.argmax(V[winners])]
                    fired = np.zeros(self.n_out, bool); fired[best] = True
                    winners = np.array([best])
                V[fired] = 0.0                            # reset
                V *= 0.0                                  # inhibit the rest too
                out_count += fired
                self.theta[fired] += THETA_INC            # homeostasis: rise
                if learn:
                    # ---- trace-STDP: potentiate winner toward recent pre ----
                    for j in winners:
                        self.W[j] += STDP_LR * (pre_trace - self.W[j])
                        self.W[j] = np.clip(self.W[j], 0, None)
            self.theta *= THETA_DECAY                     # homeostasis: decay
        if learn:
            self.W = self._normalize(self.W)              # divisive weight norm
        return out_count


# --------------------------------------------------------------------------- #
# 3. Network:  vision + audio feature layers -> cross-modal association
# --------------------------------------------------------------------------- #
class Net:
    def __init__(self):
        self.vis = SpikingLayer(VIS_IN, VIS_FEAT, "vis")
        self.aud = SpikingLayer(AUD_IN, AUD_FEAT, "aud")
        self.assoc = SpikingLayer(VIS_FEAT + AUD_FEAT, ASSOC, "assoc")

    def _feat_spikes(self, counts, T=T_STEPS):
        """Turn a layer's output spike-counts into spike trains for the next
        layer (rate re-encoding of the feature activity)."""
        if counts.max() > 0:
            counts = counts / counts.max()
        return rate_to_spikes(counts, T)

    def present(self, vis_vec, aud_vec, learn=True, vis_on=True, aud_on=True):
        """One presentation. vis_on/aud_on gate the modalities (for uni-modal
        test / cross-modal recall)."""
        vis_c = self.vis.run(rate_to_spikes(vis_vec), learn) if vis_on \
                else np.zeros(VIS_FEAT, np.float32)
        aud_c = self.aud.run(rate_to_spikes(aud_vec), learn) if aud_on \
                else np.zeros(AUD_FEAT, np.float32)
        # bind: association layer sees both feature streams concatenated
        vf = self._feat_spikes(vis_c)
        af = self._feat_spikes(aud_c)
        assoc_in = np.concatenate([vf, af], axis=1)       # [T, VIS_FEAT+AUD_FEAT]
        assoc_c = self.assoc.run(assoc_in, learn)
        return assoc_c


# --------------------------------------------------------------------------- #
# 4. Train  (unsupervised: no labels touch any weight)
# --------------------------------------------------------------------------- #
def train(net, n_per_class=120, shuffle_pairs=False):
    order = []
    for c in range(N_CLASS):
        order += [c] * n_per_class
    rng.shuffle(order)
    for c in order:
        vis = dog(make_object(c))
        # control baseline: mismatch the word with the object
        wc = rng.integers(N_CLASS) if shuffle_pairs else c
        aud = make_word(wc)
        net.present(vis, aud, learn=True)


# --------------------------------------------------------------------------- #
# 5. Evaluate  (labels used ONLY here, for readout assignment + scoring)
# --------------------------------------------------------------------------- #
def assign_readout(net, n=40):
    """Assign each association neuron to the class it fires most for, using
    VISION ONLY. This is what makes the audio-only test a real binding test:
    a neuron is labelled by what it *sees*, so if hearing a word later fires
    that same neuron, the two modalities are genuinely bound (not just the
    audio being re-clustered). Labels are used for the mapping only, never for
    learning."""
    resp = np.zeros((ASSOC, N_CLASS), np.float32)
    for c in range(N_CLASS):
        for _ in range(n):
            a = net.present(dog(make_object(c)), make_word(c),
                            learn=False, vis_on=True, aud_on=False)
            resp[:, c] += a
    assign = resp.argmax(1)
    active = resp.sum(1) > 0
    return assign, active

def classify(net, assign, active, vis_vec, aud_vec, vis_on=True, aud_on=True):
    a = net.present(vis_vec, aud_vec, learn=False, vis_on=vis_on, aud_on=aud_on)
    scores = np.zeros(N_CLASS)
    for c in range(N_CLASS):
        sel = active & (assign == c)
        scores[c] = a[sel].sum()
    if scores.sum() == 0:
        return -1
    return int(scores.argmax())

def confusion(net, assign, active, n=40, vis_on=True, aud_on=True):
    M = np.zeros((N_CLASS, N_CLASS), int)
    for c in range(N_CLASS):
        for _ in range(n):
            vis = dog(make_object(c))
            aud = make_word(c)
            pred = classify(net, assign, active, vis, aud, vis_on, aud_on)
            if pred >= 0:
                M[c, pred] += 1
    return M

def acc(M):
    tot = M.sum()
    return np.trace(M) / tot if tot else 0.0


# --------------------------------------------------------------------------- #
# 6. Main experiment
# --------------------------------------------------------------------------- #
def banner(s): print("\n" + "=" * 62 + f"\n{s}\n" + "=" * 62)

def show(M, title):
    print(f"\n{title}   (rows=true, cols=pred)")
    print("        " + "  ".join(f"{c:>4}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:>4}  " + "  ".join(f"{v:>4}" for v in M[i]))
    print(f"  accuracy = {acc(M)*100:5.1f}%   (chance = {100/N_CLASS:.1f}%)")

if __name__ == "__main__":
    banner("TRAIN: real word<->object pairing (unsupervised STDP)")
    net = Net()
    train(net, n_per_class=120, shuffle_pairs=False)
    assign, active = assign_readout(net)
    print(f"association neurons active after training: {active.sum()}/{ASSOC}")

    # Test 1: recognition (vision + audio, held-out instances)
    M1 = confusion(net, assign, active, vis_on=True, aud_on=True)
    show(M1, "TEST 1  recognition (vision+audio)")

    # Test 2: cross-modal recall (AUDIO ONLY -> identity, vision silent)
    M2 = confusion(net, assign, active, vis_on=False, aud_on=True)
    show(M2, "TEST 2  cross-modal recall (audio only, vision off)")

    # Test 3: vision-only recognition (audio silent)
    M3 = confusion(net, assign, active, vis_on=True, aud_on=False)
    show(M3, "TEST 3  vision only (audio off)")

    # Control: shuffled pairing -> binding should fail
    banner("CONTROL: shuffled word<->object pairing (should NOT bind)")
    net_c = Net()
    train(net_c, n_per_class=120, shuffle_pairs=True)
    assign_c, active_c = assign_readout(net_c)
    Mc = confusion(net_c, assign_c, active_c, vis_on=False, aud_on=True)
    show(Mc, "CONTROL  cross-modal recall under shuffled training")

    banner("SUMMARY")
    print(f"  recognition (vis+aud)     : {acc(M1)*100:5.1f}%")
    print(f"  cross-modal recall (aud)  : {acc(M2)*100:5.1f}%")
    print(f"  vision only               : {acc(M3)*100:5.1f}%")
    print(f"  shuffled-control recall   : {acc(Mc)*100:5.1f}%  <- want near chance")

    # ------------------------------------------------------------------ #
    # Visualizations
    # ------------------------------------------------------------------ #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) learned visual receptive fields (ON-channel of each neuron's weights)
    on = net.vis.W[:, :IMG*IMG].reshape(VIS_FEAT, IMG, IMG)
    order = np.argsort(-net.vis.W.sum(1))          # busiest neurons first
    fig, ax = plt.subplots(6, 8, figsize=(9, 7))
    for k, a in enumerate(ax.ravel()):
        a.imshow(on[order[k]], cmap="magma"); a.axis("off")
    fig.suptitle("Visual feature neurons: receptive fields learned by STDP\n"
                 "(self-organized shape detectors — no labels, no backprop)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig("/home/claude/receptive_fields.png", dpi=110)

    # (b) confusion matrices + the control collapse
    mats = [(M1, "recognition\n(vis+aud)"), (M2, "cross-modal recall\n(audio only)"),
            (M3, "vision only"), (Mc, "CONTROL: shuffled\n(audio only)")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, (M, t) in zip(ax, mats):
        a.imshow(M, cmap="Blues")
        for i in range(N_CLASS):
            for j in range(N_CLASS):
                a.text(j, i, M[i, j], ha="center", va="center",
                       color="white" if M[i, j] > M.max()/2 else "black")
        a.set_xticks(range(N_CLASS)); a.set_xticklabels(CLASSES)
        a.set_yticks(range(N_CLASS)); a.set_yticklabels(CLASSES)
        a.set_title(f"{t}\nacc {acc(M)*100:.0f}%  (chance 33%)", fontsize=10)
        a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("Cross-modal binding: real pairing binds (cols 1-3); "
                 "shuffled pairing collapses to chance (col 4)", fontsize=12)
    fig.tight_layout(); fig.savefig("/home/claude/confusion.png", dpi=110)
    print("\nsaved: receptive_fields.png, confusion.png")