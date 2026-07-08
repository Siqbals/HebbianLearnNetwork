# Hebbian Cross-Modal Binding — a no-backprop learning prototype

A spiking neural network that learns to associate a spoken word with a visual
object — entirely through **local, biologically-inspired plasticity (STDP)**.
No backpropagation, no labels used for learning, no loss function. The network
only ever sees "this picture and this sound happened at the same moment,"
repeatedly, and has to figure out the rest — the same way a human learns "cup"
means *that*, long before they understand what a label is.

This project started from a simple question: **the [DishBrain experiment](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6)
trained 800,000 real neurons to play Pong with no training data and no
backprop — how?** The answer (local Hebbian/STDP plasticity + a structured
feedback loop) turned into this: a from-scratch attempt to reproduce the same
mechanism in software, and see how far it can go.

---

## What this actually demonstrates

- **Local learning works.** Cross-modal word↔object binding emerges purely
  from repeated co-occurrence and a spike-timing-based update rule — no
  global error signal anywhere in the system.
- **A built-in falsifiability check.** Every experiment here is run against a
  shuffled-pairing control (mismatch the word and the object during training).
  A real binding mechanism should fail under shuffling; ours does — recall
  collapses to chance the moment the correlation is destroyed. This is what
  makes the "it learned something real" claim actually testable rather than
  wishful.
- **Real limits, found empirically, not assumed.** This repo also documents
  where the mechanism breaks — position invariance, detector fragmentation
  under naive competitive learning, and how much a network's accuracy depends
  on the *separability* of the raw input, not just its size. See
  [Known Limitations](#known-limitations--things-we-learned-the-hard-way).

---

## How it works

```
 photo (jpeg) ──▶ [DoG retina encode] ──▶ [Visual feature layer]  ──┐
                    (fixed transform)      (STDP + WTA,             │
                                             self-organizing)        ▼
                                                              [Cross-modal
                                                               association layer] ─▶ "can" / "cup" / ...
                                                               (STDP + WTA)
 audio (m4a) ────▶ [mel spectrogram]  ──▶ [Auditory feature layer] ─┘
                    (fixed transform)      (STDP + WTA,
                                             self-organizing)
```

**Substrate:** Leaky Integrate-and-Fire (LIF) spiking neurons — membrane
potential integrates input spikes, leaks, fires + resets at threshold.

**Learning rule:** trace-based **STDP** (spike-timing-dependent plasticity).
Each synapse only ever "sees" its own two neurons' spike timing — if a
presynaptic spike reliably precedes a postsynaptic spike, the synapse
strengthens; otherwise it doesn't. Purely local. No backward pass, no global
loss, no gradient of any kind.

**Stability mechanisms** (without these, plain STDP glues everything to
everything — see [Known Limitations](#known-limitations--things-we-learned-the-hard-way)):
- **Winner-take-all lateral inhibition** — forces neurons to specialize instead
  of all learning the same generic pattern.
- **Adaptive threshold homeostasis** — a neuron that fires too often becomes
  temporarily harder to trigger, keeping usage of the neuron population even.
- **Divisive weight normalization** — keeps synaptic weights in a bounded,
  comparable range.

**Cross-modal binding:** a third layer receives converging input from both
feature layers. Because the "can" photo and the "can" sound arrive at the same
moment during training, whichever association neuron responds to both gets its
synapses to *both* modalities strengthened together — this is the entire
mechanism behind "the sound and the sight become the same concept." No label,
no supervision signal, ever touches a synaptic weight.

---

## Repo contents

| File | What it does |
|---|---|
| `hebbian_binding.py` | Core library — LIF neurons, STDP, the `Net` class, synthetic data generators, and real-file loaders (`load_real_image`, `load_real_audio`). Import from this, or run it directly for a self-contained synthetic-data demo. |
| `train_on_real_files.py` | Trains and evaluates on your own real photos + recordings. Prints confusion matrices for recognition, cross-modal recall, vision-only, and a shuffled-pairing control. |
| `test_single_input.py` | Live, single-example sanity checks — "show it a cup right now, what does it actually say" — rather than only aggregate accuracy. |
| `build_network_viz.py` | Trains fresh on your real files and outputs an interactive 3D visualization (`network_3d_map.html`) of the learned neurons and their strongest connections — colored by which class each neuron ended up representing. |
| `gen_network_viz.py` | Same visualization, but from the synthetic generators (for the original 3-class demo). |

---

## Quickstart

### 1. Install dependencies
```bash
pip install numpy scipy matplotlib opencv-python librosa soundfile
```
You'll also need **ffmpeg** installed and on your system PATH (used to decode
audio files). Verify with `ffmpeg -version`.

### 2. Try the synthetic demo first (no files needed)
```bash
python hebbian_binding.py
```
Trains on built-in synthetic shape/sound generators, runs the full evaluation
suite, and saves `receptive_fields.png` + `confusion.png`.

### 3. Train on your own objects
Put six files in the same folder — a photo and a short (~0.5s) audio clip of
you saying the word, for each class:
```
can.jpeg   can.m4a
cup.jpeg   cup.m4a
controller.jpeg   controller.m4a
```
(Class names are set in `CLASSES` at the top of `hebbian_binding.py` — edit
this list and the `FILES` dict in `train_on_real_files.py` to match your own
objects/filenames.)

```bash
python train_on_real_files.py
```

### 4. Visualize what it learned
```bash
python build_network_viz.py
```
Then open `network_3d_map.html` directly in a browser (double-click it — no
server needed). Drag to rotate, scroll to zoom, hover a neuron for its ID.

---

## Known limitations — things we learned the hard way

This project treats negative and partial results as first-class findings, not
bugs to hide. A few load-bearing ones:

- **No position invariance (yet).** The network only recognizes an object in
  roughly the same framing it was trained on — shifting a trained object ~6px
  off-center dropped accuracy from 98% to 55% in testing. Plain Hebbian
  learning doesn't share knowledge across spatial positions the way a CNN's
  weight-sharing does; fixing this needs a temporal *trace* rule (see
  Roadmap) rather than more training on jittered stills — we confirmed that
  training on 100 random-jitter presentations doesn't consolidate into one
  position-general detector, it **fragments** across ~35 different neurons,
  each too weak to fire reliably (this is a known competitive-learning
  failure mode, not a hyperparameter issue).
- **Neuron count is not the bottleneck you'd expect.** Scaling from
  200/100/60 neurons (vision/audio/association) up to 500/500/500 made
  accuracy *worse*, not better, and increasing training presentations
  proportionally didn't fix it either. The actual bottleneck was upstream:
  **input separability.** If two classes' audio encodings are highly
  correlated (we measured 71.6% correlation between two crude synthetic test
  sounds), no amount of network capacity or training time can teach the
  network a distinction that isn't present in the data. Check separability
  first, before touching architecture.
- **Modalities aren't equally reliable.** In every real-file run so far,
  vision-only recognition has been strong (~85-98%) while audio-only
  cross-modal recall lagged well behind (~40-50%). Don't assume a weak overall
  number means the whole mechanism is broken — check per-modality performance
  before diagnosing.
- **The control condition matters more than the headline accuracy.** Every
  result in this repo is reported alongside a shuffled-pairing control run.
  A high accuracy number alone doesn't prove binding occurred; the same
  number under shuffled training would.

---

## Roadmap (v2 ideas, not yet implemented)

- **Position invariance via a temporal trace rule** (Földiák, 1991) — bind
  representations across positions using *motion* (an object swept smoothly
  across the frame) instead of independent static jittered stills, which we
  showed doesn't consolidate.
- **Weight-shared local learning** (à la [SoftHebb](https://arxiv.org/abs/2209.11883))
  as an alternative/complementary route to position invariance, trading some
  biological purity for CNN-like generalization while keeping the update rule
  local.
- **Bounding-box localization**, which falls out for free once spatial
  structure is preserved through the network rather than flattened at the
  input layer.

---

## References / prior work this builds on

- Kagan et al., 2022, *In vitro neurons learn and exhibit sentience when
  embodied in a simulated game-world* (Neuron) — the DishBrain experiment.
- Diehl & Cook, 2015, *Unsupervised learning of digit recognition using
  spike-timing-dependent plasticity* — the STDP + WTA + homeostasis recipe
  this network's competitive layers are based on.
- Földiák, 1991, *Learning invariance from transformation sequences* — the
  trace-rule approach to position invariance referenced in the Roadmap.
- Journé et al., 2022, *Hebbian Deep Learning Without Feedback* (SoftHebb).
- Vong et al., 2024, *Grounded language acquisition through the eyes and ears
  of a single child* (Science) — real-world validation that word-referent
  binding from raw co-occurrence (no explicit labels) produces genuine
  linguistic competence, using headcam data from an actual child.

---

## Honest status

This is a research prototype, not a production recognition system. Every
number in this README (or produced by these scripts) should be read alongside
its shuffled-control counterpart, and treated as evidence about a specific,
small-scale mechanism — not as a claim that this approach currently rivals
backprop-trained models at real-world object or speech recognition. It
doesn't, yet, and closing that gap (if it's even fully closable — see the
discussion of Hebbian learning's fundamental information disadvantage vs.
gradient-based credit assignment) is an open research question, not a solved
one.
