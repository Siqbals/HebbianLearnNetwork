"""
Train + evaluate the Hebbian cross-modal network on real files.

Expects these six files in the SAME folder as this script (and as
hebbian_binding.py, which this imports from):

    can.jpeg   can.m4a
    cup.jpeg   cup.m4a
    controller.jpeg   controller.m4a
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hebbian_binding import *   # CLASSES must be ["can","cup","controller"] in that file

FILES = {
    "can":        ("can.jpeg", "can.m4a"),
    "cup":        ("cup.jpeg", "cup.m4a"),
    "controller": ("controller.jpeg", "controller.m4a"),
}

banner = lambda s: print("\n" + "="*62 + f"\n{s}\n" + "="*62)

def show(M, title):
    print(f"\n{title}   (rows=true, cols=pred)")
    print("        " + "  ".join(f"{c:>10}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:>10}  " + "  ".join(f"{v:>10}" for v in M[i]))
    print(f"  accuracy = {acc(M)*100:5.1f}%   (chance = {100/N_CLASS:.1f}%)")

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

def classify_real(net, assign, active, vis_vec, aud_vec, vis_on=True, aud_on=True):
    a = net.present(vis_vec, aud_vec, learn=False, vis_on=vis_on, aud_on=aud_on)
    scores = np.zeros(N_CLASS)
    for c in range(N_CLASS):
        sel = active & (assign == c)
        scores[c] = a[sel].sum()
    return int(scores.argmax()) if scores.sum() > 0 else -1

def confusion_real(net, assign, active, base, n=40, vis_on=True, aud_on=True):
    M = np.zeros((N_CLASS, N_CLASS), int)
    for c in range(N_CLASS):
        for _ in range(n):
            vis = augment(base[c][0])
            aud = augment(base[c][1])
            pred = classify_real(net, assign, active, vis, aud, vis_on, aud_on)
            if pred >= 0:
                M[c, pred] += 1
    return M


if __name__ == "__main__":
    banner(f"TRAIN on real files: {', '.join(f'{k}.jpeg/{k}.m4a' for k in CLASSES)}")
    net = Net()
    base = train_on_real_files(net, FILES, n_per_class=120, shuffle_pairs=False)

    assign, active = assign_readout_real(net, base)
    print(f"association neurons active: {active.sum()}/{ASSOC}")

    M1 = confusion_real(net, assign, active, base, vis_on=True, aud_on=True)
    show(M1, "TEST 1  recognition (real vision + real audio)")

    M2 = confusion_real(net, assign, active, base, vis_on=False, aud_on=True)
    show(M2, "TEST 2  cross-modal recall (real audio only)")

    M3 = confusion_real(net, assign, active, base, vis_on=True, aud_on=False)
    show(M3, "TEST 3  vision only (real photo only)")

    banner("CONTROL: shuffled real word<->object pairing")
    net_c = Net()
    base_c = train_on_real_files(net_c, FILES, n_per_class=120, shuffle_pairs=True)
    assign_c, active_c = assign_readout_real(net_c, base_c)
    Mc = confusion_real(net_c, assign_c, active_c, base_c, vis_on=False, aud_on=True)
    show(Mc, "CONTROL  cross-modal recall under shuffled real pairing")

    banner("SUMMARY (real files)")
    print(f"  recognition (vis+aud)     : {acc(M1)*100:5.1f}%")
    print(f"  cross-modal recall (aud)  : {acc(M2)*100:5.1f}%")
    print(f"  vision only               : {acc(M3)*100:5.1f}%")
    print(f"  shuffled-control recall   : {acc(Mc)*100:5.1f}%  <- want near chance")