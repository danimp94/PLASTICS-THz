# S3 background spectrum figure (R3.7): empty-holder system response, exp4 Dec 2024.
# Same smoothing/windowing as R1.4 (apply_pre_sg w=7 p=2, 0.1 s windowing).
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
OUT = os.path.join(REPO, "results", "r14_snr_noise_dr")
FIGD = os.path.join(OUT, "figures")
os.makedirs(FIGD, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "train_v2", os.path.join(REPO, "src", "scripts", "train_v2.py"))
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

# ---- 0. verify inputs ----
refs = [os.path.join(REPO, "data", "experiment_4_plastics", "processed",
                      f"REF_{t}.csv") for t in (1, 8, 15)] + [
    os.path.join(REPO, "data", "experiment_4_plastics", "processed",
                 "new_sample", f"REF_{t}.csv") for t in (16, 23)]
assert all(os.path.exists(f) for f in refs), "missing ref"
frames = []
for f in sorted(refs):
    df = pd.read_csv(f, delimiter=";")
    assert list(df.columns[:4]) == ["Sample", "Frequency (GHz)",
                                    "LG (mV)", "HG (mV)"], f
    df["SourceFile"] = os.path.basename(f)
    df["Day"] = 1
    frames.append(df)
e4 = pd.concat(frames, ignore_index=True)
freqs = sorted(e4["Frequency (GHz)"].unique().tolist())
assert freqs == [float(x) for x in range(100, 591, 10)]
print("inputs OK: 5 refs, schema match, 50 freqs", flush=True)

# ---- 1. smooth + window, per-freq medians ----
t4 = tv.grouped_window_averages(tv.apply_pre_sg(e4, 7, 2), (100 / 12) * 0.1)
CHS = {"HG": ("HG (mV) mean", "HG (mV) std deviation"),
       "LG": ("LG (mV) mean", "LG (mV) std deviation")}
fr = np.array(freqs)
dat = {"freq": fr}
for ch, (mc, sc) in CHS.items():
    sig, flo = [], []
    for f, g in t4.groupby("Frequency (GHz)"):
        m = g[mc].to_numpy(float)
        s = g[sc].to_numpy(float)
        m = m[np.isfinite(m)]
        s = s[np.isfinite(s) & (s > 0)]
        sig.append(float(np.median(np.abs(m))))
        flo.append(float(np.median(s)))
    dat[ch + "_sig"], dat[ch + "_floor"] = np.array(sig), np.array(flo)
C = pd.DataFrame(dat)

# floor cross-check vs shipped TABLE B
B = pd.read_csv(os.path.join(OUT, "sys_noise_floor_per_freq.csv"), sep=";")
for ch in ("HG", "LG"):
    m = B[B.channel == ch].set_index("freq")["noise_floor_mV"]
    cc = C.set_index("freq")[ch + "_floor"]
    d = (cc - m).abs() / m
    print(f"{ch} floor cross-check: max rel diff = {d.max():.2e}", flush=True)
    assert d.max() < 1e-9
C.to_csv(os.path.join(FIGD, "background_spectrum_curves.csv"),
         index=False, sep=";")

# ---- optional exp3 overlay: numeric stability verdict first ----
e3f = [os.path.join(REPO, "data", "experiment_3_repeatibility", "processed",
                    f"REF_{t}.csv") for t in range(10, 16)]
assert all(os.path.exists(f) for f in e3f)
fr3 = []
for f in sorted(e3f):
    df = pd.read_csv(f, delimiter=";")
    df["SourceFile"] = os.path.basename(f)
    df["Day"] = 1
    fr3.append(df)
t3 = tv.grouped_window_averages(
    tv.apply_pre_sg(pd.concat(fr3, ignore_index=True), 7, 2), (100 / 12) * 0.1)
ov = {}
for ch, (mc, _) in CHS.items():
    ov[ch] = (t3.groupby("Frequency (GHz)")[mc]
              .apply(lambda m: float(np.median(
                  np.abs(m.to_numpy(float)[np.isfinite(m.to_numpy(float))])))))
ratio = {}
for ch in ("HG", "LG"):
    a = C.set_index("freq")[ch + "_sig"]
    b = ov[ch]
    r = (b / a).replace([np.inf, -np.inf], np.nan).dropna()
    ratio[ch] = float(r.median())
    print(f"exp3/exp4 signal median ratio {ch}: {ratio[ch]:.3f} "
          f"(spread p10-p90: {float(r.quantile(0.1)):.3f}-"
          f"{float(r.quantile(0.9)):.3f})", flush=True)
USE_OVERLAY = False  # single-experiment figure: Dec exp4 only (same era as
# the plastics, same curves behind the S3 tables). Nov exp3 dropped.
print(f"overlay included: {USE_OVERLAY}", flush=True)

# ---- 2. figure ----
KSET = {}
with open(os.path.join(OUT, "subset_lists_canonical.txt")) as fh:
    for line in fh:
        line = line.strip()
        if line.startswith("---"):
            break
        if line.startswith("K") and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in ("K1", "K3", "K5", "K10", "K20"):
                try:
                    KSET[k] = [int(x) for x in v.split(",") if x.strip()]
                except ValueError:
                    pass
assert set(KSET) == {"K1", "K3", "K5", "K10", "K20"}, KSET.keys()

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})
BLUE, RED = "#0072B2", "#D55E00"   # Okabe-Ito, colorblind-safe
LBLUE, LRED = "#56B4E9", "#E69F00"

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axvspan(100, 210, color="0.9", zorder=0)
ax.axvspan(220, 330, color="0.85", zorder=0)
ax.axvspan(350, 590, color="0.9", zorder=0)
top = 3e3
ax.text(155, top, "HG dead zone", ha="center", va="top", fontsize=12,
        color="0.35")
ax.text(275, top, "transition", ha="center", va="top", fontsize=12,
        color="0.35")
ax.text(470, top, "LG rolloff", ha="center", va="top", fontsize=12,
        color="0.35")
ax.plot(fr, C["HG_sig"], color=RED, lw=2.2, label="HG signal", zorder=3)
ax.plot(fr, C["HG_floor"], color=RED, lw=1.4, ls=(0, (3, 2)),
        label="HG floor", zorder=3)
ax.plot(fr, C["LG_sig"], color=BLUE, lw=2.2, label="LG signal", zorder=3)
ax.plot(fr, C["LG_floor"], color=BLUE, lw=1.4, ls=(0, (3, 2)),
        label="LG floor", zorder=3)
if USE_OVERLAY:
    o3 = pd.DataFrame({"freq": ov["HG"].index.to_numpy(float),
                       "HG3": ov["HG"].to_numpy(float),
                       "LG3": ov["LG"].to_numpy(float)}).sort_values("freq")
    ax.plot(o3["freq"], o3["HG3"], color=RED, lw=1.0, ls=":",
            alpha=0.7, label="HG, Nov (exp3)", zorder=3)
    ax.plot(o3["freq"], o3["LG3"], color=BLUE, lw=1.0, ls=":",
            alpha=0.7, label="LG, Nov (exp3)", zorder=3)
ax.set_xlim(100, 590)
ax.set_ylim(5e-2, 5e3)
ax.set_yscale("log")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Amplitude (mV)")
ax.grid(True, which="major", alpha=0.3)

# legend pinned top-right per author request; overlap audited and reported
lines = ax.get_lines()
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
leg = ax.legend(loc="upper right", framealpha=0.95)
fig.canvas.draw()
bb = leg.get_window_extent(renderer).padded(4)
n = 0
for ln in lines:
    x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
    dp = ax.transData.transform(np.column_stack([x, y]))
    inside = ((dp[:, 0] >= bb.x0) & (dp[:, 0] <= bb.x1)
              & (dp[:, 1] >= bb.y0) & (dp[:, 1] <= bb.y1))
    n += int(inside.sum())
print(f"legend loc: upper right (curve points inside box: {n})", flush=True)

fig.tight_layout()
fig.savefig(os.path.join(FIGD, "background_spectrum.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(FIGD, "background_spectrum.png"), dpi=300,
            bbox_inches="tight")
print("figure written", flush=True)
