# Plastic-vs-refs overlay: one random exp5 specimen against exp4 (Dec) and
# exp3 (Nov) empty-holder median signals. Same SG smoothing (w=7, p=2) and
# 0.1 s repo windowing as R1.4; per-frequency median of window |mean|.
import glob
import importlib.util
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
OUT = os.path.join(REPO, "results", "r14_snr_noise_dr")
FIGD = os.path.join(OUT, "figures")

_spec = importlib.util.spec_from_file_location(
    "train_v2", os.path.join(REPO, "src", "scripts", "train_v2.py"))
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

DP = (100 / 12) * 0.1
CHS = {"HG": "HG (mV) mean", "LG": "LG (mV) mean"}


def prep(files):
    frames = []
    for f in sorted(files):
        df = pd.read_csv(f, delimiter=";")
        df["SourceFile"] = os.path.basename(f)
        df["Day"] = 1
        frames.append(df)
    return tv.grouped_window_averages(
        tv.apply_pre_sg(pd.concat(frames, ignore_index=True), 7, 2), DP)


def medsig(t):
    out = {}
    for ch, mc in CHS.items():
        o = (t.groupby("Frequency (GHz)")[mc]
             .apply(lambda m: float(np.median(
                 np.abs(m.to_numpy(float)[np.isfinite(m.to_numpy(float))])))))
        out[ch] = o.sort_index()
    return out


e5 = sorted(glob.glob(os.path.join(
    REPO, "data", "experiment_5_plastics", "processed", "*.csv")))
assert len(e5) == 60
random.seed(42)
pick = random.choice(e5)
print(f"random pick (seed 42): {os.path.basename(pick)}", flush=True)
plastic = os.path.basename(pick).split("_")[0]
pls = medsig(prep([pick]))

refs4 = [os.path.join(REPO, "data", "experiment_4_plastics", "processed",
                       f"REF_{t}.csv") for t in (1, 8, 15)] + [
    os.path.join(REPO, "data", "experiment_4_plastics", "processed",
                 "new_sample", f"REF_{t}.csv") for t in (16, 23)]
ref4 = medsig(prep(refs4))
# cross-check vs shipped curves file
C = pd.read_csv(os.path.join(FIGD, "background_spectrum_curves.csv"), sep=";")
for ch in ("HG", "LG"):
    d = abs(ref4[ch].to_numpy(float)
            - C.set_index("freq")[ch + "_sig"].to_numpy(float))
    print(f"exp4 {ch} cross-check max abs diff: {d.max():.2e}", flush=True)
    assert d.max() < 1e-9

refs3 = [os.path.join(REPO, "data", "experiment_3_repeatibility", "processed",
                       f"REF_{t}.csv") for t in range(10, 16)]
ref3 = medsig(prep(refs3))

fr = pls["HG"].index.to_numpy(float)
plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})
BLUE, RED = "#0072B2", "#D55E00"
fig, (axh, axl) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for ax in (axh, axl):
    ax.axvspan(100, 210, color="0.9", zorder=0)
    ax.axvspan(220, 330, color="0.85", zorder=0)
    ax.axvspan(350, 590, color="0.9", zorder=0)
    ax.set_xlim(100, 590)
    ax.set_ylim(5e-2, 5e3)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
axh.text(155, 3e3, "HG dead zone", ha="center", va="top", fontsize=12,
         color="0.35")
axh.text(275, 3e3, "transition", ha="center", va="top", fontsize=12,
         color="0.35")
axh.text(470, 3e3, "LG rolloff", ha="center", va="top", fontsize=12,
         color="0.35")
axh.plot(fr, ref4["HG"].to_numpy(float), color=RED, lw=2.2,
         label="REF exp4 (Dec)", zorder=3)
axh.plot(ref3["HG"].index.to_numpy(float), ref3["HG"].to_numpy(float),
         color=RED, lw=1.2, ls=":", alpha=0.8, label="REF exp3 (Nov)",
         zorder=3)
axh.plot(fr, pls["HG"].to_numpy(float), color="black", lw=2.2,
         label=f"plastic {plastic} (exp5)", zorder=4)
axh.set_ylabel("HG (mV)")
axl.plot(fr, ref4["LG"].to_numpy(float), color=BLUE, lw=2.2,
         label="REF exp4 (Dec)", zorder=3)
axl.plot(ref3["LG"].index.to_numpy(float), ref3["LG"].to_numpy(float),
         color=BLUE, lw=1.2, ls=":", alpha=0.8, label="REF exp3 (Nov)",
         zorder=3)
axl.plot(fr, pls["LG"].to_numpy(float), color="black", lw=2.2,
         label=f"plastic {plastic} (exp5)", zorder=4)
axl.set_ylabel("LG (mV)")
axl.set_xlabel("Frequency (GHz)")


def place_legend(ax):
    lines = ax.get_lines()
    cands = ["upper right", "upper left", "lower right", "lower left",
             "center right", "center left", "center",
             ("center", (0.30, 0.72)), ("center", (0.70, 0.30)),
             ("center", (0.32, 0.30)), ("center", (0.72, 0.72))]
    best, best_n, bestleg = cands[0], None, ("upper right", None)
    for cand in cands:
        if isinstance(cand, tuple):
            loc, anchor = cand
            leg = ax.legend(loc=loc, bbox_to_anchor=anchor, framealpha=0.95)
            tag = f"{loc}@{anchor}"
        else:
            loc, anchor = cand, None
            leg = ax.legend(loc=loc, framealpha=0.95)
            tag = cand
        fig.canvas.draw()
        bb = leg.get_window_extent(renderer).padded(4)
        n = 0
        for ln in lines:
            x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
            dp = ax.transData.transform(np.column_stack([x, y]))
            n += int((((dp[:, 0] >= bb.x0) & (dp[:, 0] <= bb.x1)
                       & (dp[:, 1] >= bb.y0) & (dp[:, 1] <= bb.y1))).sum())
        print(f"legend {ax.get_ylabel()} {tag}: {n} points inside",
              flush=True)
        if best_n is None or n < best_n:
            best, best_n, bestleg = tag, n, (loc, anchor)
        leg.remove()
        if n == 0:
            break
    loc, anchor = bestleg
    if anchor is None:
        ax.legend(loc=loc, framealpha=0.95)
    else:
        ax.legend(loc=loc, bbox_to_anchor=anchor, framealpha=0.95)
    print(f"legend {ax.get_ylabel()}: {best} ({best_n})", flush=True)
    assert best_n == 0


fig.canvas.draw()
renderer = fig.canvas.get_renderer()
place_legend(axh)
place_legend(axl)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "plastic_vs_refs.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGD, "plastic_vs_refs.png"), dpi=300,
            bbox_inches="tight")
print("figure written", flush=True)
