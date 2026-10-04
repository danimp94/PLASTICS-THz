# R1.4 supplement: self-contained SNR (exp5) + system floor/DR (exp4 refs).
# Reuses repo code only: train_v2.apply_pre_sg (SG smoothing) and
# train_v2.grouped_window_averages (0.1 s windowing). No ML, no pivoting.
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
OUT = os.path.join(REPO, "results", "r14_snr_noise_dr")
os.makedirs(OUT, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "train_v2", os.path.join(REPO, "src", "scripts", "train_v2.py"))
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

SG_W, SG_P = 7, 2          # repo defaults: train_v2 PRE_SG_W / PRE_SG_P
WINDOW_S = 0.1
DP = (100 / 12) * WINDOW_S  # % of per-frequency dwell per 0.1 s window (12 s dwell)
CHS = {"HG": ("HG (mV) mean", "HG (mV) std deviation"),
       "LG": ("LG (mV) mean", "LG (mV) std deviation")}

K10 = [320, 330, 350, 360, 370, 400, 420, 460]
K20 = [230, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
       420, 430, 440, 450, 460, 470, 500, 520]
K3 = [350, 370, 360]  # {350,370} universal >=4/5 + modal 3rd slot 360 (3/5 folds)
SUBSETS = {"K3": K3, "K10": K10, "K20": K20}


def load_arm(files):
    frames = []
    for f in sorted(files):
        df = pd.read_csv(f, delimiter=";")
        df["SourceFile"] = os.path.basename(f)
        df["Day"] = 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def smooth_window(df):
    sm = tv.apply_pre_sg(df, SG_W, SG_P)
    return tv.grouped_window_averages(sm, DP)


def lin2db(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 20 * np.log10(x), np.nan)


# ---- step 0: verification ----
e5_files = sorted(glob.glob(os.path.join(
    REPO, "data", "experiment_5_plastics", "processed", "*.csv")))
ref_files = [os.path.join(REPO, "data", "experiment_4_plastics", "processed",
                          f"REF_{t}.csv") for t in (1, 8, 15)] + [
    os.path.join(REPO, "data", "experiment_4_plastics", "processed",
                 "new_sample", f"REF_{t}.csv") for t in (16, 23)]
assert len(e5_files) == 60, f"exp5 files: {len(e5_files)}"
assert all(os.path.exists(f) for f in ref_files), "missing ref file"
e5 = load_arm(e5_files)
rf = load_arm(ref_files)
assert list(e5.columns[:4]) == ["Sample", "Frequency (GHz)", "LG (mV)", "HG (mV)"]
assert list(rf.columns[:4]) == ["Sample", "Frequency (GHz)", "LG (mV)", "HG (mV)"]
f5 = sorted(e5["Frequency (GHz)"].unique().tolist())
fr = sorted(rf["Frequency (GHz)"].unique().tolist())
assert f5 == fr == [float(x) for x in range(100, 591, 10)], "freq lists differ"
print(f"step0 OK: 60 exp5 files, 5 ref files, schema match, "
      f"50 freqs 100-590 step 10 both arms", flush=True)
n5 = np.median([pd.read_csv(f, delimiter=";",
                         usecols=["Frequency (GHz)"]).groupby("Frequency (GHz)")
                .size().median() for f in e5_files])
n4 = np.median([pd.read_csv(f, delimiter=";",
                         usecols=["Frequency (GHz)"]).groupby("Frequency (GHz)")
                .size().median() for f in ref_files])
w5 = max(1, int(n5 * DP / 100))
w4 = max(1, int(n4 * DP / 100))
print(f"rows/freq median: exp5={n5:.0f} -> ~{w5} samples/0.1s window; "
      f"exp4refs={n4:.0f} -> ~{w4} samples/0.1s window", flush=True)

# ---- smooth + window ----
t5 = smooth_window(e5)
t4 = smooth_window(rf)
print(f"windowed rows: exp5={len(t5)}, exp4refs={len(t4)}", flush=True)

# ---- TABLE A: per-plastic SNR, exp5 only ----
# Two-stage aggregation (documented in METHODS_NOTE.md): per-specimen median
# first, then median/min ACROSS the 5 specimens. A pooled min over raw windows
# is meaningless: any window whose smoothed mean crosses zero gives
# |mean|/std -> -inf dB, so pooled minima collapse to ~-300 dB zero-crossing
# artefacts rather than noise information.
t5["plastic"] = t5["Sample"].str[0]
spec_rows = []
for ch, (mc, sc) in CHS.items():
    d = t5[["plastic", "SourceFile", "Frequency (GHz)", mc, sc]].copy()
    d = d[np.isfinite(d[mc]) & np.isfinite(d[sc]) & (d[sc] > 0)]
    d["snr"] = d[mc].abs() / d[sc]
    d["absmean"] = d[mc].abs()
    g = d.groupby(["plastic", "SourceFile", "Frequency (GHz)"])
    s = g.agg(snr_med=("snr", "median"), sig_med=("absmean", "median"),
              noi_med=(sc, "median"), n_win=("snr", "size")).reset_index()
    s["channel"] = ch
    spec_rows.append(s)
SPEC = pd.concat(spec_rows, ignore_index=True)
A = SPEC.groupby(["plastic", "Frequency (GHz)", "channel"]).agg(
    snr_median_linear=("snr_med", "median"),
    snr_min_spec=("snr_med", "min"),
    signal_mV=("sig_med", "median"),
    own_noise_mV=("noi_med", "median"),
    n_windows=("n_win", "sum")).reset_index()
A["snr_median_dB"] = lin2db(A["snr_median_linear"])
A["snr_min_dB"] = lin2db(A["snr_min_spec"])
A = A.rename(columns={"Frequency (GHz)": "freq"})
A[["plastic", "freq", "channel", "snr_median_linear", "snr_median_dB",
   "snr_min_dB", "signal_mV", "own_noise_mV", "n_windows"]].to_csv(
    os.path.join(OUT, "snr_per_plastic_per_freq.csv"), index=False, sep=";")

# ---- TABLE A subsets ----
srows = []
for name, freqs in SUBSETS.items():
    sub = SPEC[SPEC["Frequency (GHz)"].isin([float(f) for f in freqs])]
    for ch in CHS:
        d = sub[sub["channel"] == ch]
        for p, grp in d.groupby("plastic"):
            med = float(grp["snr_med"].median())
            mn = float(grp["snr_med"].min())
            srows.append({"plastic": p, "subset": name, "channel": ch,
                          "freqs": ",".join(str(f) for f in freqs),
                          "snr_median_linear": med,
                          "snr_median_dB": float(lin2db([med])[0]),
                          "snr_min_dB": float(lin2db([mn])[0]),
                          "n_windows": int(grp["n_win"].sum())})
pd.DataFrame(srows).to_csv(
    os.path.join(OUT, "snr_per_plastic_per_subset.csv"), index=False, sep=";")

# ---- TABLE B: system floor + DR, exp4 refs only ----
frows, drows = [], []
for ch, (mc, sc) in CHS.items():
    for f, grp in t4.groupby("Frequency (GHz)"):
        stds = grp[sc].to_numpy(dtype=float)
        stds = stds[np.isfinite(stds) & (stds > 0)]
        means = grp[mc].to_numpy(dtype=float)
        means = means[np.isfinite(means)]
        med = float(np.median(stds))
        mad = float(np.median(np.abs(stds - med)))
        ult = med + 3 * mad
        mx = float(np.abs(means).max())
        dr = float(20 * np.log10(mx / med)) if med > 0 and mx > 0 else np.nan
        band = "100-210" if f <= 210 else "210-600"
        frows.append({"channel": ch, "freq": f, "n_windows": len(stds),
                      "noise_floor_mV": med, "ult_mV": ult})
        drows.append({"channel": ch, "band": band, "freq": f, "DR_dB": dr})
pd.DataFrame(frows).to_csv(
    os.path.join(OUT, "sys_noise_floor_per_freq.csv"), index=False, sep=";")
D = pd.DataFrame(drows)
band = D.groupby(["channel", "band"]).agg(
    DR_dB_median=("DR_dB", "median"), n_freqs=("DR_dB", "size")).reset_index()
band["freq"] = "BAND_MEDIAN"
Dout = pd.concat([D, band.rename(columns={"DR_dB_median": "DR_dB"})],
                 ignore_index=True).sort_values(["channel", "band", "freq"])
Dout[["channel", "band", "freq", "DR_dB"]].to_csv(
    os.path.join(OUT, "sys_dynamic_range.csv"), index=False, sep=";")

print("wrote 4 csv files", flush=True)

# ---- console summary ----
print("=== per-plastic worst-channel min SNR (dB) + 5 lowest-SNR freqs ===")
for p in sorted(A["plastic"].unique()):
    sub = A[A["plastic"] == p]
    w = sub.loc[sub["snr_min_dB"].idxmin()]
    lo5 = sub.sort_values("snr_median_dB").head(5)
    print(f"{p}: worst min {w['snr_min_dB']:.1f} dB "
          f"({w['channel']} {w['freq']:.0f} GHz); "
          f"lowest-med: " + ", ".join(
              f"{r['freq']:.0f}/{r['channel']}:{r['snr_median_dB']:.1f}dB"
              for _, r in lo5.iterrows()))
print("=== band DR per channel ===")
print(band.to_string(index=False))
print("=== sub-10 dB SNR flags: per plastic x channel (median over freqs) ===")
fl = A.groupby(["plastic", "channel"]).agg(
    worst_med_dB=("snr_median_dB", "min"),
    n_freqs_below_10dB=("snr_median_dB", lambda s: int((s < 10).sum())),
    n_freqs=("snr_median_dB", "size")).reset_index()
print(fl.sort_values("worst_med_dB").to_string(index=False))
