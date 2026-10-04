# R1.4 follow-ups 1+2: live-band minima + pooled effective SNR (exp5 only).
# Same smoothing/windowing reuse as run_r14.py (apply_pre_sg w=7 p=2,
# grouped_window_averages 0.1 s, dp=(100/12)*0.1). No ML, no commits.
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
OUT = os.path.join(REPO, "results", "r14_snr_noise_dr")

_spec = importlib.util.spec_from_file_location(
    "train_v2", os.path.join(REPO, "src", "scripts", "train_v2.py"))
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

SG_W, SG_P = 7, 2
DP = (100 / 12) * 0.1
CHS = {"HG": ("HG (mV) mean", "HG (mV) std deviation"),
       "LG": ("LG (mV) mean", "LG (mV) std deviation")}
LIVE = {"LG-live": ("LG", 110.0, 210.0), "HG-live": ("HG", 320.0, 590.0)}


def lin2db(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 20 * np.log10(x), np.nan)


e5_files = sorted(glob.glob(os.path.join(
    REPO, "data", "experiment_5_plastics", "processed", "*.csv")))
assert len(e5_files) == 60
frames = []
for f in e5_files:
    df = pd.read_csv(f, delimiter=";")
    df["SourceFile"] = os.path.basename(f)
    df["Day"] = 1
    frames.append(df)
e5 = pd.concat(frames, ignore_index=True)

t5 = tv.grouped_window_averages(tv.apply_pre_sg(e5, SG_W, SG_P), DP)
t5["plastic"] = t5["Sample"].str[0]

# specimen-level medians (same two-stage rule as TABLE A)
spec_rows = []
for ch, (mc, sc) in CHS.items():
    d = t5[["plastic", "SourceFile", "Frequency (GHz)", mc, sc]].copy()
    d = d[np.isfinite(d[mc]) & np.isfinite(d[sc]) & (d[sc] > 0)]
    d["snr"] = d[mc].abs() / d[sc]
    g = d.groupby(["plastic", "SourceFile", "Frequency (GHz)"])
    s = g.agg(snr_med=("snr", "median")).reset_index()
    s["channel"] = ch
    spec_rows.append(s)
SPEC = pd.concat(spec_rows, ignore_index=True)

# ---- follow-up 1: live-band minima, derived from TABLE A ----
# (First attempt pooled medians across the band by mistake — caught because
# B/320 reads -5.7 dB in TABLE A while the pooled version printed +26 dB.
# Correct rule per spec: min OVER FREQS of the per-freq specimen-medians.)
A = pd.read_csv(os.path.join(OUT, "snr_per_plastic_per_freq.csv"), sep=";")


def band(df, ch, lo, hi, name):
    d = df[(df["channel"] == ch) & (df["freq"] >= lo) & (df["freq"] <= hi)]
    return (d.groupby("plastic")
            .agg(worst_median_dB=("snr_median_dB", "min"),
                 worst_min_dB=("snr_min_dB", "min"),
                 n_freqs=("freq", "nunique"))
            .reset_index().assign(liveband=name, channel=ch))
LIVE_DF = pd.concat([band(A, "LG", 110.0, 210.0, "LG-live"),
                     band(A, "HG", 320.0, 590.0, "HG-live"),
                     band(A, "HG", 330.0, 590.0, "HG-live(330+)")],
                    ignore_index=True)
LIVE_DF.to_csv(os.path.join(OUT, "snr_liveband_minima.csv"),
               index=False, sep=";")
print("=== live-band minima per plastic ===")
print(LIVE_DF.sort_values(["liveband", "worst_median_dB"]).to_string(
    index=False))
sub0 = LIVE_DF[LIVE_DF["worst_median_dB"] < 0]
print(f"sub-0 dB live-band worst-medians: {len(sub0)}")

# ---- follow-up 2: pooled effective SNR ----
# Reading (documented): windows are the units. Grand mean = mean of window
# means pooled across the 5 specimens; denominator = std of window means /
# sqrt(n_windows) = standard error of the mean level.
prows = []
for ch, (mc, sc) in CHS.items():
    d = t5[["plastic", "Frequency (GHz)", mc, sc]].copy()
    d = d[np.isfinite(d[mc]) & np.isfinite(d[sc]) & (d[sc] > 0)]
    for (p, f), grp in d.groupby(["plastic", "Frequency (GHz)"]):
        m = grp[mc].to_numpy(dtype=float)
        grand = float(m.mean())
        sem = float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else np.nan
        eff = abs(grand) / sem if sem and sem > 0 else np.nan
        prows.append({"plastic": p, "freq": f, "channel": ch,
                      "grand_mean_mV": grand, "snr_eff_linear": eff,
                      "snr_eff_dB": float(lin2db([eff])[0]),
                      "n_windows": int(len(m))})
P = pd.DataFrame(prows)
P.to_csv(os.path.join(OUT, "snr_per_plastic_per_freq_pooled.csv"),
         index=False, sep=";")
print(f"pooled rows: {len(P)} (expect 1200)")
print("=== pooled SNR_eff extremes ===")
print(P.sort_values("snr_eff_dB").head(5).to_string(index=False))
print(P.sort_values("snr_eff_dB", ascending=False).head(5).to_string(
    index=False))
print(f"sub-0 dB pooled: {int((P['snr_eff_dB'] < 0).sum())}")
