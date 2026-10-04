# R1.4 follow-up 3: canonical K-subset extension (K1/K5/K50) + re-verified lists.
# Canonical source = results/exp_5_v2/lodo_stability.csv universals (baseline identical
# to alpha for every K, asserted below) + lodo_per_fold.csv modal slots where the
# universal set is short (K1: empty; K3: 2/3; K5: 4/5). No manuscript file exists in
# the repo (doc/ holds only sample info) — Table-3 print check left to the author.
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
from collections import Counter

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


def lin2db(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 20 * np.log10(x), np.nan)


# ---- (a) recover canonical lists from training outputs ----
stab = pd.read_csv(os.path.join(REPO, "results", "exp_5_v2",
                                "lodo_stability.csv"), sep=";")
pf = pd.read_csv(os.path.join(REPO, "results", "exp_5_v2", "lodo_per_fold.csv"),
                 sep=";")
CANON, NOTES = {}, {}


def universal(k):
    u = {}
    for norm in ("baseline", "alpha"):
        u[norm] = sorted(stab[(stab["K"] == k) & (stab["norm_mode"] == norm)
                              & (stab["in_universal"])]["freq"].tolist())
    assert u["baseline"] == u["alpha"], f"K{k} arms differ!"
    return u["baseline"]


def complete(k):
    """Exact-K consensus: universal set topped up to K by extra-slot fold
    count desc, stability mean-rank asc (the nested selector's own ordering).
    Returns (canon_list, note)."""
    u = universal(k)
    if len(u) >= k:
        return sorted(u)[:k], f"universal set complete ({len(u)}/{k})"
    folds = pf[(pf["K"] == k)].drop_duplicates("fold")["selected_freqs"]
    c = Counter()
    for s in folds:
        c.update(int(x) for x in str(s).split(",") if int(x) not in u)
    mr = {int(r["freq"]): float(r["mean_rank"])
          for _, r in stab[(stab["K"] == k)
                           & (stab["norm_mode"] == "baseline")].iterrows()}
    fill = sorted(c, key=lambda f: (-c[f], mr.get(f, 1e9)))[:k - len(u)]
    assert len(u) + len(fill) == k, (k, u, fill)
    note = (f"universal ({len(u)}) {[int(x) for x in u]} + modal " +
            ", ".join(f"{f} ({c[f]}/5 folds, mr {mr.get(f, float('nan')):.1f})"
                      for f in fill))
    return sorted([int(x) for x in u] + fill), note


for _k in (1, 3, 5, 10, 20, 50):
    CANON[f"K{_k}"], NOTES[f"K{_k}"] = complete(_k)
# sanity: K50 must be the full sweep; K10/K20 spot-checks
assert CANON["K50"] == list(range(100, 591, 10)), "K50 is NOT the full 50!"
assert CANON["K10"] == [320, 330, 340, 350, 360, 370, 400, 420, 450, 460]
assert CANON["K20"] == [230, 310, 320, 330, 340, 350, 360, 370, 380, 390,
                        400, 410, 420, 430, 440, 450, 460, 470, 500, 520]
assert all(len(v) == int(k[1:]) for k, v in CANON.items()), "exact-K violated"
SUBSETS = {k: CANON[k] for k in ("K1", "K3", "K5", "K10", "K20", "K50")}
for k, v in SUBSETS.items():
    print(f"{k} (n={len(v)}): {v} | {NOTES[k]}", flush=True)

with open(os.path.join(OUT, "subset_lists_canonical.txt"), "w") as fh:
    for k in ("K1", "K3", "K5", "K10", "K20", "K50"):
        fh.write(f"{k}: {','.join(str(f) for f in SUBSETS[k])}\n")
    fh.write("--- notes ---\n")
    for k in ("K1", "K3", "K5", "K10", "K20", "K50"):
        fh.write(f"{k}: {NOTES[k]}\n")

# ---- (b) subset summaries, identical two-stage rule ----
e5_files = sorted(glob.glob(os.path.join(
    REPO, "data", "experiment_5_plastics", "processed", "*.csv")))
assert len(e5_files) == 60
frames = []
for f in e5_files:
    df = pd.read_csv(f, delimiter=";")
    df["SourceFile"] = os.path.basename(f)
    df["Day"] = 1
    frames.append(df)
t5 = tv.grouped_window_averages(
    tv.apply_pre_sg(pd.concat(frames, ignore_index=True), SG_W, SG_P), DP)
t5["plastic"] = t5["Sample"].str[0]
spec_rows = []
for ch, (mc, sc) in CHS.items():
    d = t5[["plastic", "SourceFile", "Frequency (GHz)", mc, sc]].copy()
    d = d[np.isfinite(d[mc]) & np.isfinite(d[sc]) & (d[sc] > 0)]
    d["snr"] = d[mc].abs() / d[sc]
    g = d.groupby(["plastic", "SourceFile", "Frequency (GHz)"])
    s = g.agg(snr_med=("snr", "median"), n_win=("snr", "size")).reset_index()
    s["channel"] = ch
    spec_rows.append(s)
SPEC = pd.concat(spec_rows, ignore_index=True)

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
NEW = pd.DataFrame(srows)
old_path = os.path.join(OUT, "snr_per_plastic_per_subset.csv")
OLD = pd.read_csv(old_path, sep=";")
key = ["plastic", "subset", "channel"]
# Regression scope: K1/K3/K5 only (K10 8->exact 10 and K20 19->exact 20
# legitimately changed freqs). SNR medians must be bit-identical.
prev = OLD[OLD["subset"].isin(["K1", "K3", "K5"])]
chk = NEW.merge(prev, on=key, suffixes=("", "_old"))
assert (abs(chk["snr_median_dB"] - chk["snr_median_dB_old"]).max() < 1e-9), \
    "K3/K10/K20 rows changed!"
# n_windows cross-check against TABLE A (independent column): subset window
# sums must equal TABLE-A n_windows summed over the subset's freqs.
A = pd.read_csv(os.path.join(OUT, "snr_per_plastic_per_freq.csv"), sep=";")
for _, r in NEW.iterrows():
    fs = [float(x) for x in str(r["freqs"]).split(",")]
    expect = A[(A["plastic"] == r["plastic"]) & (A["channel"] == r["channel"])
               & (A["freq"].isin(fs))]["n_windows"].sum()
    assert int(expect) == int(r["n_windows"]), (r["subset"], r["plastic"])
print(f"regression check OK: {len(chk)} K1/K3/K5 rows identical + "
      f"TABLE-A window sums match all {len(NEW)} rows", flush=True)
NEW.to_csv(old_path, index=False, sep=";")
print(f"wrote {len(NEW)} rows (expect {12 * 6 * 2}=144)", flush=True)
print(NEW[NEW["subset"].isin(["K1", "K5"])].sort_values(
    ["subset", "channel", "snr_median_dB"]).to_string(index=False))
