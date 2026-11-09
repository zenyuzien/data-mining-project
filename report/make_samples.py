"""Sample-image figure and dataset statistics for the Introduction (random samples, seed 0 -- not cherry-picked)."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = glob.glob("/home/rushi/dm/extracted/*/fairdomain")[0]
OUT = "/home/rushi/dm/report"
df = pd.read_csv(os.path.join(ROOT, "data_summary.csv"))
tr = df[df.use == "training"].reset_index(drop=True)
rng = np.random.default_rng(0)
picks = {c: rng.choice(np.where(tr.glaucoma == c)[0], 3, replace=False) for c in ("no", "yes")}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
fig, axs = plt.subplots(2, 6, figsize=(6.6, 2.75))
col = 0
for c, name in (("no", "No glaucoma"), ("yes", "Glaucoma")):
    for i in picks[c]:
        r = tr.iloc[i]
        d = np.load(os.path.join(ROOT, "Training", r.filename))
        for row, key in enumerate(("slo_fundus", "oct_fundus")):
            ax = axs[row, col]
            ax.imshow(d[key], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#eb6834" if c == "yes" else "#2a78d6"); s.set_linewidth(1.6)
        axs[1, col].set_xlabel(f"{r.age:.0f} y, MD {r.md:.1f}", fontsize=7, color="#52514e")
        col += 1
axs[0, 0].set_ylabel("SLO fundus"); axs[1, 0].set_ylabel("OCT fundus")
fig.text(0.29, 0.965, "No glaucoma", ha="center", color="#2a78d6", fontsize=9, weight="bold")
fig.text(0.72, 0.965, "Glaucoma", ha="center", color="#eb6834", fontsize=9, weight="bold")
fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.12, wspace=0.05, hspace=0.04)
fig.savefig(os.path.join(OUT, "figs", "samples.pdf"))
print({c: [tr.iloc[i].filename for i in picks[c]] for c in picks})

# dataset statistics
S = {}
for use in ("training", "testing"):
    d = df[df.use == use]
    S[use] = {c: dict(n=int((d.glaucoma == c).sum()), age=float(d[d.glaucoma == c].age.mean()), age_sd=float(d[d.glaucoma == c].age.std()),
                      md=float(d[d.glaucoma == c].md.mean()), md_sd=float(d[d.glaucoma == c].md.std())) for c in ("no", "yes")}
# pixel statistics from a 300-sample subset
sub = tr.sample(300, random_state=0)
px = {k: {c: [] for c in ("no", "yes")} for k in ("slo_fundus", "oct_fundus")}
for _, r in sub.iterrows():
    d = np.load(os.path.join(ROOT, "Training", r.filename))
    for k in px:
        px[k][r.glaucoma].append(d[k].astype(float))
for k in px:
    S[k] = {c: dict(mean=float(np.mean(px[k][c])), sd=float(np.std(px[k][c]))) for c in ("no", "yes")}
print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(OUT, "data_stats.json"), "w"), indent=1)
