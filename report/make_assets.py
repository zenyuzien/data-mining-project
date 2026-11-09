"""Builds every table (report/tables/*.tex) and figure (report/figs/*.pdf) of the report from the raw result files,
so no number in the report is typed by hand. Run:  ./venv/bin/python report/make_assets.py
Also writes report/numbers.json with the key numbers used in the text.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/home/rushi/dm"
OUT = os.path.join(ROOT, "report")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figs")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, ROOT)
from run_next import metrics  # noqa: E402  (shared metric code: group AUC, ES-AUC, worst-group)

BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

# ----------------------------------------------------------------------------- load
S1 = {r["name"]: r for r in (json.load(open(p)) for p in glob.glob(f"{ROOT}/results/*.json") if ".smoke" not in p and "summary" not in p)}
S2 = {r["name"]: r for r in (json.load(open(p)) for p in glob.glob(f"{ROOT}/results_next/*.json") if ".smoke" not in p and "ensemble" not in p)}
S3 = {r["name"]: r for r in (json.load(open(p)) for p in glob.glob(f"{ROOT}/results_sweep3/*.json") if ".smoke" not in p and "ensembles" not in p)}
T0 = {r["name"]: r for r in json.load(open(f"{ROOT}/results_t0/tier0.json"))}
ENS3 = json.load(open(f"{ROOT}/results_sweep3/ensembles.json"))
ENS2 = json.load(open(f"{ROOT}/results_next/ensemble.json"))

csvp = glob.glob(f"{ROOT}/extracted/*/fairdomain/data_summary.csv")[0]
DF = pd.read_csv(csvp)
META = DF[DF["use"] == "testing"].reset_index(drop=True)
Y = (META["glaucoma"] == "yes").astype(int).values


def probs(sweep, name, key="test"):
    d = {1: "results_t0", 2: "results_next", 3: "results_sweep3"}[sweep]
    return np.load(f"{ROOT}/{d}/probs/{name}.npz")[key]


def boot_ci(p, y=Y, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


f3 = lambda v: "--" if v is None else f"{v:.3f}"
NUM = {}

# ----------------------------------------------------------------------------- dataset table
def pct(n, d):
    return f"{n} ({100 * n / d:.0f}\\%)"


rows = []
for use, nm in (("training", "Training"), ("testing", "Testing")):
    d = DF[DF["use"] == use]
    n = len(d)
    rows.append([nm, n, pct((d.glaucoma == "yes").sum(), n), pct((d.gender == "female").sum(), n),
                 pct((d.race == "white").sum(), n), pct((d.race == "black").sum(), n), pct((d.race == "asian").sum(), n),
                 pct((d.ethnicity == "hispanic").sum(), n)])
with open(f"{TAB}/dataset.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n Split & $N$ & Glaucoma & Female & White & Black & Asian & Hispanic \\\\\n\\midrule\n")
    for r in rows:
        f.write(" & ".join(str(x) for x in r) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- label maps
LAB1 = {"resnet18_scratch_late": "ResNet18 (scratch)", "resnet50_pt_late": "ResNet50", "resnet50_pt_early": "ResNet50 (early fusion)",
        "efficientnet_b0_pt_late": "EfficientNet-B0", "densenet121_pt_late": "DenseNet121", "convnext_tiny_pt_late": "ConvNeXt-Tiny",
        "vit_small_pt_late": "ViT-Small", "deit_small_pt_early": "DeiT-Small (early fusion)", "swin_tiny_pt_late": "Swin-Tiny",
        "coatnet0_pt_late": "CoAtNet-0", "maxvit_tiny_pt_early": "MaxViT-Tiny (early fusion)",
        "efficientnet_b0_pt_slo": "EfficientNet-B0, SLO only", "efficientnet_b0_pt_oct": "EfficientNet-B0, OCT only"}
FAM = {"CNN": "CNN", "Modern CNN": "CNN", "Transformer": "Transformer", "Hybrid CNN+Attn": "Hybrid", "Ablation": "Single modality"}

# ----------------------------------------------------------------------------- sweep 1 table (all 13)
order = sorted(S1.values(), key=lambda r: -r["test_auc"])
with open(f"{TAB}/sweep1.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\nModel & Type & Params (M) & Val AUC & Test AUC & TTA AUC & Acc. & Worst-group AUC \\\\\n\\midrule\n")
    for r in order:
        t0 = T0.get(r["name"], {})
        f.write(f"{LAB1[r['name']]} & {FAM[r['family']]} & {r['params_M']:.1f} & {f3(r['val_auc'])} & {f3(r['test_auc'])} & "
                f"{f3(t0.get('test_auc_tta'))} & {f3(r['test_acc'])} & {f3(r['worst_group_auc'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
NUM["s1_best"] = max(S1.values(), key=lambda r: r["test_auc"])["name"]

# ----------------------------------------------------------------------------- sweep 2 table
LAB2 = {"base_recipe": "Swin-T + recipe", "base_recipe_s1": "Swin-T + recipe (seed 1)", "base_recipe_s2": "Swin-T + recipe (seed 2)",
        "base_gated_moddrop": "Gated fusion + modality dropout", "vit_small_xattn": "ViT-S cross-attention fusion",
        "dinov2_small_late": "DINOv2-S fine-tune", "base_fair_groupw": "Group-reweighted loss", "base_fair_fis_like": "FIS-like loss",
        "base_fair_adv_race": "Adversarial race debiasing", "base_xdom_slo2oct": "Train SLO $\\to$ test OCT",
        "base_xdom_oct2slo": "Train OCT $\\to$ test SLO", "base_tab_age": "+ age as input", "base_tab_age_md_UPPERBOUND": "+ age and MD (upper bound)"}
order = sorted(S2.values(), key=lambda r: -r["test_auc"])
with open(f"{TAB}/sweep2.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nVariant (Swin-Tiny backbone) & Val AUC & Test AUC & 95\\% CI & ES-AUC race & Worst-group & Other-domain AUC \\\\\n\\midrule\n")
    for r in order:
        ci = r.get("test_auc_ci95")
        f.write(f"{LAB2[r['name']]} & {f3(r['val_auc'])} & {f3(r['test_auc'])} & [{ci[0]:.3f}, {ci[1]:.3f}] & {f3(r.get('es_auc_race'))} & "
                f"{f3(r.get('worst_group_auc'))} & {f3(r.get('test_auc_other_domain'))} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
sw = [S2[n]["test_auc"] for n in ("base_recipe", "base_recipe_s1", "base_recipe_s2")]
NUM["swin_recipe_mean"], NUM["swin_recipe_std"] = float(np.mean(sw)), float(np.std(sw, ddof=1))
NUM["md_upper"] = S2["base_tab_age_md_UPPERBOUND"]["test_auc"]
NUM["age_only"] = S2["base_tab_age"]["test_auc"]

# ----------------------------------------------------------------------------- sweep 3 table (tiers 1-2, seeds, tiers 3-4)
LAB3 = {"eb0_recipe": "EfficientNet-B0 + recipe", "eb0_recipe_s1": "\\quad seed 1", "eb0_recipe_s2": "\\quad seed 2",
        "eb0_recipe_ep12": "\\quad 12 epochs", "eb0_mixup": "\\quad + mixup/CutMix", "eb0_droppath": "\\quad + drop-path 0.2, dropout 0.5",
        "eb0_lowlr": "\\quad lower LR (2e-4)", "eb0_gated_moddrop": "\\quad gated fusion + modality dropout",
        "effnet_b2": "EfficientNet-B2", "effnet_b3": "EfficientNet-B3", "regnety_016": "RegNetY-016", "densenet121": "DenseNet121",
        "best_fair_adv": "Drop-path config + adversarial race debiasing"}
names3 = ["eb0_recipe", "eb0_recipe_s1", "eb0_recipe_s2", "eb0_recipe_ep12", "eb0_mixup", "eb0_droppath", "eb0_lowlr",
          "eb0_gated_moddrop", "effnet_b2", "effnet_b3", "regnety_016", "densenet121", "best_fair_adv"]
with open(f"{TAB}/sweep3.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nConfiguration & Val AUC & Test AUC & 95\\% CI & TTA AUC & ES-AUC race & Worst-group \\\\\n\\midrule\n")
    for n in names3:
        r = S3[n]
        ci = r["test_auc_ci95"]
        f.write(f"{LAB3[n]} & {f3(r['val_auc'])} & {f3(r['test_auc'])} & [{ci[0]:.3f}, {ci[1]:.3f}] & {f3(r['tta']['test_auc'])} & "
                f"{f3(r.get('es_auc_race'))} & {f3(r.get('worst_group_auc'))} \\\\\n")
        if n == "eb0_gated_moddrop":
            f.write("\\midrule\n")
        if n == "densenet121":
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
sd = [S3[n]["test_auc"] for n in ("eb0_recipe", "eb0_recipe_s1", "eb0_recipe_s2")]
NUM["eb0_recipe_mean"], NUM["eb0_recipe_std"] = float(np.mean(sd)), float(np.std(sd, ddof=1))
NUM["eb0_recipe_val_mean"] = float(np.mean([S3[n]["val_auc"] for n in ("eb0_recipe", "eb0_recipe_s1", "eb0_recipe_s2")]))
NUM["droppath"] = S3["eb0_droppath"]["test_auc"]
NUM["s1_effnet"] = S1["efficientnet_b0_pt_late"]["test_auc"]
NUM["s1_effnet_val"] = S1["efficientnet_b0_pt_late"]["val_auc"]
NUM["trainval"] = [S3[f"best_trainval_s{i}"]["test_auc"] for i in range(3)]
NUM["cv_folds"] = [S3[f"best_cv_f{i}"]["test_auc"] for i in range(5)]

# ----------------------------------------------------------------------------- ablation table
LABA = {"abl_noema": "EMA off", "abl_headlr1": "Head LR $\\times$1 (instead of $\\times$5)", "abl_flipaug": "Flip + mild jitter only (no rotate/scale/shift)",
        "abl_sweep1_repro": "All three reverted, no label smoothing, 15 epochs (approx. sweep-1 setup)"}
with open(f"{TAB}/ablation.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\nEfficientNet-B0 variant & Val AUC & Test AUC & TTA AUC \\\\\n\\midrule\n")
    f.write(f"Recipe (3 seeds, mean $\\pm$ s.d.) & {NUM['eb0_recipe_val_mean']:.3f} & {NUM['eb0_recipe_mean']:.3f} $\\pm$ {NUM['eb0_recipe_std']:.3f} & "
            f"{np.mean([S3[n]['tta']['test_auc'] for n in ('eb0_recipe', 'eb0_recipe_s1', 'eb0_recipe_s2')]):.3f} \\\\\n\\midrule\n")
    for n in ("abl_noema", "abl_headlr1", "abl_flipaug", "abl_sweep1_repro"):
        r = S3.get(n)
        if r and r.get("status") == "ok":
            f.write(f"{LABA[n]} & {f3(r['val_auc'])} & {f3(r['test_auc'])} & {f3(r['tta']['test_auc'])} \\\\\n")
    f.write(f"Sweep-1 run (original code, one seed) & {f3(S1['efficientnet_b0_pt_late']['val_auc'])} & {f3(S1['efficientnet_b0_pt_late']['test_auc'])} & "
            f"{f3(T0['efficientnet_b0_pt_late']['test_auc_tta'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
for n in ("abl_noema", "abl_headlr1", "abl_flipaug", "abl_sweep1_repro"):
    if n in S3 and S3[n].get("status") == "ok":
        NUM[n] = S3[n]["test_auc"]

# ----------------------------------------------------------------------------- ensemble table + CIs
ens_rows = []  # (label, source, n_members, p_plain, p_tta, member_test_aucs) -- only ensembles of at most 3 models
t0_ens = {r["name"]: r for r in json.load(open(f"{ROOT}/results_t0/tier0.json")) if r["kind"] == "ensemble"}
m = t0_ens["top3_by_val"]["members"]
ens_rows.append(("Top 3 of sweep 1 by validation AUC (Swin-T, ViT-S, DenseNet121)", "Sweep 1", len(m),
                 np.mean([probs(1, x) for x in m], 0), np.mean([probs(1, x, "test_tta") for x in m], 0), [S1[x]["test_auc"] for x in m]))
for n, lab in (("trainval3", "3 seeds trained on train+val (EfficientNet-B0)"), ("tier12_top3", "Top 3 of the recipe study by validation AUC")):
    m = ENS3[n]["members"]
    assert len(m) <= 3
    ens_rows.append((lab, "Sweep 3", len(m), np.mean([probs(3, x) for x in m], 0), np.mean([probs(3, x, "test_tta") for x in m], 0),
                     [S3[x]["test_auc"] for x in m]))
ens_rows.sort(key=lambda r: -roc_auc_score(Y, r[3]))
with open(f"{TAB}/ensembles.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nEnsemble (average of probabilities) & $n$ & Members' mean AUC & Test AUC & 95\\% CI & TTA AUC & ES-AUC race \\\\\n\\midrule\n")
    for lab, src, k, p, pt, ma in ens_rows:
        ci = boot_ci(p)
        mm = metrics(p, Y, META, boot=10)
        f.write(f"{lab} & {k} & {np.mean(ma):.3f} & {roc_auc_score(Y, p):.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {roc_auc_score(Y, pt):.3f} & {mm['es_auc_race']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
NUM["ens3"] = {r[0]: dict(auc=float(roc_auc_score(Y, r[3])), members_mean=float(np.mean(r[5]))) for r in ens_rows}

# ----------------------------------------------------------------------------- per-group table
sel = [("EfficientNet-B0 (sweep 1)", probs(1, "efficientnet_b0_pt_late")), ("Swin-T + recipe (sweep 2)", probs(2, "base_recipe")),
       ("EfficientNet-B0 + drop-path (sweep 3)", probs(3, "eb0_droppath")),
       ("3-model ensemble, top 3 of sweep 1", [r for r in ens_rows if r[1] == "Sweep 1"][0][3]),
       ("3-model ensemble, top 3 of recipe study", [r for r in ens_rows if r[0].startswith("Top 3 of the recipe")][0][3])]
cols = ["gender:female", "gender:male", "race:asian", "race:black", "race:white", "ethnicity:non-hispanic", "ethnicity:hispanic"]
with open(f"{TAB}/groups.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & Female & Male & Asian & Black & White & Non-Hisp. & Hispanic \\\\\n")
    n_row = " & ".join(str(int(((META[c.split(':')[0]] == c.split(':')[1])).sum())) for c in cols)
    f.write(f"$n$ (test) & {n_row} \\\\\n\\midrule\n")
    for lab, p in sel:
        g = metrics(p, Y, META, boot=10)["group_auc"]
        f.write(f"{lab} & " + " & ".join(f3(g[c]["auc"]) for c in cols) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- hyper-parameter table
HP = [("ResNet18 (scratch)", "3e-4", 64, "1e-2"), ("ResNet50", "2e-4", 32, "1e-2"), ("EfficientNet-B0", "5e-4", 32, "1e-2"),
      ("DenseNet121", "2e-4", 32, "1e-2"), ("ConvNeXt-Tiny", "1e-4", 32, "5e-2"), ("ViT-S / DeiT-S / Swin-T", "5e-5", 32, "5e-2"),
      ("CoAtNet-0", "1e-4", 32, "5e-2"), ("MaxViT-Tiny", "5e-5", 16, "5e-2")]
with open(f"{TAB}/hparams.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\nBackbone & Peak LR & Batch size & Weight decay \\\\\n\\midrule\n")
    for r in HP:
        f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------------------- FIGURE: results overview (dot plot with 95% CI)
items = []  # (label, auc, lo, hi, kind)
def add(label, p, kind):
    ci = boot_ci(p)
    items.append((label, float(roc_auc_score(Y, p)), ci[0], ci[1], kind))
for lab, src, k, p, pt, ma in ens_rows:
    if lab.startswith(("Top 3 of the recipe", "Top 3 of sweep 1")):
        add("3-model ensemble: " + ("recipe study top 3" if "recipe" in lab else "sweep-1 top 3"), p, "Ensemble")
add("EfficientNet-B0 (sweep 1)", probs(1, "efficientnet_b0_pt_late"), "Single model")
add("EfficientNet-B0 + drop-path", probs(3, "eb0_droppath"), "Single model")
add("RegNetY-016 + recipe", probs(3, "regnety_016"), "Single model")
add("Swin-T + recipe", probs(2, "base_recipe"), "Single model")
add("Swin-T (sweep 1)", probs(1, "swin_tiny_pt_late"), "Single model")
add("DenseNet121 (sweep 1)", probs(1, "densenet121_pt_late"), "Single model")
add("ViT-S (sweep 1)", probs(1, "vit_small_pt_late"), "Single model")
add("ConvNeXt-T (sweep 1)", probs(1, "convnext_tiny_pt_late"), "Single model")
add("ResNet50 (sweep 1)", probs(1, "resnet50_pt_late"), "Single model")
add("ResNet18 from scratch", probs(1, "resnet18_scratch_late"), "Baseline")
items.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.4, 4.3))
colors = {"Ensemble": ORANGE, "Single model": BLUE, "Baseline": GRAY}
for i, (lab, a, lo, hi, kind) in enumerate(items):
    ax.plot([lo, hi], [i, i], color=colors[kind], lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot([a], [i], "o", ms=6, color=colors[kind], mec="white", mew=1.2, zorder=3)
    ax.text(hi + 0.003, i, f"{a:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([r[0] for r in items], fontsize=8)
ax.set_xlim(0.69, 0.885)
ax.set_xlabel("Test AUC (point = estimate, line = bootstrap 95% CI)")
ax.grid(axis="x", color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
for k, c in colors.items():
    ax.plot([], [], "o", color=c, label=k)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/results_overview.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------- FIGURE: overfitting curves
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
for r, lab, c in ((S1["efficientnet_b0_pt_late"], "Sweep 1 (plain)", BLUE), (S3["eb0_recipe"], "Recipe (EMA, strong aug.)", ORANGE)):
    h = r["history"]
    ep = [x["epoch"] for x in h]
    axs[0].plot(ep, [x["train_loss"] for x in h], color=c, lw=1.6, label=lab)
    axs[1].plot(ep, [x["val_auc"] for x in h], color=c, lw=1.6, label=lab)
axs[0].set_yscale("log")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Training loss")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Validation AUC")
for a in axs:
    a.grid(color="#e6e5e1", lw=0.6)
    a.set_axisbelow(True)
axs[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIG}/overfitting.pdf")
plt.close(fig)
NUM["s1_effnet_final_loss"] = S1["efficientnet_b0_pt_late"]["history"][-1]["train_loss"] if "history" in S1["efficientnet_b0_pt_late"] else None

# ----------------------------------------------------------------------------- FIGURE: modality contribution
fig, ax = plt.subplots(figsize=(3.4, 2.3))
vals = [("SLO only", S1["efficientnet_b0_pt_slo"]["test_auc"]), ("OCT only", S1["efficientnet_b0_pt_oct"]["test_auc"]),
        ("SLO + OCT", S1["efficientnet_b0_pt_late"]["test_auc"])]
for i, (lab, v) in enumerate(vals):
    ax.bar(i, v, width=0.55, color=BLUE if i == 2 else GRAY, zorder=2)
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(3))
ax.set_xticklabels([v[0] for v in vals], fontsize=8)
ax.set_ylim(0.70, 0.85)
ax.set_ylabel("Test AUC")
ax.grid(axis="y", color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{FIG}/modality.pdf")
plt.close(fig)


# ============================================================================= MAIN-PAPER ASSETS (final model = single two-branch EfficientNet-B0, plain recipe)
def rowinfo(label, p, ptta, kind, modal):
    ci = boot_ci(p)
    m = metrics(p, Y, META, boot=10)
    return dict(label=label, kind=kind, modal=modal, auc=float(roc_auc_score(Y, p)), lo=ci[0], hi=ci[1],
                tta=None if ptta is None else float(roc_auc_score(Y, ptta)), es=m["es_auc_race"], worst=m["worst_group_auc"])

MAIN = [
    rowinfo("ResNet18, two-branch, from scratch (baseline)", probs(1, "resnet18_scratch_late"), probs(1, "resnet18_scratch_late", "test_tta"), "Baseline", "SLO+OCT"),
    rowinfo("EfficientNet-B0, SLO only", probs(1, "efficientnet_b0_pt_slo"), probs(1, "efficientnet_b0_pt_slo", "test_tta"), "Other EfficientNet-B0 runs", "SLO"),
    rowinfo("EfficientNet-B0, OCT only", probs(1, "efficientnet_b0_pt_oct"), probs(1, "efficientnet_b0_pt_oct", "test_tta"), "Other EfficientNet-B0 runs", "OCT"),
    rowinfo("Final: EfficientNet-B0, two-branch", probs(1, "efficientnet_b0_pt_late"), probs(1, "efficientnet_b0_pt_late", "test_tta"), "Final model", "SLO+OCT"),
    rowinfo("Same model, re-run with similar settings", probs(3, "abl_sweep1_repro"), probs(3, "abl_sweep1_repro", "test_tta"), "Other EfficientNet-B0 runs", "SLO+OCT"),
    rowinfo("Same architecture, stronger recipe + drop-path", probs(3, "eb0_droppath"), probs(3, "eb0_droppath", "test_tta"), "Other EfficientNet-B0 runs", "SLO+OCT"),
    rowinfo("Same architecture, stronger recipe, flips only", probs(3, "abl_flipaug"), probs(3, "abl_flipaug", "test_tta"), "Other EfficientNet-B0 runs", "SLO+OCT"),
]
NUM["final"] = {k: MAIN[3][k] for k in ("auc", "lo", "hi", "tta", "es", "worst")}
all_eb0 = [S3[n]["test_auc"] for n in S3 if n.startswith(("eb0_", "abl_"))]
NUM["eb0_other_runs_min"], NUM["eb0_other_runs_max"] = float(min(all_eb0)), float(max(all_eb0))
with open(f"{TAB}/main_results.tex", "w") as f:
    f.write("\\begin{tabular}{lcrrrrr}\n\\toprule\nModel & Input & Test AUC & 95\\% CI & + flip TTA & ES-AUC race & Worst-group AUC \\\\\n\\midrule\n")
    for i, r in enumerate(MAIN):
        if i in (1, 3, 4):
            f.write("\\midrule\n")
        lab = r["label"] if i != 3 else "\\textbf{" + r["label"] + "}"
        f.write(f"{lab} & {r['modal']} & {r['auc']:.3f} & [{r['lo']:.3f}, {r['hi']:.3f}] & {f3(r['tta'])} & {r['es']:.3f} & {r['worst']:.3f} \\\\\n")
        if i == 6:
            f.write(f"Same architecture, stronger recipe, 3 seeds (mean $\\pm$ s.d.) & SLO+OCT & {NUM['eb0_recipe_mean']:.3f} $\\pm$ {NUM['eb0_recipe_std']:.3f} & -- & -- & -- & -- \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

fig, ax = plt.subplots(figsize=(6.4, 2.9))
cols_k = {"Final model": ORANGE, "Other EfficientNet-B0 runs": BLUE, "Baseline": GRAY}
short = ["ResNet18 from scratch (baseline)", "EfficientNet-B0, SLO only", "EfficientNet-B0, OCT only", "Final: EfficientNet-B0 SLO+OCT",
         "Same model, re-run (similar settings)", "Stronger recipe + drop-path", "Stronger recipe, flips only"]
for i, r in enumerate(MAIN):
    y = len(MAIN) - 1 - i
    ax.plot([r["lo"], r["hi"]], [y, y], color=cols_k[r["kind"]], lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot([r["auc"]], [y], "o", ms=6, color=cols_k[r["kind"]], mec="white", mew=1.2, zorder=3)
    ax.text(r["hi"] + 0.003, y, f"{r['auc']:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(MAIN)))
ax.set_yticklabels(short[::-1], fontsize=8)
ax.set_xlim(0.69, 0.885)
ax.set_xlabel("Test AUC (point = estimate, line = bootstrap 95% CI)")
ax.grid(axis="x", color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
for k, c in cols_k.items():
    ax.plot([], [], "o", color=c, label=k)
ax.legend(frameon=False, loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/results_main.pdf")
plt.close(fig)

sel_main = [("ResNet18 from scratch (baseline)", probs(1, "resnet18_scratch_late")), ("EfficientNet-B0, SLO only", probs(1, "efficientnet_b0_pt_slo")),
            ("EfficientNet-B0, OCT only", probs(1, "efficientnet_b0_pt_oct")), ("Final: EfficientNet-B0, two-branch", probs(1, "efficientnet_b0_pt_late"))]
with open(f"{TAB}/groups_main.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & Female & Male & Asian & Black & White & Non-Hisp. & Hispanic \\\\\n")
    f.write(f"$n$ (test) & " + " & ".join(str(int(((META[c.split(':')[0]] == c.split(':')[1])).sum())) for c in cols) + " \\\\\n\\midrule\n")
    for lab, p in sel_main:
        g = metrics(p, Y, META, boot=10)["group_auc"]
        f.write(f"{lab} & " + " & ".join(f3(g[c]["auc"]) for c in cols) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

json.dump(NUM, open(f"{OUT}/numbers.json", "w"), indent=2, default=float)
print(json.dumps(NUM, indent=1, default=float))
