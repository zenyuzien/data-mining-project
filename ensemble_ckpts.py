"""Tier 0 of sweep 3 (no training): rebuild the 13 sweep-1 models from ckpt/*.pt, compute validation/test
probabilities (plain + horizontal-flip TTA), then evaluate ensembles. Subset choice uses VALIDATION AUC only.

Outputs: results_t0/probs/<name>.npz, results_t0/tier0.json, results_t0/tier0.md
Usage: python ensemble_ckpts.py [--smoke]
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_t0")
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    import timm
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    import run_all
    from dataset import FairDomainGlaucoma
    from run_next import metrics

    os.makedirs(os.path.join(OUT, "probs"), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = FairDomainGlaucoma("training"), FairDomainGlaucoma("testing")
    _, va_i = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=tr.y.numpy(), random_state=0)
    te_i = np.arange(64 if a.smoke else len(te))
    if a.smoke:
        va_i = va_i[:32]
    yv, yt = tr.y.numpy()[va_i], te.y.numpy()[te_i]
    meta = te.meta.iloc[te_i].reset_index(drop=True)

    @torch.no_grad()
    def predict(model, X, idx, tta):
        model.eval()
        ps = []
        for s in range(0, len(idx), 128):
            x = X[torch.as_tensor(idx[s:s + 128])].float().to(dev) / 255
            pr = torch.softmax(model(x), 1)[:, 1]
            if tta:
                pr = (pr + torch.softmax(model(x.flip(-1)), 1)[:, 1]) / 2
            ps.append(pr.float().cpu())
        return torch.cat(ps).numpy()

    per = {}
    for e in run_all.EXPERIMENTS:
        name = e["name"]
        ck = os.path.join(HERE, "ckpt", f"{name}.pt")
        if not os.path.exists(ck):
            print(f"[skip] {name}: no checkpoint", flush=True)
            continue
        t0 = time.time()
        try:
            cfg = dict(e, pt=False)  # weights come from the checkpoint; avoid re-downloading pretrained ones
            model, _ = run_all.build_model(cfg, torch, nn, timm)
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model.to(dev)
            z = dict(val=predict(model, tr.x, va_i, False), val_tta=predict(model, tr.x, va_i, True),
                     test=predict(model, te.x, te_i, False), test_tta=predict(model, te.x, te_i, True))
            np.savez(os.path.join(OUT, "probs", f"{name}.npz"), y_test=yt, y_val=yv, **z)
            per[name] = dict(z, family=e["family"], fusion=e["fusion"])
            print(f"[ok] {name}: val {roc_auc_score(yv, z['val']):.4f} test {roc_auc_score(yt, z['test']):.4f} "
                  f"test+TTA {roc_auc_score(yt, z['test_tta']):.4f} ({time.time() - t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001 - one broken checkpoint must not stop the rest
            print(f"[FAILED] {name}: {type(ex).__name__}: {ex}\n{traceback.format_exc()}", flush=True)

    if len(per) < 2:
        sys.exit("fewer than 2 models available; nothing to ensemble")

    vauc = {n: roc_auc_score(yv, z["val"]) for n, z in per.items()}
    fusion = [n for n in per if per[n]["fusion"] in ("late", "early")]          # exclude SLO-only / OCT-only ablations
    pretrained = [n for n in fusion if n != "resnet18_scratch_late"]
    by_val = sorted(pretrained, key=lambda n: -vauc[n])

    # greedy forward selection (with replacement) on validation AUC
    chosen, cur, best_v = [], None, -1
    for _ in range(8):
        cand = {n: roc_auc_score(yv, np.mean([per[m]["val"] for m in chosen + [n]], 0)) for n in pretrained}
        n = max(cand, key=cand.get)
        if cand[n] <= best_v + 1e-4:
            break
        chosen.append(n)
        best_v = cand[n]
    diverse = [n for n in ("efficientnet_b0_pt_late", "swin_tiny_pt_late", "densenet121_pt_late",
                           "vit_small_pt_late", "coatnet0_pt_late") if n in per]
    sets = {"all_pretrained_fusion": pretrained, "top3_by_val": by_val[:3], "top5_by_val": by_val[:5],
            "diverse5": diverse, "greedy_by_val": chosen}

    rows = []
    for n in sorted(per, key=lambda k: -vauc[k]):
        r = metrics(per[n]["test"], yt, meta, boot=100)
        rows.append(dict(name=n, kind="single", members=[n], val_auc=vauc[n], test_auc=r["test_auc"],
                         test_auc_tta=roc_auc_score(yt, per[n]["test_tta"]), es_auc_race=r["es_auc_race"],
                         worst_group_auc=r["worst_group_auc"]))
    for sname, members in sets.items():
        if len(set(members)) < 2:
            continue
        p = np.mean([per[m]["test"] for m in members], 0)
        pt = np.mean([per[m]["test_tta"] for m in members], 0)
        r = metrics(p, yt, meta)
        rows.append(dict(name=sname, kind="ensemble", members=members,
                         val_auc=roc_auc_score(yv, np.mean([per[m]["val"] for m in members], 0)),
                         test_auc=r["test_auc"], test_auc_ci95=r["test_auc_ci95"],
                         test_auc_tta=roc_auc_score(yt, pt), es_auc_race=r["es_auc_race"],
                         es_auc_gender=r["es_auc_gender"], worst_group_auc=r["worst_group_auc"]))
    json.dump(rows, open(os.path.join(OUT, "tier0.json"), "w"), indent=2)

    f = lambda v: "-" if v is None else f"{v:.4f}"
    L = ["| name | kind | #members | val AUC | test AUC [95% CI] | test AUC + flip TTA | ES-AUC race | worst-grp AUC |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["test_auc"]):
        ci = r.get("test_auc_ci95")
        cis = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        L.append(f"| {r['name']} | {r['kind']} | {len(r['members'])} | {f(r['val_auc'])} | {r['test_auc']:.4f}{cis} | "
                 f"{f(r['test_auc_tta'])} | {f(r['es_auc_race'])} | {f(r['worst_group_auc'])} |")
    L.append("\nEnsemble members:")
    for sname, members in sets.items():
        L.append(f"- {sname}: {', '.join(members)}")
    open(os.path.join(OUT, "tier0.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
