"""Mega-script #2: the "next experiments" from LITERATURE_AND_NEXT_RUNS.md.

Run AFTER run_all.py has finished (it picks the best backbone from results/*.json automatically).

Groups of experiments (all use a stronger recipe: GPU augmentation incl. rotate/scale/shift, EMA weights,
label smoothing, validation-based model selection):
  Recipe      base_recipe (+ seeds s1, s2 for mean/std)
  Fusion      gated fusion + modality dropout, ViT cross-attention fusion
  Foundation  DINOv2-small (self-supervised ViT) fine-tune
  Fairness    group-reweighted loss, "FIS-like" loss, adversarial race debiasing
  Domain      cross-domain transfer: train SLO -> test OCT, and the reverse
  Tabular     + age, and + age + md (md is almost the label: clearly-labelled UPPER BOUND, not a fair result)
Then an ensemble of the top-3 runs by validation AUC.

Every experiment runs in its own subprocess (timeout, OOM retry, NaN guard, pretrained fallback), so one
failure never stops the others. Results: results_next/<name>.json, probs in results_next/probs/,
logs in logs_next/, ranked summary in results_next/summary.{csv,md}. Resumable (skips finished; --force redoes).

Metrics added vs run_all: ES-AUC (race/gender/ethnicity), bootstrap 95% CIs. ES-AUC here is
AUC / (1 + sum_g |AUC - AUC_g|) -- written from memory of the Harvard FairVision papers; VERIFY against the paper.
The "FIS-like" loss is an approximation of the idea (individual x group loss scaling), not the paper's exact method.

Usage:
  python run_next.py --list
  python run_next.py                       # everything
  python run_next.py --smoke --base-arch efficientnet_b0   # quick wiring check
  python run_next.py --only base_recipe base_fair_groupw
  python run_next.py --epochs 25 --timeout 14400 --base-arch convnext_tiny
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, LOGS, CKPT = (os.path.join(HERE, d) for d in ("results_next", "logs_next", "ckpt_next"))
PROBS = os.path.join(RES, "probs")
PREV_RES = os.path.join(HERE, "results")


def E(name, family, **kw):
    d = dict(name=name, family=family, arch="BASE", fusion="late", pt=True, loss="ce", tab=[], mod_drop=0.0,
             seed=0, xdom=None, aug="strong", ema=True, ls=0.1, head_lr_mult=5.0, kw={})
    d.update(kw)
    return d


EXPERIMENTS = [
    E("base_recipe", "Recipe"),
    E("base_gated_moddrop", "Fusion", fusion="gated", mod_drop=0.2),
    E("vit_small_xattn", "Fusion", arch="vit_small_patch16_224", fusion="xattn", lr=5e-5, bs=32, wd=5e-2),
    E("dinov2_small_late", "Foundation", arch="vit_small_patch14_dinov2.lvd142m", lr=3e-5, bs=32, wd=5e-2,
      kw=dict(img_size=224)),
    E("base_fair_groupw", "Fairness", loss="groupw"),
    E("base_fair_fis_like", "Fairness", loss="fis"),
    E("base_fair_adv_race", "Fairness", loss="adv"),
    E("base_xdom_slo2oct", "Domain", fusion="slo", xdom="slo2oct"),
    E("base_xdom_oct2slo", "Domain", fusion="oct", xdom="oct2slo"),
    E("base_tab_age", "Tabular", tab=["age"]),
    E("base_tab_age_md_UPPERBOUND", "Tabular", tab=["age", "md"]),
    E("base_recipe_s1", "Seeds", seed=1),
    E("base_recipe_s2", "Seeds", seed=2),
]
BY_NAME = {e["name"]: e for e in EXPERIMENTS}


# ----------------------------------------------------------------------------- shared metrics
def metrics(p, y, meta, boot=300):
    from sklearn.metrics import accuracy_score, roc_auc_score
    auc = float(roc_auc_score(y, p))
    out = dict(test_auc=auc, test_acc=float(accuracy_score(y, p > 0.5)))
    rng = np.random.default_rng(0)

    def ci(pp, yy):
        vals = []
        for _ in range(boot):
            ii = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[ii])) == 2:
                vals.append(roc_auc_score(yy[ii], pp[ii]))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None

    out["test_auc_ci95"] = ci(p, y)
    groups = {}
    for attr in ("gender", "race", "ethnicity"):
        for g in sorted(meta[attr].unique()):
            m = (meta[attr] == g).values
            if m.sum() >= 20 and len(np.unique(y[m])) == 2:
                groups[f"{attr}:{g}"] = dict(n=int(m.sum()), auc=float(roc_auc_score(y[m], p[m])), ci95=ci(p[m], y[m]))
    out["group_auc"] = groups
    for attr in ("gender", "race", "ethnicity"):
        gs = [v["auc"] for k, v in groups.items() if k.startswith(attr + ":")]
        out[f"es_auc_{attr}"] = float(auc / (1 + sum(abs(auc - g) for g in gs))) if gs else None
    big = [v["auc"] for k, v in groups.items() if v["n"] >= 100 and not k.startswith("ethnicity")]
    out["worst_group_auc"] = min(big) if big else None
    out["auc_gap_gender_race"] = (max(big) - min(big)) if big else None
    return out


# ----------------------------------------------------------------------------- worker
def worker(cfg, epochs, smoke):
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    sys.path.insert(0, HERE)
    from dataset import FairDomainGlaucoma

    name = cfg["name"]
    out = dict(cfg, status="running", epochs=epochs, smoke=smoke)
    t0 = time.time()
    suffix = ".smoke" if smoke else ""
    path = os.path.join(RES, f"{name}{suffix}.json")

    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.backends.cudnn.benchmark = True

        tr, te = FairDomainGlaucoma("training"), FairDomainGlaucoma("testing")
        tr_i, va_i = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=tr.y.numpy(), random_state=0)
        if smoke:
            tr_i, va_i, te_i = tr_i[:64], va_i[:32], np.arange(64)
        else:
            te_i = np.arange(len(te))
        out["n_train"], out["n_val"], out["n_test"] = len(tr_i), len(va_i), len(te_i)

        Xtr, Ytr, Xte = tr.x.to(dev), tr.y.to(dev), te.x.to(dev)
        yte = te.y.numpy()[te_i]
        races = sorted(tr.meta["race"].unique())
        Rtr = torch.tensor([races.index(r) for r in tr.meta["race"]], device=dev)
        n_race = len(races)

        tabs = cfg["tab"]
        Ttr = Tte = None
        if tabs:
            a = tr.meta[tabs].values.astype(np.float32)
            mu, sd = a[tr_i].mean(0), a[tr_i].std(0) + 1e-6
            Ttr = torch.tensor((a - mu) / sd, device=dev)
            Tte = torch.tensor((te.meta[tabs].values.astype(np.float32) - mu) / sd, device=dev)

        # ---- GPU augmentation (flip always; strong = rotate/scale/shift too)
        def augment(x):
            B = x.shape[0]
            flip = torch.rand(B, device=dev) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            amp = 1.0
            if cfg["aug"] == "strong":
                ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(15)
                sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.1
                tx, ty = ((torch.rand(B, device=dev) * 2 - 1) * 0.1 for _ in range(2))
                c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
                theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
                x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                                  padding_mode="border", align_corners=False)
                amp = 2.0
            x = x * (1 + 0.1 * amp * (torch.rand(B, 2, 1, 1, device=dev) * 2 - 1)) \
                + 0.05 * amp * (torch.rand(B, 2, 1, 1, device=dev) * 2 - 1)
            return x.clamp(0, 1)

        # ---- model
        used_pt = [cfg["pt"]]

        def mk(in_chans):
            try:
                return timm.create_model(cfg["arch"], pretrained=cfg["pt"], num_classes=0, in_chans=in_chans, **cfg["kw"])
            except Exception as e:  # noqa: BLE001
                if not cfg["pt"]:
                    raise
                print(f"[warn] pretrained unavailable ({type(e).__name__}: {e}); RANDOM init", flush=True)
                used_pt[0] = False
                return timm.create_model(cfg["arch"], pretrained=False, num_classes=0, in_chans=in_chans, **cfg["kw"])

        class GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lam):
                ctx.lam = lam
                return x.view_as(x)

            @staticmethod
            def backward(ctx, g):
                return -ctx.lam * g, None

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = cfg["fusion"]
                f = self.f
                if f in ("late", "gated", "xattn"):
                    self.a, self.b = mk(1), mk(1)
                    da, db = self.a.num_features, self.b.num_features
                elif f == "early":
                    self.a = mk(2)
                    da = db = self.a.num_features
                else:
                    self.a = mk(1)
                    da = db = self.a.num_features
                if f == "late":
                    dim = da + db
                elif f == "gated":
                    self.pa, self.pb = nn.Linear(da, 256), nn.Linear(db, 256)
                    self.gate = nn.Linear(512, 256)
                    dim = 256
                elif f == "xattn":
                    heads = 6 if da % 6 == 0 else 8
                    self.ca = nn.MultiheadAttention(da, heads, batch_first=True)
                    self.cb = nn.MultiheadAttention(db, heads, batch_first=True)
                    self.na, self.nb = nn.LayerNorm(da), nn.LayerNorm(db)
                    dim = da + db
                else:
                    dim = da
                if tabs:
                    self.tab = nn.Sequential(nn.Linear(len(tabs), 16), nn.ReLU())
                    dim += 16
                self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 2))
                self.adv = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, n_race)) if cfg["loss"] == "adv" else None

            def forward(self, x, tab=None, lam=1.0):
                x = (x - 0.5) / 0.25
                f = self.f
                if f in ("late", "gated"):
                    fa, fb = self.a(x[:, :1]), self.b(x[:, 1:])
                    if self.training and cfg["mod_drop"] > 0:
                        r = torch.rand(x.shape[0], 1, device=x.device)
                        fa = fa * (~(r < cfg["mod_drop"] / 2)).float()
                        fb = fb * (~((r >= cfg["mod_drop"] / 2) & (r < cfg["mod_drop"]))).float()
                    if f == "late":
                        z = torch.cat([fa, fb], 1)
                    else:
                        pa, pb = self.pa(fa), self.pb(fb)
                        g = torch.sigmoid(self.gate(torch.cat([pa, pb], 1)))
                        z = g * pa + (1 - g) * pb
                elif f == "xattn":
                    ta, tb = self.a.forward_features(x[:, :1]), self.b.forward_features(x[:, 1:])
                    ta2 = self.na(ta + self.ca(ta, tb, tb, need_weights=False)[0])
                    tb2 = self.nb(tb + self.cb(tb, ta, ta, need_weights=False)[0])
                    z = torch.cat([ta2[:, 0], tb2[:, 0]], 1)
                elif f == "early":
                    z = self.a(x)
                elif f == "slo":
                    z = self.a(x[:, :1])
                else:  # oct
                    z = self.a(x[:, 1:])
                if tabs:
                    z = torch.cat([z, self.tab(tab)], 1)
                adv = self.adv(GRL.apply(z, lam)) if self.adv is not None else None
                return self.head(z), adv

        # ---- loss weights for groupw
        ytr_np = tr.y.numpy()
        rtr_np = Rtr.cpu().numpy()
        cell = np.ones((n_race, 2))
        for r in range(n_race):
            for c in range(2):
                cell[r, c] = max(1, ((rtr_np[tr_i] == r) & (ytr_np[tr_i] == c)).sum())
        cw = torch.tensor(len(tr_i) / (n_race * 2 * cell), dtype=torch.float32, device=dev)
        cw = cw / cw[torch.tensor(rtr_np[tr_i], device=dev), torch.tensor(ytr_np[tr_i], device=dev)].mean()

        @torch.no_grad()
        def predict(m, X, T, idx, swap=False):
            m.eval()
            ps = []
            for s in range(0, len(idx), 128):
                b = torch.as_tensor(idx[s:s + 128], device=dev)
                x = X[b].float() / 255
                if swap:
                    x = x[:, [1, 0]]
                lg, _ = m(x, T[b] if T is not None else None)
                ps.append(torch.softmax(lg, 1)[:, 1].float().cpu())
            return torch.cat(ps).numpy()

        def train(bs):
            torch.manual_seed(cfg["seed"])
            model = Net().to(dev)
            out["pretrained_used"] = used_pt[0]
            out["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            out["batch_size"] = bs
            ema = None
            if cfg["ema"]:
                import copy
                ema = copy.deepcopy(model).eval()
            head_params = [p for n, p in model.named_parameters() if n.startswith(("head", "adv", "tab", "pa", "pb", "gate", "ca", "cb", "na", "nb"))]
            hid = {id(p) for p in head_params}
            body = [p for p in model.parameters() if id(p) not in hid]
            lr = cfg["lr"]
            opt = torch.optim.AdamW([dict(params=body, lr=lr), dict(params=head_params, lr=lr * cfg["head_lr_mult"])],
                                    weight_decay=cfg["wd"])
            spe = max(1, len(tr_i) // bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, [lr, lr * cfg["head_lr_mult"]], total_steps=epochs * spe, pct_start=0.15)
            tr_idx = torch.as_tensor(tr_i, device=dev)
            best_val, best, hist, step = -1.0, None, [], 0
            for ep in range(1, epochs + 1):
                model.train()
                perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
                tot = 0.0
                for s in range(spe):
                    idx = perm[s * bs:(s + 1) * bs]
                    x = augment(Xtr[idx].float() / 255)
                    y, r = Ytr[idx], Rtr[idx]
                    tab = Ttr[idx] if tabs else None
                    prog = step / (epochs * spe)
                    lam = 2 / (1 + math.exp(-10 * prog)) - 1
                    lg, adv = model(x, tab, lam)
                    ce = F.cross_entropy(lg, y, reduction="none", label_smoothing=cfg["ls"])
                    if cfg["loss"] == "groupw":
                        w = cw[r, y]
                    elif cfg["loss"] == "fis":
                        with torch.no_grad():
                            mean = ce.mean() + 1e-8
                            gl = torch.zeros(n_race, device=dev).index_add_(0, r, ce) / \
                                torch.zeros(n_race, device=dev).index_add_(0, r, torch.ones_like(ce)).clamp(min=1)
                            w = ((ce / mean) * (gl[r] / mean)).clamp(0.2, 5.0)
                            w = w / w.mean()
                    else:
                        w = torch.ones_like(ce)
                    loss = (w * ce).mean()
                    if adv is not None:
                        loss = loss + 0.5 * F.cross_entropy(adv, r)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at epoch {ep}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    step += 1
                    tot += loss.item()
                    if ema is not None:
                        d = min(0.995, (1 + step) / (10 + step))
                        with torch.no_grad():
                            for pe, pm in zip(ema.parameters(), model.parameters()):
                                pe.lerp_(pm, 1 - d)
                            for be, bm in zip(ema.buffers(), model.buffers()):
                                be.copy_(bm)
                ev = ema if ema is not None else model
                pv = predict(ev, Xtr, Ttr, va_i)
                pt = predict(ev, Xte, Tte, te_i)
                vauc = float(roc_auc_score(Ytr.cpu().numpy()[va_i], pv))
                tauc = float(roc_auc_score(yte, pt))
                row = dict(epoch=ep, train_loss=tot / spe, val_auc=vauc, test_auc=tauc)
                if cfg["xdom"]:
                    row["test_auc_other_domain"] = float(roc_auc_score(yte, predict(ev, Xte, Tte, te_i, swap=True)))
                hist.append(row)
                print(f"ep {ep:2d} loss={tot / spe:.4f} val_auc={vauc:.4f} test_auc={tauc:.4f}"
                      + (f" other_domain={row['test_auc_other_domain']:.4f}" if cfg["xdom"] else "")
                      + f" [{time.time() - t0:.0f}s]", flush=True)
                if vauc > best_val:  # selection on VALIDATION only
                    best_val = vauc
                    best = dict(epoch=ep, pt=pt, pv=pv, row=row)
                    if not smoke:
                        os.makedirs(CKPT, exist_ok=True)
                        torch.save(ev.state_dict(), os.path.join(CKPT, f"{name}.pt"))
            return best_val, best, hist

        bs = min(cfg["bs"], 8) if smoke else cfg["bs"]
        for attempt in range(3):
            try:
                best_val, best, hist = train(bs)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs //= 2
                print(f"[warn] CUDA OOM -> retrying with batch size {bs}", flush=True)
                if bs < 2 or attempt == 2:
                    raise

        meta = te.meta.iloc[te_i].reset_index(drop=True)
        out.update(status="ok", best_epoch=best["epoch"], val_auc=best_val, history=hist)
        out.update(metrics(best["pt"], yte, meta))
        if cfg["xdom"]:
            out["test_auc_other_domain"] = best["row"].get("test_auc_other_domain")
        os.makedirs(PROBS, exist_ok=True)
        np.savez(os.path.join(PROBS, f"{name}{suffix}.npz"), test=best["pt"], val=best["pv"], y_test=yte)
    except BaseException as e:  # noqa: BLE001
        out.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(out["traceback"], flush=True)
    finally:
        out["seconds"] = round(time.time() - t0, 1)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
    sys.exit(0 if out["status"] == "ok" else 1)


# ----------------------------------------------------------------------------- parent
def resolve_base(arch_override):
    """Pick the best late-fusion pretrained backbone from run_all.py results (by validation AUC)."""
    if arch_override:
        defaults = dict(efficientnet_b0=(5e-4, 32, 1e-2), resnet50=(2e-4, 32, 1e-2), convnext_tiny=(1e-4, 32, 5e-2),
                        densenet121=(2e-4, 32, 1e-2), resnet18=(3e-4, 64, 1e-2), swin_tiny_patch4_window7_224=(5e-5, 32, 5e-2),
                        vit_small_patch16_224=(5e-5, 32, 5e-2), coatnet_0_rw_224=(1e-4, 32, 5e-2))
        lr, bs, wd = defaults.get(arch_override, (1e-4, 32, 1e-2))
        return dict(arch=arch_override, lr=lr, bs=bs, wd=wd, source="--base-arch")
    best = None
    for p in glob.glob(os.path.join(PREV_RES, "*.json")):
        if ".smoke" in p:
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("status") == "ok" and r.get("fusion") == "late" and r.get("pt") and r.get("family") != "Ablation":
            if best is None or r["val_auc"] > best["val_auc"]:
                best = r
    if best is None:
        return dict(arch="efficientnet_b0", lr=5e-4, bs=32, wd=1e-2, source="fallback (no run_all results found)")
    return dict(arch=best["arch"], lr=best["lr"], bs=best["bs"], wd=best["wd"], source=f"best of run_all: {best['name']}")


def load_result(name, smoke):
    p = os.path.join(RES, f"{name}{'.smoke' if smoke else ''}.json")
    try:
        return json.load(open(p))
    except Exception:
        return None


def summarize(smoke):
    rows = [load_result(e["name"], smoke) or dict(name=e["name"], status="not run") for e in EXPERIMENTS]
    ens = None
    try:
        ens = json.load(open(os.path.join(RES, "ensemble.json")))
    except Exception:
        pass
    ok = sorted([r for r in rows if r.get("status") == "ok"], key=lambda r: -r["test_auc"])
    rest = [r for r in rows if r.get("status") != "ok"]
    cols = ["name", "family", "arch", "fusion", "loss", "tab", "pretrained_used", "best_epoch", "val_auc", "test_auc",
            "es_auc_race", "es_auc_gender", "worst_group_auc", "auc_gap_gender_race", "test_auc_other_domain", "seconds", "status"]
    sfx = ".smoke" if smoke else ""
    with open(os.path.join(RES, f"summary{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in ok + rest:
            w.writerow(r)
    fm = lambda v: "-" if v is None or v == "" else (f"{v:.4f}" if isinstance(v, float) else str(v))
    L = ["| # | name | family | test AUC [95% CI] | val AUC | ES-AUC race | ES-AUC gender | worst-grp AUC | other-domain AUC | s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        ci = r.get("test_auc_ci95")
        cis = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        L.append(f"| {i} | {r['name']} | {r['family']} | {r['test_auc']:.4f}{cis} | {fm(r['val_auc'])} | {fm(r.get('es_auc_race'))} | "
                 f"{fm(r.get('es_auc_gender'))} | {fm(r.get('worst_group_auc'))} | {fm(r.get('test_auc_other_domain'))} | {r['seconds']:.0f} |")
    for r in rest:
        L.append(f"| - | {r['name']} | | **{r.get('status')}** {(r.get('error') or '')[:70]} | | | | | | |")
    if ens:
        L.append(f"\nEnsemble of {ens['members']}: test AUC {ens['test_auc']:.4f}, ES-AUC race {fm(ens.get('es_auc_race'))}, "
                 f"worst-group AUC {fm(ens.get('worst_group_auc'))}")
    seeds = [r for r in ok if r["name"] in ("base_recipe", "base_recipe_s1", "base_recipe_s2")]
    if len(seeds) >= 2:
        a = np.array([r["test_auc"] for r in seeds])
        L.append(f"\nSeed variability of base_recipe ({len(a)} seeds): test AUC {a.mean():.4f} +- {a.std(ddof=1):.4f}")
    L.append("\nNOTE: base_tab_age_md_UPPERBOUND uses `md` (nearly the label) -- not a fair comparison.")
    md = "\n".join(L)
    open(os.path.join(RES, f"summary{sfx}.md"), "w").write(md + "\n")
    return md


def make_ensemble(smoke):
    """Average test probabilities of the top-3 (by validation AUC) fair, fusion-model runs."""
    import pandas as pd
    cands = []
    for e in EXPERIMENTS:
        r = load_result(e["name"], smoke)
        if r and r.get("status") == "ok" and "UPPER" not in e["name"] and not e["xdom"] and e["fusion"] not in ("slo", "oct"):
            cands.append(r)
    if len(cands) < 2:
        return
    cands = sorted(cands, key=lambda r: -r["val_auc"])[:3]
    sfx = ".smoke" if smoke else ""
    ps = [np.load(os.path.join(PROBS, f"{r['name']}{sfx}.npz")) for r in cands]
    p = np.mean([z["test"] for z in ps], 0)
    y = ps[0]["y_test"]
    root = glob.glob("/home/rushi/dm/extracted/*/fairdomain")[0]
    df = pd.read_csv(os.path.join(root, "data_summary.csv"))
    meta = df[df["use"] == "testing"].reset_index(drop=True).iloc[:len(y)]
    res = metrics(p, y, meta)
    res["members"] = [r["name"] for r in cands]
    json.dump(res, open(os.path.join(RES, "ensemble.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--base-arch", default=None, help="override backbone auto-selection")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--worker")
    a = ap.parse_args()

    if a.worker:
        worker(json.loads(a.worker), 1 if a.smoke else a.epochs, a.smoke)
        return
    if a.list:
        for e in EXPERIMENTS:
            print(f"{e['name']:30s} {e['family']:11s} arch={e['arch']:36s} fusion={e['fusion']:6s} loss={e['loss']:7s} tab={e['tab']}")
        return

    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    names = a.only or [e["name"] for e in EXPERIMENTS]
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        sys.exit(f"unknown experiment(s): {bad}. Use --list.")
    base = resolve_base(a.base_arch)
    print(f"BASE backbone: {base}", flush=True)

    t_all = time.time()
    try:
        for i, n in enumerate(names, 1):
            prev = load_result(n, a.smoke)
            if prev and prev.get("status") == "ok" and not a.force:
                print(f"[{i}/{len(names)}] {n}: already done (test AUC {prev['test_auc']:.4f}), skipping", flush=True)
                continue
            cfg = dict(BY_NAME[n])
            if cfg["arch"] == "BASE":
                cfg.update(arch=base["arch"], lr=base["lr"], bs=base["bs"], wd=base["wd"])
            print(f"[{i}/{len(names)}] {n}: starting ({cfg['arch']}) ...", flush=True)
            cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--worker", json.dumps(cfg), "--epochs", str(a.epochs)]
            if a.smoke:
                cmd.append("--smoke")
            log = os.path.join(LOGS, f"{n}{'.smoke' if a.smoke else ''}.log")
            t0 = time.time()
            try:
                with open(log, "w") as lf:
                    rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=HERE).returncode
                err = None if rc == 0 else f"worker exited with code {rc} (see {log})"
            except subprocess.TimeoutExpired:
                err = f"timeout after {a.timeout}s"
            except Exception as e:  # noqa: BLE001
                err = f"launcher error: {type(e).__name__}: {e}"
            r = load_result(n, a.smoke)
            if err and (r is None or r.get("status") != "failed"):
                r = dict(cfg, status="failed", error=err, seconds=round(time.time() - t0, 1))
                json.dump(r, open(os.path.join(RES, f"{n}{'.smoke' if a.smoke else ''}.json"), "w"), indent=2)
            if r and r.get("status") == "ok":
                print(f"    -> OK  test AUC {r['test_auc']:.4f}  val AUC {r['val_auc']:.4f}  ({r['seconds']:.0f}s)", flush=True)
            else:
                print(f"    -> FAILED: {(r or {}).get('error', err)}  (continuing)", flush=True)
            summarize(a.smoke)
    except KeyboardInterrupt:
        print("\ninterrupted - writing summary of what finished", flush=True)
    try:
        make_ensemble(a.smoke)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ensemble step failed: {type(e).__name__}: {e}", flush=True)
    print(f"\nTotal {time.time() - t_all:.0f}s\n")
    print(summarize(a.smoke))
    print(f"\nSaved: {RES}/summary{'.smoke' if a.smoke else ''}.csv / .md")


if __name__ == "__main__":
    main()
