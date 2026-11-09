"""Mega-script: run a suite of glaucoma-detection experiments sequentially.

Each experiment runs in its OWN subprocess, so a crash / OOM / CUDA error / timeout in one
cannot stop the others. Results go to results/<name>.json, logs to logs/<name>.log, and a
ranked summary to results/summary.{csv,md}.

Usage:
  python run_all.py                     # run every experiment (skips ones already finished OK)
  python run_all.py --list              # show experiments
  python run_all.py --only resnet50_pt_late vit_small_pt_late
  python run_all.py --smoke             # 1 tiny epoch per experiment: checks everything builds/runs
  python run_all.py --force             # rerun even if a result exists
  python run_all.py --epochs 20 --timeout 7200
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
RES, LOGS, CKPT = (os.path.join(HERE, d) for d in ("results", "logs", "ckpt"))

# name, arch, fusion ('late' = one backbone per modality, 'early' = 2-channel input, 'slo'/'oct' = single),
# pretrained, lr, batch size, weight decay, family
EXPERIMENTS = [
    dict(name="resnet18_scratch_late",   arch="resnet18",                      fusion="late",  pt=False, lr=3e-4, bs=64, wd=1e-2, family="CNN"),
    dict(name="resnet50_pt_late",        arch="resnet50",                      fusion="late",  pt=True,  lr=2e-4, bs=32, wd=1e-2, family="CNN"),
    dict(name="resnet50_pt_early",       arch="resnet50",                      fusion="early", pt=True,  lr=2e-4, bs=32, wd=1e-2, family="CNN"),
    dict(name="efficientnet_b0_pt_late", arch="efficientnet_b0",               fusion="late",  pt=True,  lr=5e-4, bs=32, wd=1e-2, family="CNN"),
    dict(name="densenet121_pt_late",     arch="densenet121",                   fusion="late",  pt=True,  lr=2e-4, bs=32, wd=1e-2, family="CNN"),
    dict(name="convnext_tiny_pt_late",   arch="convnext_tiny",                 fusion="late",  pt=True,  lr=1e-4, bs=32, wd=5e-2, family="Modern CNN"),
    dict(name="vit_small_pt_late",       arch="vit_small_patch16_224",         fusion="late",  pt=True,  lr=5e-5, bs=32, wd=5e-2, family="Transformer"),
    dict(name="deit_small_pt_early",     arch="deit_small_patch16_224",        fusion="early", pt=True,  lr=5e-5, bs=32, wd=5e-2, family="Transformer"),
    dict(name="swin_tiny_pt_late",       arch="swin_tiny_patch4_window7_224",  fusion="late",  pt=True,  lr=5e-5, bs=32, wd=5e-2, family="Transformer"),
    dict(name="coatnet0_pt_late",        arch="coatnet_0_rw_224",              fusion="late",  pt=True,  lr=1e-4, bs=32, wd=5e-2, family="Hybrid CNN+Attn"),
    dict(name="maxvit_tiny_pt_early",    arch="maxvit_tiny_rw_224",            fusion="early", pt=True,  lr=5e-5, bs=16, wd=5e-2, family="Hybrid CNN+Attn"),
    dict(name="efficientnet_b0_pt_slo",  arch="efficientnet_b0",               fusion="slo",   pt=True,  lr=5e-4, bs=32, wd=1e-2, family="Ablation"),
    dict(name="efficientnet_b0_pt_oct",  arch="efficientnet_b0",               fusion="oct",   pt=True,  lr=5e-4, bs=32, wd=1e-2, family="Ablation"),
]
BY_NAME = {e["name"]: e for e in EXPERIMENTS}


# ----------------------------------------------------------------------------- worker side
def build_model(cfg, torch, nn, timm):
    """Returns (model, pretrained_actually_used)."""
    used_pt = cfg["pt"]

    def mk(in_chans):
        nonlocal used_pt
        try:
            return timm.create_model(cfg["arch"], pretrained=cfg["pt"], num_classes=0, in_chans=in_chans)
        except Exception as e:  # download failure, missing weights, etc.
            if not cfg["pt"]:
                raise
            print(f"[warn] pretrained weights unavailable ({type(e).__name__}: {e}); using RANDOM init", flush=True)
            used_pt = False
            return timm.create_model(cfg["arch"], pretrained=False, num_classes=0, in_chans=in_chans)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            f = cfg["fusion"]
            self.f = f
            if f == "late":
                self.a, self.b = mk(1), mk(1)
                dim = self.a.num_features + self.b.num_features
            elif f == "early":
                self.a = mk(2)
                dim = self.a.num_features
            else:  # 'slo' or 'oct'
                self.a = mk(1)
                dim = self.a.num_features
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 2))

        def forward(self, x):
            x = (x - 0.5) / 0.25
            if self.f == "late":
                z = torch.cat([self.a(x[:, :1]), self.b(x[:, 1:])], 1)
            elif self.f == "early":
                z = self.a(x)
            elif self.f == "slo":
                z = self.a(x[:, :1])
            else:
                z = self.a(x[:, 1:])
            return self.head(z)

    m = Net()
    return m, used_pt


def worker(name, epochs, smoke):
    import numpy as np
    import timm
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, HERE)
    from dataset import FairDomainGlaucoma

    cfg = BY_NAME[name]
    out = dict(cfg, status="running", epochs=epochs, smoke=smoke)
    t0 = time.time()
    path = os.path.join(RES, f"{name}{'.smoke' if smoke else ''}.json")

    def dump():
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    try:
        torch.manual_seed(0)
        np.random.seed(0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True

        tr_full = FairDomainGlaucoma("training", augment=True)
        tr_noaug = FairDomainGlaucoma("training", augment=False)
        te = FairDomainGlaucoma("testing")
        idx = np.arange(len(tr_full))
        tr_i, va_i = train_test_split(idx, test_size=0.15, stratify=tr_full.y.numpy(), random_state=0)
        if smoke:
            tr_i, va_i = tr_i[:64], va_i[:32]
            te_sub = np.arange(64)
        else:
            te_sub = np.arange(len(te))
        train_set, val_set, test_set = Subset(tr_full, tr_i), Subset(tr_noaug, va_i), Subset(te, te_sub)
        out["n_train"], out["n_val"], out["n_test"] = len(train_set), len(val_set), len(test_set)

        @torch.no_grad()
        def predict(model, ds):
            model.eval()
            ps, ys = [], []
            for x, y in DataLoader(ds, 64, num_workers=2):
                ps.append(torch.softmax(model(x.to(dev)), 1)[:, 1].float().cpu())
                ys.append(y)
            return torch.cat(ps).numpy(), torch.cat(ys).numpy()

        def train(bs):
            model, used_pt = build_model(cfg, torch, nn, timm)
            model.to(dev)
            out["pretrained_used"] = used_pt
            out["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            out["batch_size"] = bs
            tl = DataLoader(train_set, bs, shuffle=True, num_workers=4, drop_last=True)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, cfg["lr"], total_steps=epochs * len(tl), pct_start=0.15)
            lossf = nn.CrossEntropyLoss()
            best_val, best, hist = -1.0, None, []
            for ep in range(1, epochs + 1):
                model.train()
                tot = 0.0
                for x, y in tl:
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                    loss = lossf(model(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at epoch {ep} (lr too high / unstable)")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tot += loss.item()
                pv, yv = predict(model, val_set)
                pt, yt = predict(model, test_set)
                vauc = float(roc_auc_score(yv, pv))
                tauc = float(roc_auc_score(yt, pt))
                hist.append(dict(epoch=ep, train_loss=tot / len(tl), val_auc=vauc, test_auc=tauc))
                print(f"ep {ep:2d} loss={tot / len(tl):.4f} val_auc={vauc:.4f} test_auc={tauc:.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
                if vauc > best_val:  # model selection on VALIDATION only
                    best_val, best = vauc, dict(epoch=ep, pt=pt, yt=yt)
                    if not smoke:
                        os.makedirs(CKPT, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(CKPT, f"{name}.pt"))
            return best_val, best, hist

        bs = cfg["bs"]
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
        pt, yt = best["pt"], best["yt"]
        out.update(
            status="ok", best_epoch=best["epoch"], val_auc=best_val,
            test_auc=float(roc_auc_score(yt, pt)), test_acc=float(accuracy_score(yt, pt > 0.5)),
            history=hist,
        )
        meta = te.meta.iloc[te_sub].reset_index(drop=True)
        groups = {}
        for attr in ("gender", "race", "ethnicity"):
            for g in sorted(meta[attr].unique()):
                m = (meta[attr] == g).values
                if m.sum() >= 5 and len(np.unique(yt[m])) == 2:
                    groups[f"{attr}:{g}"] = dict(n=int(m.sum()), auc=float(roc_auc_score(yt[m], pt[m])))
        out["group_auc"] = groups
        big = [v["auc"] for k, v in groups.items() if v["n"] >= 100 and not k.startswith("ethnicity")]
        out["worst_group_auc"] = min(big) if big else None
        out["auc_gap_gender_race"] = (max(big) - min(big)) if big else None
    except BaseException as e:  # noqa: BLE001 - we want to record absolutely everything
        out.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(out["traceback"], flush=True)
    finally:
        out["seconds"] = round(time.time() - t0, 1)
        dump()
    sys.exit(0 if out["status"] == "ok" else 1)


# ----------------------------------------------------------------------------- parent side
def load_result(name, smoke):
    p = os.path.join(RES, f"{name}{'.smoke' if smoke else ''}.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def summarize(smoke):
    rows = []
    for e in EXPERIMENTS:
        r = load_result(e["name"], smoke)
        if r is None:
            rows.append(dict(name=e["name"], status="not run"))
        else:
            rows.append(r)
    ok = sorted([r for r in rows if r.get("status") == "ok"], key=lambda r: -r["test_auc"])
    rest = [r for r in rows if r.get("status") != "ok"]
    cols = ["name", "family", "fusion", "arch", "pretrained_used", "params_M", "best_epoch",
            "val_auc", "test_auc", "test_acc", "worst_group_auc", "auc_gap_gender_race", "seconds", "status"]
    suffix = ".smoke" if smoke else ""
    with open(os.path.join(RES, f"summary{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in ok + rest:
            w.writerow(r)
    fmt = lambda v: "-" if v is None or v == "" else (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = ["| # | name | family | fusion | pretrained | test AUC | val AUC | acc | worst-grp AUC | time(s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok, 1):
        lines.append(f"| {i} | {r['name']} | {r['family']} | {r['fusion']} | {r['pretrained_used']} | "
                     f"{fmt(r['test_auc'])} | {fmt(r['val_auc'])} | {fmt(r['test_acc'])} | "
                     f"{fmt(r.get('worst_group_auc'))} | {r['seconds']:.0f} |")
    for r in rest:
        err = (r.get("error") or "")[:80]
        lines.append(f"| - | {r['name']} | | | | **{r.get('status')}** {err} | | | | |")
    md = "\n".join(lines)
    with open(os.path.join(RES, f"summary{suffix}.md"), "w") as f:
        f.write(md + "\n")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="per-experiment seconds")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--worker")
    a = ap.parse_args()

    if a.worker:
        worker(a.worker, 1 if a.smoke else a.epochs, a.smoke)
        return
    if a.list:
        for e in EXPERIMENTS:
            print(f"{e['name']:28s} {e['family']:16s} {e['arch']:32s} fusion={e['fusion']:5s} pretrained={e['pt']}")
        return

    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    names = a.only or [e["name"] for e in EXPERIMENTS]
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        sys.exit(f"unknown experiment(s): {bad}. Use --list.")

    t_all = time.time()
    try:
        for i, n in enumerate(names, 1):
            prev = load_result(n, a.smoke)
            if prev and prev.get("status") == "ok" and not a.force:
                print(f"[{i}/{len(names)}] {n}: already done (test AUC {prev['test_auc']:.4f}), skipping", flush=True)
                continue
            print(f"[{i}/{len(names)}] {n}: starting ...", flush=True)
            cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--worker", n, "--epochs", str(a.epochs)]
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
                # worker died without writing a proper result (segfault, kill, timeout) - record it
                r = dict(BY_NAME[n], status="failed", error=err, seconds=round(time.time() - t0, 1))
                with open(os.path.join(RES, f"{n}{'.smoke' if a.smoke else ''}.json"), "w") as f:
                    json.dump(r, f, indent=2)
            if r and r.get("status") == "ok":
                print(f"    -> OK  test AUC {r['test_auc']:.4f}  val AUC {r['val_auc']:.4f}  ({r['seconds']:.0f}s)", flush=True)
            else:
                print(f"    -> FAILED: {(r or {}).get('error', err)}  (continuing)", flush=True)
            summarize(a.smoke)  # refresh summary after each experiment so partial results are never lost
    except KeyboardInterrupt:
        print("\ninterrupted - writing summary of what finished", flush=True)
    print(f"\nTotal {time.time() - t_all:.0f}s\n")
    print(summarize(a.smoke))
    print(f"\nSaved: {RES}/summary{'.smoke' if a.smoke else ''}.csv / .md")


if __name__ == "__main__":
    main()
