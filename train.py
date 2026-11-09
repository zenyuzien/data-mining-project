"""Baseline: two-branch ResNet18 (SLO branch + OCT branch) with late feature fusion.

Usage:
  python train.py                      # fusion, 15 epochs
  python train.py --mode slo           # SLO only
  python train.py --mode oct           # OCT only
  python train.py --pretrained         # ImageNet weights (needs internet)
"""
import argparse

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import FairDomainGlaucoma


class FusionNet(nn.Module):
    def __init__(self, mode="fusion", pretrained=False, arch="resnet18"):
        super().__init__()
        self.mode = mode
        mk = lambda: timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=1)
        self.slo = mk() if mode in ("fusion", "slo") else None
        self.oct = mk() if mode in ("fusion", "oct") else None
        dim = (self.slo.num_features if self.slo else 0) + (self.oct.num_features if self.oct else 0)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 2))

    def forward(self, x):
        # x: B,2,H,W with channel 0 = SLO, 1 = OCT. Normalise to roughly zero-mean.
        x = (x - 0.5) / 0.25
        feats = []
        if self.slo is not None:
            feats.append(self.slo(x[:, 0:1]))
        if self.oct is not None:
            feats.append(self.oct(x[:, 1:2]))
        return self.head(torch.cat(feats, dim=1))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(dev)), 1)[:, 1].cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def report(p, y, meta):
    print(f"  overall  AUC={roc_auc_score(y, p):.4f}  ACC={accuracy_score(y, p > 0.5):.4f}")
    for attr in ("gender", "race", "ethnicity"):
        parts = []
        for g in sorted(meta[attr].unique()):
            m = (meta[attr] == g).values
            if len(np.unique(y[m])) < 2:
                continue
            parts.append(f"{g}(n={m.sum()}) AUC={roc_auc_score(y[m], p[m]):.3f}")
        print(f"  {attr:9s}", " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fusion", choices=["fusion", "slo", "oct"])
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = FairDomainGlaucoma("training", augment=True)
    te = FairDomainGlaucoma("testing")
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=4, drop_last=True)
    el = DataLoader(te, 128, num_workers=2)

    model = FusionNet(a.mode, a.pretrained, a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    lossf = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        p, y = evaluate(model, el, dev)
        auc = roc_auc_score(y, p)
        print(f"ep {ep:2d}  train_loss={tot / len(tl):.4f}  test_AUC={auc:.4f}  ACC={accuracy_score(y, p > 0.5):.4f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), f"best_{a.mode}.pt")
            best_p = p

    print(f"\nBest test AUC {best:.4f} (checkpoint: best_{a.mode}.pt). Per-group at best epoch:")
    report(best_p, y, te.meta)


if __name__ == "__main__":
    main()
