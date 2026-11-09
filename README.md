# Glaucoma detection from SLO + OCT fundus images (FairDomain data)

Two-branch deep models (CNN / transformer / hybrid) that classify glaucoma from a pair of 224x224 images (SLO fundus +
OCT-derived fundus), evaluated overall and per demographic group. The write-up is in `report/report.pdf`.

## Layout
- `dataset.py`, `train.py` - data loader and the first baseline (two-branch ResNet18).
- `run_all.py` - sweep 1: 13 backbones/fusion settings (results in `results/`, logs in `logs/`).
- `run_next.py` - sweep 2: stronger recipe on Swin-Tiny, fairness losses, cross-domain, tabular (`results_next/`).
- `ensemble_ckpts.py` - Tier 0: ensembles of the sweep-1 checkpoints (`results_t0/`).
- `run_sweep3.py` - sweep 3 + ablations: EfficientNet-B0 regularisation, 5-fold / train+val, ensembles (`results_sweep3/`).
- `report/` - LaTeX source (`report.tex`), PDF, and `make_assets.py`, which builds every table and figure from the raw results.
- `WORKLOG.md`, `LITERATURE_AND_NEXT_RUNS.md`, `SWEEP3_PLAN.md` - running log, literature notes, sweep-3 plan.

## Not included
- The dataset (FairDomain, CC BY-NC-ND 4.0, ~500 MB): place the extracted `fairdomain/` folder under `extracted/<any name>/`.
- Model checkpoints (several GB) and the Python environment. Test-set probabilities of each run are kept in `*/probs/*.npz`.

## Reproduce
```
python -m venv venv && ./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
./venv/bin/pip install numpy pandas scikit-learn matplotlib timm tqdm
./venv/bin/python run_all.py            # sweep 1 (resumable, one subprocess per experiment)
./venv/bin/python run_next.py           # sweep 2
./venv/bin/python ensemble_ckpts.py     # tier 0 (needs ckpt/ from sweep 1)
./venv/bin/python run_sweep3.py         # sweep 3
./venv/bin/python run_sweep3.py --only abl_noema abl_headlr1 abl_flipaug abl_sweep1_repro
./venv/bin/python report/make_assets.py && cd report && tectonic report.tex
```
Hardware used: one Tesla P100 (16 GB), PyTorch 2.14 (cu126 build).
