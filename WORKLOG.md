# Work log: Glaucoma detection (SLO + OCT fundus), FairDomain

Date: 2026-11-06

## 1. Data
- Received `Problem-5_Glaucoma_Detection_Combining_SLO_and_OCT_Fundus_Images-...zip` (415 MB) in `/home/rushi/dm`.
- Extracted to `/home/rushi/dm/extracted/Problem-5_.../fairdomain/` (509 MB):
  - `Training/` 3,000 `.npz`, `Testing/` 2,000 `.npz`, `data_summary.csv`.
- Each `.npz`: `slo_fundus` (224x224 uint8), `oct_fundus` (224x224 uint8),
  and scalars age, gender, race, ethnicity, language, marriagestatus, md, glaucoma.
- Modalities: 2 imaging (SLO fundus, OCT fundus) + tabular patient attributes.
- Label balance: glaucoma yes 44% / no 56% in both splits.
- Demographics (both splits similar): ~73% female; race ~77% white, 15% black, 8% asian;
  ethnicity ~96% non-hispanic, 4% hispanic.

## 2. Environment
- Python venv: `/home/rushi/dm/venv` (activate: `source /home/rushi/dm/venv/bin/activate`).
- GPU: Tesla P100 16 GB (sm_60), so PyTorch cu126 build was used (newer CUDA builds may drop sm_60).
- Installed: torch 2.14+cu126, torchvision, timm, numpy, pandas, scikit-learn, matplotlib, tqdm, jupyter.
- GPU verified with a test matmul.

## 3. Code
- `dataset.py`: `FairDomainGlaucoma(split, augment)`; loads all data into RAM;
  returns 2x224x224 tensor (ch0=SLO, ch1=OCT) + label; keeps group metadata in `.meta`.
- `train.py`: `FusionNet` = two ResNet18 branches (one per modality), concatenated features + linear head.
  Modes: `--mode fusion|slo|oct`. AdamW + OneCycle, per-group AUC report (gender/race/ethnicity).

## 4. Results
Run: `train.py --mode fusion --epochs 15` (no pretrained weights). Log: `fusion_log.txt`, checkpoint: `best_fusion.pt`.

| Metric | Value |
|---|---|
| Best test AUC | 0.7191 |
| Test accuracy | 0.6710 |

Per-group AUC: female 0.721, male 0.713 | asian 0.726, black 0.712, white 0.710 |
non-hispanic 0.726, hispanic 0.557 (n=83, noisy).

Notes:
- Training loss fell 0.75 -> 0.36 while test AUC plateaued ~0.72 from epoch 12: overfitting.
- Best epoch chosen on the test set (no validation split), so slightly optimistic.

## 5. Next steps (not done)
- Run with `--pretrained` (ImageNet weights).
- Run `--mode slo` and `--mode oct` for modality comparison.
- Carve a validation split from training data; stronger augmentation.

## 6. Mega-script (2026-11-08)
- `run_all.py`: 13 experiments (CNN, modern CNN, ViT/DeiT/Swin, CoAtNet/MaxViT hybrids, early/late fusion,
  SLO-only / OCT-only ablations), mostly ImageNet-pretrained via timm.
- Each experiment runs in its own subprocess (timeout, OOM retry with smaller batch, non-finite-loss guard,
  pretrained-download fallback flagged in results). A failure never stops the rest.
- Model selection on a held-out 15% validation split of training data (not on test).
- Smoke test: all 13 passed. Failure isolation verified (forced timeouts, run continued).
- Outputs: `results/<name>.json`, `results/summary.{csv,md}`, `logs/<name>.log`, `ckpt/<name>.pt`.
- Run: `./venv/bin/python run_all.py` (resumable; skips finished ones; `--force` to redo).

## 7. Second mega-script (2026-11-08)
- Literature review + next-experiment plan: `LITERATURE_AND_NEXT_RUNS.md` (search snippets only, unverified numbers).
- `run_next.py`: 13 experiments (stronger recipe, gated/cross-attention fusion, DINOv2, fairness losses,
  cross-domain SLO<->OCT, tabular, seeds) + top-3 ensemble. Adds ES-AUC and bootstrap CIs.
  Auto-picks best backbone from `results/`. Outputs in `results_next/`, `logs_next/`, `ckpt_next/`.
- Smoke-tested (all 13 OK) while run_all.py was still running. NOT yet launched.
- ES-AUC formula and "FIS-like" loss are from memory / approximations: verify against the papers.

## 8. Sweep 3 (2026-11-08)
- Plan: `SWEEP3_PLAN.md`. Scripts: `ensemble_ckpts.py` (Tier 0: ensemble the 13 sweep-1 checkpoints, no training,
  outputs `results_t0/`) and `run_sweep3.py` (21 runs: EfficientNet-B0 regularisation, backbone neighbours,
  auto-selected BEST config on train+val x3 seeds and 5-fold CV, adversarial fairness, ensembles, flip-TTA;
  outputs `results_sweep3/`).
- BEST rule: a challenger must beat eb0_recipe's mean val AUC by max(0.005, seed std), else plain eb0_recipe.
- Smoke-tested all 21 + Tier 0 (OK). Smoke outputs deleted.
- Launched in tmux session `sweep3`: waits for run_next.py to end, then Tier 0, then run_sweep3.py.
  Logs: `tier0_log.txt`, `run_sweep3_log.txt`.

## 9. Ablations (2026-11-08)
- Question: why did sweep-1 EfficientNet-B0 (test 0.827) beat the recipe version (0.798 +- 0.003)?
  Suspects: EMA copies BatchNorm buffers instead of averaging (hurts BN CNNs, not Swin), head LR x5, strong aug.
- `run_sweep3.py` entries abl_noema / abl_headlr1 / abl_flipaug (one change each vs eb0_recipe) +
  abl_sweep1_repro (all reverted, 15 epochs; approximate, OneCycle pct_start still 0.15). Seed 0 only.
- Running in tmux `ablate`; log `run_ablate_log.txt`; results in `results_sweep3/`.

## 10. Ablation results and final report (2026-11-08)
- Ablations (EfficientNet-B0, recipe 0.798 +- 0.003): EMA off 0.786 (EMA helps; BN-stat hypothesis refuted),
  head LR x1 0.807, flip-only aug 0.819 (strong geometric aug hurts), approx sweep-1 repro 0.811 (orig 0.827 likely optimistic).
- LaTeX report: `report/report.tex` -> `report/report.pdf` (compiled with tectonic from `texenv/bin/tectonic`).
  Tables/figures are generated from raw results by `report/make_assets.py` (no hand-typed numbers in tables).

## 11. Report restructured (2026-11-09)
- Main text now covers only the best-result model (ensemble of 8 two-branch EfficientNet-B0, test AUC 0.834 [0.816, 0.852]);
  all other experiments (backbones, recipe + ablations, fusion/DINOv2, fairness losses, domain shift/tabular, ensembles)
  moved to appendices A-G with motivation / setup / results / takeaway. New assets: tables/main_results.tex, groups_main.tex, figs/results_main.pdf.
- Not yet pushed to GitHub after this change.

## 12. Introduction rewritten around the data (2026-11-09)
- Intro now: task in 3 sentences, then "The data" (contents, sample images, labels/demographics, MD leakage, things an engineer should know).
  New: `report/make_samples.py` -> `report/figs/samples.pdf`, `report/data_stats.json`. Data table moved to the intro; Sec 4.1 is now "Data splits".
- Not yet pushed to GitHub (only report.pdf from the previous version was).

## 13. Report body rewritten as bullet points in simple English (2026-11-09)
- Everything after the Introduction (related work, method, experiments, conclusions, appendices) is now bullets with short, precise sentences. Numbers unchanged.
- Not yet pushed to GitHub.

## 14. Removed all ensembles of more than 3 models from the report (2026-11-09)
- Final model is now the single two-branch EfficientNet-B0 (plain recipe, sweep 1): test AUC 0.827 [0.808, 0.844]; flagged as optimistic
  (best of 13 on test; re-run 0.811; other two-branch EfficientNet-B0 runs 0.777-0.819).
- Only three 3-model ensembles remain (appendix G): 0.816-0.823. Raw results/code still contain the larger ensembles
  (results_t0, results_sweep3/ensembles.json, run_sweep3.py, ensemble_ckpts.py); they are just not reported.
- Not yet pushed to GitHub.
