# Sweep 3 plan

Written 2026-11-08, while sweep 2 (`run_next.py`) is on experiment 10 of 13.
Still pending: `base_tab_age`, `base_tab_age_md_UPPERBOUND`, seeds `s1`/`s2`, top-3 ensemble.
Re-check those before launching (decision points marked **[WAIT]**).

## 1. What the results so far say
1. **Pretraining is the big win.** Scratch ResNet18 0.717 -> pretrained 0.77-0.83 test AUC.
2. **Both modalities help.** EfficientNet-B0: fusion 0.827 vs OCT-only 0.764 vs SLO-only 0.758.
3. **We are at a plateau and inside the noise.** Sweep 2 (Swin base) spans 0.785-0.812 with 95% CI about +-0.02.
   Gated fusion, DINOv2, FIS-like, adversarial: all indistinguishable from the plain recipe.
   Only group-reweighting looks worse (0.787).
4. **Severe overfitting.** Final train loss 0.004-0.015 (the 2,550 train images are memorised).
   Validation AUC peaks early for some models (Swin best epoch 5 of 15 in sweep 1).
   So the lever is regularisation / more data / variance reduction, not bigger models.
5. **Fairness gaps are small except hispanic.** Gender and race AUCs sit within ~0.03 of each other.
   Hispanic (n=83) is lower (0.70-0.77) but its CI is huge; it is not a reliable finding.
6. **Val and test rankings disagree** (Swin best val 0.844, EfficientNet best test 0.827), so single-run
   rankings are not trustworthy. Use seeds and validation-based decisions only.
7. **Cross-domain is poor** (SLO->OCT 0.68, OCT->SLO 0.70): the modalities carry different information,
   which is also why fusion helps.
8. **Cost:** EfficientNet-B0 ~22 s/epoch, Swin ~50 s/epoch, ConvNeXt ~75 s/epoch.

## 2. Hypotheses to test
- H1: EfficientNet-B0 + the new recipe (EMA, strong aug, label smoothing) beats both sweep-1 EfficientNet and Swin+recipe.
- H2: Stronger regularisation (mixup/cutmix, drop-path, fewer epochs) cuts the overfitting and gains >= 0.01 AUC.
- H3: Ensembling diverse models gives the largest cheap gain (typically +0.01-0.02 AUC with 2,000 test samples).
- H4: Training on train+val (3,000 vs 2,550 images) helps; a 5-fold ensemble does better still.
- H5: Fairness methods do not reliably help here; keep only one (adversarial or FIS-like) as a documented negative result.

## 3. Experiments, in priority order
Base = `efficientnet_b0`, late fusion, recipe, unless noted. Each run ~5-10 min.

**Tier 0 - no training (do first, ~15 min)**
- T0a. Compute test probabilities for the 13 sweep-1 checkpoints (`ckpt/*.pt`; sweep 1 did not save probs) and
  ensemble diverse subsets (EffNet + Swin + DenseNet + ViT + CoAtNet). Report AUC, ES-AUC, CI.
- T0b. Horizontal-flip test-time augmentation on the best models.
- Needs a small loader script that rebuilds each architecture from `run_all.py`'s config.

**Tier 1 - regularisation on EfficientNet-B0 (H1, H2)**
1. `eb0_recipe` (25 epochs) - reference.
2. `eb0_recipe_ep12` - shorter schedule.
3. `eb0_mixup` - mixup/cutmix alpha 0.2 (add to the worker's loss).
4. `eb0_droppath` - drop_path_rate 0.2, head dropout 0.5.
5. `eb0_lowlr` - lr 2e-4 and 5x lower body lr than head.
6. `eb0_gated_moddrop` - gated fusion + modality dropout (the sweep-2 nominal best).

**Tier 2 - backbone neighbours (cheap)**
7. `efficientnet_b2`, 8. `efficientnet_b3`, 9. `regnety_016`, 10. `densenet121` with the recipe.

**Tier 3 - data and variance (H4)**
11. Best config from Tiers 1-2 trained on **train+val** (fixed epoch count taken from the validation curve), 3 seeds.
12. **5-fold CV ensemble** on the 3,000 training images with the best config; average the 5 models on test.

**Tier 4 - fairness and domain (H5), only if time allows**
13. Best fairness method (adversarial race or FIS-like) on the Tier-3 config; report ES-AUC with CIs.
14. Domain adaptation on top (e.g. SLO-trained + OCT feature alignment) - only if the course/task needs it.

**Skip / deprioritise:** group-reweighted loss (worse), cross-attention ViT (0.795, 2x cost), ConvNeXt (slow, no gain),
more foundation models (no gain over plain recipe; RETFound weights need gated access).

## 4. Decision rules (avoid fooling ourselves)
- Choose configs on **validation AUC averaged over >= 3 seeds**, never on test.
- Treat differences below the seed standard deviation, or below ~0.01, as ties; prefer the cheaper/simpler one.
- Report test AUC with bootstrap CI and the per-group table once, for the final chosen config and the ensemble.
- Keep the UPPERBOUND (`md`) run out of every comparison.

## 5. Budget
Tier 0: ~15 min. Tiers 1-2: 10 runs x ~8 min = ~1.5 h. Tier 3: 3 seeds + 5 folds = 8 runs x ~8 min = ~1 h.
Tier 4: ~0.5 h. Total about 3 h on the P100.

## 6. Implementation needed
- `run_sweep3.py`: reuse `run_next.py`'s worker (add mixup/cutmix, drop-path flag, train+val mode, k-fold mode, TTA).
- `ensemble_ckpts.py` for Tier 0 (rebuild sweep-1 models from checkpoints, save probs).
- Smoke-test everything before launching, as before.

## 7. Open points [WAIT]
- Result of `base_tab_age`: if age helps measurably (> seed noise), add it to the final config.
- Seed std from `base_recipe`, `_s1`, `_s2`: sets the "tie" threshold in section 4.
- Whether the task grade is on AUC alone or also on fairness metrics (changes how much of Tier 4 to run).
