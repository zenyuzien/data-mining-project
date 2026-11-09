# Literature review and next experiments

Written 2026-11-08 while `run_all.py` (13 experiments) runs in tmux session `dm`.
Source: web search result snippets only. I did NOT open the full papers, so benchmark numbers
and exact formulas below are unverified and must be checked before citing.

## 1. The dataset and task
- Our data is the **FairDomain** dataset (ECCV 2024, arXiv 2407.08813, Harvard Ophthalmology AI Lab).
  It pairs SLO fundus with en-face (OCT-derived) fundus images for glaucoma classification
  (and cup/rim segmentation). Built on Harvard-FairSeg and Harvard-FairVision data. License CC BY-NC-ND 4.0
  (no clinical/commercial use).
  The Hugging Face card says 10,000 samples per task; our copy has 5,000 (3,000 train / 2,000 test),
  so ours is likely a subset. Check the paper's splits before comparing numbers.
- Its theme is **fairness under domain shift**: the two image types are the two "domains"; methods are
  evaluated on both domain adaptation and generalization, with fairness across demographic groups.

## 2. Related work (to read in full)
| Paper | Why it matters |
|---|---|
| FairDomain (arXiv 2407.08813) | Our dataset; baseline results and the fairness-aware domain adaptation method |
| Harvard-FairVision (arXiv 2310.02492) | 30k subjects, SLO + 3D OCT; proposes Fair Identity Scaling (FIS: individual + group loss weighting) |
| Harvard-GF (arXiv 2306.09264, IEEE TMI) | Glaucoma fairness dataset; proposes Fair Identity Normalization (FIN) to equalize feature importance across identity groups |
| MultiFair | Fuses 3D OCT + 2D SLO on FairVision; reported (unverified) up to 7% AUC / 4% ES-AUC gain over best unimodal |
| FairEnc (arXiv 2605.04882) | Fair vision-language model for glaucoma on SLO; reports group-wise AUC, ES-AUC, worst-group AUC |
| Fair-Eye Net (arXiv 2601.18464) | Multimodal glaucoma pipeline with fairness constraint; numbers inconsistent across mirrors, treat as unconfirmed |
| RED-Sphere (arXiv 2607.10777) | Cross-population fundus domain generalization debiasing |
| FairCLIP / FairVLMed | Vision-language fairness on the same data family |

## 3. Metrics the field uses (we currently report only some)
- AUC overall, accuracy, **group-wise AUC**, **worst-group AUC** (we have these).
- **ES-AUC** (equity-scaled AUC): overall AUC scaled down by the between-group performance gap.
  We do NOT compute this yet. Exact formula must be taken from the FairVision/FairDomain paper.
- Possibly demographic-parity / equalized-odds gaps (check which the FairDomain paper uses).

## 4. Next experiments (priority order)
After the current 13-run sweep finishes, rank by val AUC and use the top 2-3 backbones for:

1. **Add ES-AUC to the evaluation** (code change, no training). Needed to compare with the literature.
2. **Domain-aware setup matching the paper**: treat SLO vs OCT-fundus as source/target domains
   (train on one, test on the other), plus our current joint fusion. Gives domain-shift numbers.
3. **Fairness methods on the best backbone**:
   - Group-reweighted loss (inverse group frequency, by race and gender).
   - Fair Identity Scaling (FIS) style loss weighting.
   - Fair Identity Normalization (FIN) style feature/BatchNorm per group.
   - Adversarial debiasing (gradient reversal on race).
4. **Better fusion**: cross-attention between SLO and OCT tokens (ViT/Swin), gated fusion,
   and late-fusion with a modality-dropout regularizer.
5. **Foundation-model fine-tuning**: RETFound (retinal MAE-ViT), DINOv2 / DINOv3 ViT, with linear probe
   then full fine-tune with layer-wise LR decay. Likely the biggest accuracy gain; needs weights download.
6. **Training recipe**: longer schedule with early stopping, stronger augmentation (rotation, elastic,
   RandAugment, mixup/cutmix), label smoothing, EMA weights, TTA.
7. **Robustness of conclusions**: 3-5 seeds for the top models, report mean +- std; bootstrap CIs on
   small groups (hispanic test n=83 is far too small for a stable AUC).
8. **Use tabular data carefully**: add age and `md` as input features (md is nearly the label, so run it
   only as a clearly-labelled upper bound, not as a fair comparison).
9. **Ensembling** the top models (average probabilities), and calibration check.

## 5. Open questions
- Does our 5,000-sample subset use the same split/protocol as the paper? (affects comparability)
- Is the "OCT fundus" here the en-face OCT image, matching the paper's second domain?
- Which fairness metrics does the task/assignment grade on?
