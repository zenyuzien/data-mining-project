| # | name | family | arch | test AUC [95% CI] | +flip TTA | val AUC | ES-AUC race | ES-AUC gender | worst-grp | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | abl_flipaug | A ablation | efficientnet_b0 | 0.8195 [0.802, 0.836] | 0.8242 | 0.8533 | 0.7847 | 0.8075 | 0.7994 | 668 |
| 2 | eb0_droppath | T1 regularise | efficientnet_b0 | 0.8173 [0.797, 0.839] | 0.8204 | 0.8521 | 0.7926 | 0.8121 | 0.8034 | 670 |
| 3 | abl_sweep1_repro | A ablation | efficientnet_b0 | 0.8112 [0.790, 0.830] | 0.8136 | 0.8444 | 0.7916 | 0.8050 | 0.8039 | 393 |
| 4 | eb0_gated_moddrop | T1 fusion | efficientnet_b0 | 0.8098 [0.791, 0.830] | 0.8142 | 0.8320 | 0.8011 | 0.7928 | 0.8031 | 670 |
| 5 | best_cv_f3 | T3 5-fold CV | efficientnet_b0 | 0.8081 [0.787, 0.829] | 0.8148 | 0.8198 | 0.7969 | 0.7827 | 0.7991 | 654 |
| 6 | abl_headlr1 | A ablation | efficientnet_b0 | 0.8071 [0.789, 0.829] | 0.8072 | 0.8280 | 0.7644 | 0.7988 | 0.8019 | 676 |
| 7 | regnety_016 | T2 backbone | regnety_016 | 0.8066 [0.785, 0.828] | 0.8122 | 0.8315 | 0.7725 | 0.8031 | 0.8026 | 968 |
| 8 | best_cv_f4 | T3 5-fold CV | efficientnet_b0 | 0.8059 [0.785, 0.824] | 0.8087 | 0.8270 | 0.7523 | 0.7866 | 0.7932 | 658 |
| 9 | best_cv_f2 | T3 5-fold CV | efficientnet_b0 | 0.8024 [0.780, 0.822] | 0.8082 | 0.8072 | 0.7828 | 0.7927 | 0.7934 | 658 |
| 10 | best_fair_adv | T4 fairness | efficientnet_b0 | 0.8017 [0.781, 0.821] | 0.8075 | 0.8470 | 0.7821 | 0.7954 | 0.7960 | 674 |
| 11 | eb0_mixup | T1 regularise | efficientnet_b0 | 0.7999 [0.780, 0.819] | 0.8050 | 0.8389 | 0.7645 | 0.7822 | 0.7863 | 675 |
| 12 | eb0_recipe_s2 | T1 seeds | efficientnet_b0 | 0.7998 [0.778, 0.821] | 0.8045 | 0.8345 | 0.7549 | 0.7952 | 0.7768 | 674 |
| 13 | eb0_recipe | T1 recipe | efficientnet_b0 | 0.7994 [0.779, 0.820] | 0.8024 | 0.8435 | 0.7736 | 0.7886 | 0.7821 | 672 |
| 14 | best_cv_f0 | T3 5-fold CV | efficientnet_b0 | 0.7970 [0.776, 0.820] | 0.8054 | 0.7836 | 0.7863 | 0.7759 | 0.7868 | 657 |
| 15 | effnet_b2 | T2 backbone | efficientnet_b2 | 0.7959 [0.778, 0.817] | 0.8061 | 0.8386 | 0.7652 | 0.7832 | 0.7833 | 1023 |
| 16 | eb0_recipe_s1 | T1 seeds | efficientnet_b0 | 0.7946 [0.774, 0.815] | 0.8011 | 0.8182 | 0.7466 | 0.7937 | 0.7824 | 669 |
| 17 | best_trainval_s2 | T3 train+val | efficientnet_b0 | 0.7939 [0.772, 0.817] | 0.8015 | - | 0.7615 | 0.7801 | 0.7815 | 736 |
| 18 | best_trainval_s1 | T3 train+val | efficientnet_b0 | 0.7917 [0.771, 0.812] | 0.7991 | - | 0.7125 | 0.7843 | 0.7546 | 735 |
| 19 | best_trainval_s0 | T3 train+val | efficientnet_b0 | 0.7913 [0.771, 0.813] | 0.7994 | - | 0.7526 | 0.7836 | 0.7777 | 739 |
| 20 | eb0_lowlr | T1 regularise | efficientnet_b0 | 0.7904 [0.767, 0.812] | 0.8035 | 0.8331 | 0.7788 | 0.7880 | 0.7834 | 672 |
| 21 | best_cv_f1 | T3 5-fold CV | efficientnet_b0 | 0.7885 [0.770, 0.810] | 0.7937 | 0.7876 | 0.7651 | 0.7782 | 0.7847 | 658 |
| 22 | abl_noema | A ablation | efficientnet_b0 | 0.7859 [0.765, 0.805] | 0.7924 | 0.8169 | 0.7412 | 0.7806 | 0.7605 | 641 |
| 23 | densenet121 | T2 backbone | densenet121 | 0.7806 [0.760, 0.801] | 0.7910 | 0.8279 | 0.7395 | 0.7805 | 0.7683 | 1179 |
| 24 | effnet_b3 | T2 backbone | efficientnet_b3 | 0.7787 [0.759, 0.799] | 0.7892 | 0.8209 | 0.7379 | 0.7751 | 0.7572 | 1324 |
| 25 | eb0_recipe_ep12 | T1 regularise | efficientnet_b0 | 0.7767 [0.757, 0.798] | 0.7869 | 0.8359 | 0.7598 | 0.7480 | 0.7661 | 338 |

### Ensembles (average of member probabilities)

| ensemble | #members | test AUC [95% CI] | +flip TTA | ES-AUC race | ES-AUC gender | worst-grp |
|---|---|---|---|---|---|---|
| grand_cv5+trainval3 | 8 | 0.8342 [0.816, 0.854] | 0.8354 | 0.8019 | 0.8261 | 0.8214 |
| cv5 | 5 | 0.8302 [0.812, 0.850] | 0.8309 | 0.8074 | 0.8086 | 0.8196 |
| tier12_top3 | 3 | 0.8229 [0.803, 0.844] | 0.8240 | 0.7993 | 0.8117 | 0.8155 |
| trainval3 | 3 | 0.8160 [0.796, 0.837] | 0.8206 | 0.7637 | 0.8078 | 0.8048 |

Seed noise of eb0_recipe (3 seeds): test AUC 0.7979 +- 0.0029; val AUC 0.8321 +- 0.0129

BEST config for Tier 3/4: eb0_droppath (val 0.8521) beat eb0_recipe mean 0.8321 + margin 0.0129
