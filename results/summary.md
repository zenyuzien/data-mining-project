| # | name | family | fusion | pretrained | test AUC | val AUC | acc | worst-grp AUC | time(s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | efficientnet_b0_pt_late | CNN | late | True | 0.8271 | 0.8327 | 0.7635 | 0.8163 | 324 |
| 2 | swin_tiny_pt_late | Transformer | late | True | 0.8069 | 0.8437 | 0.7385 | 0.7960 | 761 |
| 3 | densenet121_pt_late | CNN | late | True | 0.8033 | 0.8370 | 0.7400 | 0.7938 | 534 |
| 4 | coatnet0_pt_late | Hybrid CNN+Attn | late | True | 0.7930 | 0.8354 | 0.7155 | 0.7866 | 957 |
| 5 | vit_small_pt_late | Transformer | late | True | 0.7898 | 0.8409 | 0.7240 | 0.7810 | 587 |
| 6 | maxvit_tiny_pt_early | Hybrid CNN+Attn | early | True | 0.7776 | 0.8197 | 0.7145 | 0.7679 | 688 |
| 7 | deit_small_pt_early | Transformer | early | True | 0.7682 | 0.8125 | 0.7055 | 0.7560 | 308 |
| 8 | resnet50_pt_late | CNN | late | True | 0.7679 | 0.7962 | 0.7025 | 0.7409 | 564 |
| 9 | convnext_tiny_pt_late | Modern CNN | late | True | 0.7670 | 0.8217 | 0.6805 | 0.7573 | 1109 |
| 10 | efficientnet_b0_pt_oct | Ablation | oct | True | 0.7638 | 0.8094 | 0.6835 | 0.7529 | 178 |
| 11 | efficientnet_b0_pt_slo | Ablation | slo | True | 0.7583 | 0.8062 | 0.6735 | 0.7231 | 178 |
| 12 | resnet50_pt_early | CNN | early | True | 0.7472 | 0.7459 | 0.6880 | 0.7445 | 285 |
| 13 | resnet18_scratch_late | CNN | late | False | 0.7172 | 0.7732 | 0.6665 | 0.7065 | 167 |
