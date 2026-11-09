| name | kind | #members | val AUC | test AUC [95% CI] | test AUC + flip TTA | ES-AUC race | worst-grp AUC |
|---|---|---|---|---|---|---|---|
| top5_by_val | ensemble | 5 | 0.8608 | 0.8330 [0.814, 0.852] | 0.8336 | 0.8212 | 0.8257 |
| diverse5 | ensemble | 5 | 0.8608 | 0.8330 [0.814, 0.852] | 0.8336 | 0.8212 | 0.8257 |
| greedy_by_val | ensemble | 8 | 0.8651 | 0.8301 [0.811, 0.848] | 0.8314 | 0.8143 | 0.8230 |
| all_pretrained_fusion | ensemble | 10 | 0.8559 | 0.8288 [0.810, 0.848] | 0.8293 | 0.8207 | 0.8233 |
| efficientnet_b0_pt_late | single | 1 | 0.8327 | 0.8271 | 0.8278 | 0.8149 | 0.8163 |
| top3_by_val | ensemble | 3 | 0.8571 | 0.8188 [0.801, 0.840] | 0.8220 | 0.8047 | 0.8109 |
| swin_tiny_pt_late | single | 1 | 0.8437 | 0.8069 | 0.8105 | 0.7789 | 0.7960 |
| densenet121_pt_late | single | 1 | 0.8370 | 0.8033 | 0.8104 | 0.7777 | 0.7938 |
| coatnet0_pt_late | single | 1 | 0.8354 | 0.7930 | 0.7959 | 0.7757 | 0.7866 |
| vit_small_pt_late | single | 1 | 0.8409 | 0.7898 | 0.7917 | 0.7681 | 0.7810 |
| maxvit_tiny_pt_early | single | 1 | 0.8197 | 0.7776 | 0.7879 | 0.7470 | 0.7679 |
| deit_small_pt_early | single | 1 | 0.8125 | 0.7682 | 0.7744 | 0.7281 | 0.7560 |
| resnet50_pt_late | single | 1 | 0.7962 | 0.7679 | 0.7790 | 0.7354 | 0.7409 |
| convnext_tiny_pt_late | single | 1 | 0.8217 | 0.7670 | 0.7672 | 0.7306 | 0.7573 |
| efficientnet_b0_pt_oct | single | 1 | 0.8094 | 0.7638 | 0.7709 | 0.7504 | 0.7529 |
| efficientnet_b0_pt_slo | single | 1 | 0.8062 | 0.7583 | 0.7622 | 0.7272 | 0.7231 |
| resnet50_pt_early | single | 1 | 0.7459 | 0.7472 | 0.7586 | 0.7428 | 0.7445 |
| resnet18_scratch_late | single | 1 | 0.7732 | 0.7172 | 0.7252 | 0.6958 | 0.7065 |

Ensemble members:
- all_pretrained_fusion: resnet50_pt_late, resnet50_pt_early, efficientnet_b0_pt_late, densenet121_pt_late, convnext_tiny_pt_late, vit_small_pt_late, deit_small_pt_early, swin_tiny_pt_late, coatnet0_pt_late, maxvit_tiny_pt_early
- top3_by_val: swin_tiny_pt_late, vit_small_pt_late, densenet121_pt_late
- top5_by_val: swin_tiny_pt_late, vit_small_pt_late, densenet121_pt_late, coatnet0_pt_late, efficientnet_b0_pt_late
- diverse5: efficientnet_b0_pt_late, swin_tiny_pt_late, densenet121_pt_late, vit_small_pt_late, coatnet0_pt_late
- greedy_by_val: swin_tiny_pt_late, densenet121_pt_late, maxvit_tiny_pt_early, deit_small_pt_early, vit_small_pt_late, efficientnet_b0_pt_late, convnext_tiny_pt_late, swin_tiny_pt_late
