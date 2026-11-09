| # | name | family | fusion | pretrained | test AUC | val AUC | acc | worst-grp AUC | time(s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | deit_small_pt_early | Transformer | early | True | 0.6256 | 0.5020 | 0.6250 | - | 11 |
| 2 | swin_tiny_pt_late | Transformer | late | True | 0.6146 | 0.7247 | 0.6094 | - | 12 |
| 3 | convnext_tiny_pt_late | Modern CNN | late | True | 0.5886 | 0.8907 | 0.5781 | - | 25 |
| 4 | efficientnet_b0_pt_late | CNN | late | True | 0.5686 | 0.6478 | 0.5781 | - | 12 |
| 5 | vit_small_pt_late | Transformer | late | True | 0.5485 | 0.6154 | 0.5938 | - | 12 |
| 6 | efficientnet_b0_pt_oct | Ablation | oct | True | 0.5325 | 0.3401 | 0.4531 | - | 10 |
| 7 | densenet121_pt_late | CNN | late | True | 0.5305 | 0.6235 | 0.5781 | - | 17 |
| 8 | resnet50_pt_early | CNN | early | True | 0.4935 | 0.5466 | 0.5781 | - | 16 |
| 9 | coatnet0_pt_late | Hybrid CNN+Attn | late | True | 0.4755 | 0.4251 | 0.4531 | - | 21 |
| 10 | resnet18_scratch_late | CNN | late | False | 0.4144 | 0.6073 | 0.4219 | - | 11 |
| 11 | resnet50_pt_late | CNN | late | True | 0.4144 | 0.6356 | 0.5781 | - | 19 |
| - | maxvit_tiny_pt_early | | | | **failed** timeout after 5s | | | | |
| - | efficientnet_b0_pt_slo | | | | **failed** timeout after 5s | | | | |
