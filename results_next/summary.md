| # | name | family | test AUC [95% CI] | val AUC | ES-AUC race | ES-AUC gender | worst-grp AUC | other-domain AUC | s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | base_tab_age_md_UPPERBOUND | Tabular | 0.9576 [0.950, 0.965] | 0.9630 | 0.9379 | 0.9457 | 0.9406 | - | 1252 |
| 2 | base_gated_moddrop | Fusion | 0.8118 [0.793, 0.832] | 0.8416 | 0.8005 | 0.8034 | 0.8026 | - | 1254 |
| 3 | base_recipe | Recipe | 0.8109 [0.790, 0.831] | 0.8337 | 0.7900 | 0.8051 | 0.8055 | - | 1252 |
| 4 | base_recipe_s2 | Seeds | 0.8104 [0.789, 0.830] | 0.8356 | 0.7832 | 0.7910 | 0.7927 | - | 1250 |
| 5 | dinov2_small_late | Foundation | 0.8092 [0.788, 0.829] | 0.8373 | 0.7602 | 0.8065 | 0.7861 | - | 1290 |
| 6 | base_tab_age | Tabular | 0.8080 [0.786, 0.828] | 0.8399 | 0.7852 | 0.8068 | 0.8015 | - | 1364 |
| 7 | base_fair_fis_like | Fairness | 0.8062 [0.784, 0.826] | 0.8465 | 0.7788 | 0.8041 | 0.8003 | - | 1250 |
| 8 | base_fair_adv_race | Fairness | 0.8046 [0.782, 0.823] | 0.8500 | 0.7879 | 0.8036 | 0.8007 | - | 1253 |
| 9 | base_recipe_s1 | Seeds | 0.8029 [0.782, 0.823] | 0.8425 | 0.7829 | 0.8018 | 0.7974 | - | 1250 |
| 10 | vit_small_xattn | Fusion | 0.7949 [0.773, 0.815] | 0.8444 | 0.7725 | 0.7942 | 0.7864 | - | 1002 |
| 11 | base_xdom_slo2oct | Domain | 0.7937 [0.773, 0.812] | 0.7962 | 0.7592 | 0.7921 | 0.7859 | 0.6766 | 753 |
| 12 | base_fair_groupw | Fairness | 0.7868 [0.764, 0.808] | 0.8189 | 0.7599 | 0.7794 | 0.7798 | - | 1252 |
| 13 | base_xdom_oct2slo | Domain | 0.7855 [0.765, 0.804] | 0.8220 | 0.7575 | 0.7796 | 0.7786 | 0.7014 | 753 |

Ensemble of ['base_fair_adv_race', 'base_fair_fis_like', 'vit_small_xattn']: test AUC 0.8104, ES-AUC race 0.7910, worst-group AUC 0.8067

Seed variability of base_recipe (3 seeds): test AUC 0.8081 +- 0.0045

NOTE: base_tab_age_md_UPPERBOUND uses `md` (nearly the label) -- not a fair comparison.
