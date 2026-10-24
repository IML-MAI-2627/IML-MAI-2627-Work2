# splice: sensitivity to the duplicate rows of the training folds

Not an official result: the folds may not be modified (assignment, Section 1.1). `full` = as in grid v2; `dedup` = `Preprocessor(drop_duplicate_rows=True)`, exact copies (same sequence and class) removed from the training fold only. Storage = % of the original training fold kept; final case base = kNN case base after the test phase.

Check: the `full` run reproduces the per-fold accuracies of grid v2 for every model.

Rows removed per fold: 0: 142, 1: 159, 2: 157, 3: 159, 4: 159, 5: 153, 6: 153, 7: 157, 8: 147, 9: 153 (range 142-159, 5.0-5.5% of the training fold).

## Mean over the 10 folds

| model | variant | accuracy | std | macro_f1 | storage | final_case_base | time |
|---|---|---|---|---|---|---|---|
| kNN | full | 0.8439 | 0.0203 | 0.8407 | 100.0000 | 89.7108 | 0.1097 |
| SVM | full | 0.9690 | 0.0079 | 0.9650 | 100.0000 | nan | 1.2931 |
| RF | full | 0.9684 | 0.0092 | 0.9643 | 100.0000 | nan | 1.0329 |
| kNN+fcnn | full | 0.8530 | 0.0193 | 0.8414 | 41.8320 | 30.7174 | 2.0594 |
| kNN+eenth | full | 0.8166 | 0.0198 | 0.8160 | 82.5531 | 71.7971 | 0.6906 |
| kNN+drop | full | 0.8482 | 0.0188 | 0.8297 | 29.8955 | 23.2950 | 17.2618 |
| kNN+del | full | 0.8492 | 0.0203 | 0.8347 | 42.9364 | 33.6016 | 24.4059 |
| kNN | dedup | 0.8608 | 0.0190 | 0.8564 | 94.6395 | 84.5663 | 0.1100 |
| SVM | dedup | 0.9690 | 0.0079 | 0.9650 | 94.6395 | nan | 1.2088 |
| RF | dedup | 0.9662 | 0.0072 | 0.9620 | 94.6395 | nan | 1.0134 |
| kNN+fcnn | dedup | 0.8533 | 0.0190 | 0.8415 | 41.8286 | 30.7174 | 1.8867 |
| kNN+eenth | dedup | 0.8398 | 0.0204 | 0.8379 | 78.8993 | 68.9828 | 0.6457 |
| kNN+drop | dedup | 0.8505 | 0.0215 | 0.8352 | 29.0596 | 22.4520 | 15.1502 |
| kNN+del | dedup | 0.8408 | 0.0290 | 0.8260 | 41.5780 | 32.0688 | 20.8480 |

## Full vs dedup (Wilcoxon exact, Holm over all rows)

| model | metric | full | dedup | difference | folds_changed | p_raw | p_holm | significant |
|---|---|---|---|---|---|---|---|---|
| kNN | accuracy | 0.8439 | 0.8608 | 0.0169 | 10 | 0.0098 | 0.1172 | False |
| SVM | accuracy | 0.9690 | 0.9690 | 0.0000 | 0 | 1.0000 | 1.0000 | False |
| RF | accuracy | 0.9684 | 0.9662 | -0.0022 | 8 | 0.1562 | 1.0000 | False |
| kNN+fcnn | accuracy | 0.8530 | 0.8533 | 0.0003155 | 1 | 1.0000 | 1.0000 | False |
| kNN+eenth | accuracy | 0.8166 | 0.8398 | 0.0232 | 9 | 0.0039 | 0.0547 | False |
| kNN+drop | accuracy | 0.8482 | 0.8505 | 0.0022 | 10 | 0.6406 | 1.0000 | False |
| kNN+del | accuracy | 0.8492 | 0.8408 | -0.0084 | 10 | 0.2031 | 1.0000 | False |
| kNN | macro_f1 | 0.8407 | 0.8564 | 0.0157 | 10 | 0.0098 | 0.1172 | False |
| SVM | macro_f1 | 0.9650 | 0.9650 | 0.0000 | 0 | 1.0000 | 1.0000 | False |
| RF | macro_f1 | 0.9643 | 0.9620 | -0.0023 | 10 | 0.2324 | 1.0000 | False |
| kNN+fcnn | macro_f1 | 0.8414 | 0.8415 | 0.0001113 | 1 | 1.0000 | 1.0000 | False |
| kNN+eenth | macro_f1 | 0.8160 | 0.8379 | 0.0218 | 10 | 0.0039 | 0.0547 | False |
| kNN+drop | macro_f1 | 0.8297 | 0.8352 | 0.0054 | 10 | 0.4316 | 1.0000 | False |
| kNN+del | macro_f1 | 0.8347 | 0.8260 | -0.0087 | 10 | 0.3223 | 1.0000 | False |

## The grid's tests on each variant

| variant | question | test | p_value | significant | post_hoc |
|---|---|---|---|---|---|
| full | kNN vs SVM vs RF | Friedman + Iman-Davenport | 3.815e-06 | True | Nemenyi (CD=1.05): not different from best: SVM, RF |
| full | kNN vs SVM vs RF: kNN vs SVM | Wilcoxon signed-ranks + Holm | 0.0039 | True | raw p=0.001953 |
| full | kNN vs SVM vs RF: kNN vs RF | Wilcoxon signed-ranks + Holm | 0.0039 | True | raw p=0.001953 |
| full | kNN reduction | Friedman + Iman-Davenport | 0.0003179 | True | Bonferroni-Dunn vs none (CD=1.77): kNN+eenth worse |
| dedup | kNN vs SVM vs RF | Friedman + Iman-Davenport | 7.943e-07 | True | Nemenyi (CD=1.05): not different from best: SVM, RF |
| dedup | kNN vs SVM vs RF: kNN vs SVM | Wilcoxon signed-ranks + Holm | 0.0039 | True | raw p=0.001953 |
| dedup | kNN vs SVM vs RF: kNN vs RF | Wilcoxon signed-ranks + Holm | 0.0039 | True | raw p=0.001953 |
| dedup | kNN reduction | Friedman + Iman-Davenport | 0.0249 | True | Bonferroni-Dunn vs none (CD=1.77): kNN+eenth worse |

## Free answers of the best kNN

Test rows with an identical copy in their training fold: 294 of 3190 (9.2%). With `full`, the best kNN is right on 289 of them; on the other rows its accuracy is 0.8298. Overall accuracy 0.8439; if the twin rows were answered like the rest it would be 0.8298 (+0.0141).

| variant | fold | twin rows | accuracy on twins | rest rows | accuracy on the rest | overall |
|---|---|---|---|---|---|---|
| full | 0 | 41 | 0.9756 | 281 | 0.7972 | 0.8199 |
| full | 1 | 25 | 0.9600 | 293 | 0.8498 | 0.8585 |
| full | 2 | 27 | 1.0000 | 291 | 0.8625 | 0.8742 |
| full | 3 | 24 | 1.0000 | 294 | 0.8435 | 0.8553 |
| full | 4 | 25 | 1.0000 | 295 | 0.8339 | 0.8469 |
| full | 5 | 29 | 0.9655 | 290 | 0.8103 | 0.8245 |
| full | 6 | 31 | 1.0000 | 290 | 0.8517 | 0.8660 |
| full | 7 | 24 | 1.0000 | 294 | 0.8061 | 0.8208 |
| full | 8 | 37 | 0.9730 | 280 | 0.8036 | 0.8233 |
| full | 9 | 31 | 0.9677 | 288 | 0.8368 | 0.8495 |
| full | total | 294 | 0.9830 | 2896 | 0.8298 | 0.8439 |
| dedup | 0 | 41 | 0.9512 | 281 | 0.8221 | 0.8385 |
| dedup | 1 | 25 | 0.9600 | 293 | 0.8771 | 0.8836 |
| dedup | 2 | 27 | 1.0000 | 291 | 0.8591 | 0.8711 |
| dedup | 3 | 24 | 0.9167 | 294 | 0.8469 | 0.8522 |
| dedup | 4 | 25 | 1.0000 | 295 | 0.8475 | 0.8594 |
| dedup | 5 | 29 | 0.9310 | 290 | 0.8310 | 0.8401 |
| dedup | 6 | 31 | 1.0000 | 290 | 0.8793 | 0.8910 |
| dedup | 7 | 24 | 1.0000 | 294 | 0.8299 | 0.8428 |
| dedup | 8 | 37 | 0.9730 | 280 | 0.8357 | 0.8517 |
| dedup | 9 | 31 | 0.9677 | 288 | 0.8681 | 0.8777 |
| dedup | total | 294 | 0.9694 | 2896 | 0.8498 | 0.8608 |
