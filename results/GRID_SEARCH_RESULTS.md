# Grid Search CV: Results & Analysis

## 1. Experimental Overview
An exhaustive Grid Search was conducted over the `vowel` and `splice` datasets for both Random Forest (RF) and Support Vector Machine (SVM) models. The evaluation metric was the mean accuracy across the 10 predefined stratified folds.

## 2. Analysis by Dataset

### SPLICE Dataset (Imbalanced DNA sequences)
* **Maximum Accuracy Achieved:** 96.90% (Tie between SVM and RF)
* **SVM Optimal:** `C=0.1`, `kernel=poly`, `degree=5`, `class_weight=balanced` (0.62s/fold)
* **RF Optimal:** `n_estimators=200`, `min_samples_leaf=2`, `criterion=entropy` (0.21s/fold)

**Conclusions for Splice:**
1. **Imbalance Handling Worked:** As hypothesized during the EDA, the dataset's imbalance (~52% class N) required intervention. The SVM only reached its peak when `class_weight=balanced` was applied, heavily penalizing errors in the minority classes.
2. **High-dimensional complexity:** The SVM preferred a 5th-degree polynomial kernel over a linear or RBF one. Since our one-hot encoding expanded the dataset to 287 columns, a high-degree polynomial helps capture complex interactions between distant nucleotides. However, to avoid overfitting this massive space, it selected a high regularization penalty (`C=0.1`).
3. **The 96.90% Ceiling:** Both models hitting the exact same maximum accuracy strongly implies we have reached the theoretical limit (Bayes Error Rate) of this feature representation. The remaining ~3.1% is likely inherent noise or overlapping ambiguous sequences.

### VOWEL Dataset (Speech recognition)
* **Maximum Accuracy Achieved:** 99.09% (SVM) vs 97.98% (RF)
* **SVM Optimal:** `C=3.0`, `kernel=poly`, `degree=5`, `class_weight=None` (0.008s/fold)
* **RF Optimal:** `n_estimators=200`, `min_samples_leaf=1`, `criterion=entropy` (0.18s/fold)

**Conclusions for Vowel:**
1. **Validation of the Ceiling Hypothesis:** The EDA logged a hypothesis that Vowel would exhibit a ceiling effect due to repeated speakers across the train/test splits. A 99.09% accuracy confirms this: the dataset is almost perfectly separable because test instances are near-duplicates of train instances.
2. **SVM Geometric Superiority:** SVM outperformed RF by over 1%. Vowel consists of continuous numerical frequencies (formants). SVM's ability to draw smooth, continuous geometric boundaries in the numeric space fits the physics of sound better than the orthogonal, step-wise splits of a Random Forest decision tree.
3. **Hard Margins & Pure Leaves:** Because the data is so clean and separable, both models minimized regularization. The SVM chose `C=3.0` (harder margin, trusting the training data), and the RF chose `min_samples_leaf=1` (fully grown trees with no pruning).

---

## 3. Phase 2: Fine-Tuning Grids
Based on the broad search, we can now define micro-grids to squeeze out the final decimals of performance. We will center the new searches around the winning parameters.

### Fine-Tuning SVM
* **Splice:** The model loved `poly` with heavy regularization (`C=0.1`) and `balanced` weights. We will explore variations of `C` and higher/lower polynomial degrees.
* **Vowel:** The model favored `poly` with low regularization (`C=3.0`). We will push `C` higher.

```python
# Fine-tuning Grid for SVM
GRIDS_FINETUNE_SVM = {
    "splice": {
        "kernel": ["poly"],
        "class_weight": ["balanced"],
        "C": [0.01, 0.05, 0.1, 0.5],  # Zooming in around 0.1
        "degree": [4, 5, 6],  # Zooming in around 5
        "coef0": [0.5, 1.0],  # Exploring higher shifts
        "gamma": ["scale"],
    },
    "vowel": {
        "kernel": ["poly"],
        "class_weight": ["None"],
        "C": [2.0, 3.0, 4.0, 5.0],  # Zooming in around 3.0
        "degree": [4, 5, 6],
        "coef0": [-0.5, -1.0],
        "gamma": ["scale"],
    },
}
```
### Fine-Tuning Random Forest

* **Both Datasets:** criterion=entropy and max_depth=None are clear winners. We will test higher tree counts to see if variance stabilizes further, and tweak leaf sizes.
```python
# Fine-tuning Grid for RF
GRIDS_FINETUNE_RF = {
    "splice": {
        "criterion": ["entropy"],
        "max_depth": ["None"],
        "n_estimators": [150, 200, 250, 400],  # Pushing ensemble size
        "min_samples_leaf": [2, 3, 4],  # Splice needs slight regularization
        "max_features": ["sqrt", "log2"],  # Testing narrower feature subsets
    },
    "vowel": {
        "criterion": ["entropy"],
        "max_depth": ["None"],
        "n_estimators": [150, 200, 250, 400],
        "min_samples_leaf": [1, 2],  # Vowel prefers unpruned trees
        "max_features": ["sqrt", "log2"],
    },
}
```
