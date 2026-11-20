# Decision log (date, decision, reason)
- Bytes beyond the DLC are stored as 0; DLC is part of the signature so nothing is ambiguous.
- Signature = (id, dlc, d0..d7); timestamp excluded.

# Decision log (01/11/2026, decision, reason)

Bytes beyond the DLC are stored as 0. A frame with DLC=2 has only two real bytes, but the model needs 8 feature columns. Because dlc is part of the signature, a frame [05 21] with DLC 2 is never confused with [05 21 00 00 00 00 00 00] with DLC 8.
Signature = (id, dlc, d0…d7). It leaves out the timestamp and the label.
Timestamp is excluded because it is different for every frame, so including it would make every frame unique and hide all the redundancy.
Label is excluded because the same signature can appear under two labels, for example a replayed normal frame inside an attack. The audit measures this (ρc, κ) on days 2-3, which only works if the signature ignores the label.

                  min            max
source                              
dos    1478198376.389 1478201209.059
fuzzy  1478195721.904 1478201209.059
gear   1478193190.057 1478201209.059
normal 1479121434.850 1479121941.287
rpm    1478191030.045 1478201209.059

data\raw\Fuzzy_dataset.csv:1546677:1478196832.714282,0105,6,eb,01,b7,00,98,02,R
# Decision log (20/11/2026, decision, reason)

Model settings frozen to match the team pilot. Logistic Regression: StandardScaler (fitted on the training side only) + LogisticRegression (C=1.0, lbfgs, max_iter=200, class_weight balanced). Decision Tree: gini, no depth limit, class_weight balanced. Random Forest: 50 trees, class_weight balanced_subsample, full bootstrap sample (max_samples left at its default), no depth limit. All models use random_state = seed and the same 10 features (id, dlc, d0..d7). The settings are identical for Technique A and Technique B and are stored in config.yaml.
Reason: the team agreed to use the same settings as the pilot script (Hafiz). This replaces an earlier choice of 28/10/2026 (Random Forest with 100 trees and max_samples 0.5) that came from a timing test on synthetic data. No model had been trained on real data when the change was made. Estimated time of one Random Forest fit on about 11.2 million training frames is about 15 minutes (synthetic estimate on Colab with 2 cores), to be checked on the first real fit.
