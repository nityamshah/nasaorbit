# classifier.py

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, precision_score,
                             recall_score, classification_report, confusion_matrix,
                             average_precision_score)
import numpy as np
from setup import run_pipeline
import pandas as pd

X, y, groups, record_names = run_pipeline()
X = np.array(X)
y = np.array(y)
groups = np.array(groups)
record_names = np.array(record_names)

print(f"Dataset: {len(y)} RHC records from {len(np.unique(groups))} patients")
print(f"Class 0 records: {sum(y == 0)} | Class 1 records: {sum(y == 1)}")
print(f"Feature vector size: {X.shape[1]}")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_preds, all_probs, all_true = [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=3,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42
    )
    model.fit(X_train, y_train)

    #preds = model.predict(X_test)
    threshold = 0.4
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    all_preds.extend(preds)
    all_probs.extend(probs)
    all_true.extend(y_test)

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_true  = np.array(all_true)

print("\n--- Performance ---")
print(f"Accuracy:          {accuracy_score(all_true, all_preds):.3f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(all_true, all_preds):.3f}")
print(f"F1 Score:          {f1_score(all_true, all_preds):.3f}")
print(f"ROC-AUC:           {roc_auc_score(all_true, all_probs):.3f}")
print(f"PR AUC:            {average_precision_score(all_true, all_probs)}")
print(f"Precision:         {precision_score(all_true, all_preds):.3f}")
print(f"Recall:            {recall_score(all_true, all_preds):.3f}")
print("\n", classification_report(all_true, all_preds, target_names=["No CDecomp", "CDecomp"]))
print(confusion_matrix(all_true, all_preds))
