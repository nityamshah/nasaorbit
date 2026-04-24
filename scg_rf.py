from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from setup_data import run_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score


X, y, groups = run_pipeline()
X = np.array(X)
y = np.array(y)
groups = np.array(groups)

# print("Shape:", X.shape)
# print("Labels:", Counter(y))

# Group K Fold ensures that same data is not in train and test split
gkf = GroupKFold()

all_preds = []
all_probs = []
all_true = []

for fold, (train_index, test_index) in enumerate (gkf.split(X, y, groups)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #RF
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    all_preds.extend(preds)
    all_probs.extend(probs)
    all_true.extend(y_test)

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_true = np.array(all_true)

acc = accuracy_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds)
auc = roc_auc_score(all_true, all_probs)
bal_acc = balanced_accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds)
recall = recall_score(all_true, all_preds)

print("\n===== FINAL MODEL PERFORMANCE =====")
print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")