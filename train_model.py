"""
Model Training Script
Trains and compares multiple ML models for emotion detection
Saves the best model as emotion_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("typing_data.csv")
print(f"Dataset loaded: {len(df)} rows, {df['emotion'].nunique()} classes")

FEATURES = [
    "typing_speed_wpm", "keypress_duration_ms", "pause_duration_ms",
    "error_rate", "backspace_count", "avg_word_length",
    "sentence_length", "exclamation_freq"
]

X = df[FEATURES]
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Define Models ───────────────────────────────────────────────────────────
models = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
}

# ── Train & Compare ─────────────────────────────────────────────────────────
results = {}
best_model_name = None
best_accuracy = 0
best_model = None

print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    results[name] = {
        "test_accuracy"  : round(float(acc), 4),
        "cv_mean"        : round(float(cv_scores.mean()), 4),
        "cv_std"         : round(float(cv_scores.std()), 4),
    }

    print(f"\n{name}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  CV Mean       : {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = pipeline

print("\n" + "="*55)
print(f"  🏆 BEST MODEL: {best_model_name} ({best_accuracy*100:.2f}%)")
print("="*55)

# ── Detailed Report for Best Model ─────────────────────────────────────────
y_pred_best = best_model.predict(X_test)
print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best))

# ── Feature Importance (for tree-based models) ──────────────────────────────
feature_importance = {}
clf = best_model.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    for feat, imp in zip(FEATURES, importances):
        feature_importance[feat] = round(float(imp), 4)
    print("\nFeature Importances:")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<25} {imp:.4f}")

# ── Save Everything ─────────────────────────────────────────────────────────
joblib.dump(best_model, "emotion_model.pkl")
print(f"\n✅ Model saved: emotion_model.pkl")

model_info = {
    "best_model"        : best_model_name,
    "best_accuracy"     : round(float(best_accuracy), 4),
    "features"          : FEATURES,
    "classes"           : list(best_model.classes_),
    "comparison"        : results,
    "feature_importance": feature_importance,
}
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("✅ Model info saved: model_info.json")