import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully.")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())

print("\nUnique values:")
print(data.nunique())


# ============================================================
# 2. BASIC DATA EXPLORATION
# ============================================================

for col in [
    "email_types",
    "email_status",
    "email_criticality",
    "agent_effectivity",
    "agent_efficiency"
]:
    print(f"\n--- {col} ---")
    print(data[col].value_counts())


# ============================================================
# 3. CHECK CRITICALITY CONSISTENCY WITHIN THREADS
# ============================================================

criticality_per_thread = (
    data.groupby("thread_id")["email_criticality"]
    .nunique()
)

print("\nCriticality values per thread:")
print(criticality_per_thread.value_counts())


# ============================================================
# 4. CONVERT TIMESTAMP AND SORT
# ============================================================

data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    format="mixed",
    utc=True
)

data = data.sort_values(
    ["thread_id", "timestamp"]
)


# ============================================================
# 5. KEEP ONLY CUSTOMER EMAILS
# ============================================================

customer_emails = data[
    ~data["sender"].str.contains(
        "support@aetheros.com",
        case=False,
        na=False
    )
].copy()

print("\nCustomer emails:", len(customer_emails))


# ============================================================
# 6. CREATE ONE CASE PER THREAD
# ============================================================

model_data = (
    customer_emails
    .groupby("thread_id")
    .first()
    .reset_index()
)

print("\nModel data shape:", model_data.shape)

print(
    model_data[
        ["thread_id", "subject", "email_criticality"]
    ].head()
)


# ============================================================
# 7. CREATE TEXT FEATURE
# ============================================================

model_data["text"] = (
    model_data["subject"].fillna("")
    + " "
    + model_data["message_body"].fillna("")
)

print("\nCriticality distribution:")
print(model_data["email_criticality"].value_counts())


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X = model_data["text"]
y = model_data["email_criticality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining:", len(X_train))
print("Testing:", len(X_test))


# ============================================================
# 9. CREATE CRITICALITY MODEL PIPELINE
# ============================================================

criticality_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None
        )
    ),
    (
        "model",
        LogisticRegression(
            C=5,
            max_iter=1000
        )
    )
])


# ============================================================
# 10. CROSS-VALIDATION
# ============================================================

cv_scores = cross_val_score(
    criticality_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1_macro"
)

print("\n5-Fold CV Macro F1:")
print(cv_scores)

print("\nMean CV Macro F1:", cv_scores.mean())


# ============================================================
# 11. TRAIN FINAL CRITICALITY MODEL
# ============================================================

criticality_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 12. FINAL TEST PREDICTIONS
# ============================================================

y_pred = criticality_pipeline.predict(X_test)


# ============================================================
# 13. FINAL MODEL PERFORMANCE
# ============================================================

print("\n==============================")
print("FINAL CRITICALITY MODEL")
print("==============================")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["high", "medium", "low"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["high", "medium", "low"]
)

disp.plot()

plt.title("Email Criticality Confusion Matrix")
plt.show()


# ============================================================
# 15. SAVE CRITICALITY MODEL
# ============================================================

import joblib

joblib.dump(
    criticality_pipeline,
    "criticality_model.pkl"
)

print("\nCriticality model saved successfully!")