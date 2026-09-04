import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sqlite3

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

print("\nFirst 5 rows:")
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

print(
    "\nMean CV Macro F1:",
    cv_scores.mean()
)


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
# 13. FINAL CRITICALITY MODEL PERFORMANCE
# ============================================================

print("\n==============================")
print("FINAL CRITICALITY MODEL")
print("==============================")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 14. SAVE CRITICALITY MODEL
# ============================================================

joblib.dump(
    criticality_pipeline,
    "criticality_model.pkl"
)

print("\nCriticality model saved successfully!")


# ============================================================
# 15. OPERATIONAL URGENCY
# ============================================================

def calculate_priority(row):

    text = row["text"].lower()

    # --------------------------------------------------------
    # Base score from ML-predicted criticality
    # --------------------------------------------------------

    if row["email_criticality"] == "high":
        score = 60

    elif row["email_criticality"] == "medium":
        score = 35

    else:
        score = 15


    # --------------------------------------------------------
    # Operational urgency signals
    # --------------------------------------------------------

    urgency_score = 0

    # Production environment
    if "production" in text:
        urgency_score += 10

    # Service/API outage
    if any(
        phrase in text
        for phrase in [
            "production outage",
            "service outage",
            "api is down",
            "api down",
            "service is down",
            "service down",
            "system is down",
            "system down",
            "completely down",
            "outage"
        ]
    ):
        urgency_score += 15

    # Unreachable / inaccessible system
    if any(
        phrase in text
        for phrase in [
            "unreachable",
            "cannot access",
            "unable to access",
            "not accessible"
        ]
    ):
        urgency_score += 10

    # Deployment / infrastructure failure
    if any(
        phrase in text
        for phrase in [
            "deployment failed",
            "deployment failure",
            "deploy failed",
            "server failed",
            "server failure"
        ]
    ):
        urgency_score += 5

    # Multiple users / customers affected
    if any(
        phrase in text
        for phrase in [
            "all customers",
            "all users",
            "multiple customers",
            "multiple users",
            "customers are unable",
            "users are unable"
        ]
    ):
        urgency_score += 10

    # Explicit urgency
    if any(
        phrase in text
        for phrase in [
            "urgent",
            "immediately",
            "asap",
            "critical"
        ]
    ):
        urgency_score += 5

    # Prevent urgency rules from dominating criticality
    urgency_score = min(urgency_score, 40)

    score += urgency_score

    return min(score, 100)


model_data["priority_score"] = model_data.apply(
    calculate_priority,
    axis=1
)


# ============================================================
# 16. CONVERT SCORE INTO PRIORITY LEVEL
# ============================================================

def priority_level(score):

    if score >= 80:
        return "CRITICAL"

    elif score >= 60:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


model_data["priority"] = model_data["priority_score"].apply(
    priority_level
)


# ============================================================
# 17. DISPLAY PRIORITY RESULTS
# ============================================================

print("\n==============================")
print("PRIORITY SCORING")
print("==============================")

print("\nPriority distribution:")
print(model_data["priority"].value_counts())

print("\nPriority score statistics:")
print(model_data["priority_score"].describe())

print("\nSample priority results:")

print(
    model_data[
        [
            "subject",
            "email_criticality",
            "priority_score",
            "priority"
        ]
    ].head(10)
)


# ============================================================
# 18. CONFUSION MATRIX
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

plt.title(
    "Email Criticality Confusion Matrix"
)

plt.savefig(
    "criticality_confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix saved successfully!")


# ============================================================
# 19. STORE CASES IN SQLITE
# ============================================================

conn = sqlite3.connect("support_cases.db")

model_data[
    [
        "thread_id",
        "subject",
        "message_body",
        "email_criticality",
        "priority_score",
        "priority"
    ]
].to_sql(
    "cases",
    conn,
    if_exists="replace",
    index=False
)


print("\n==============================")
print("SQLITE DATABASE")
print("==============================")

print(
    "\nCases stored in database:",
    len(model_data)
)

print("\nDatabase table:")

print(
    pd.read_sql_query(
        "SELECT * FROM cases LIMIT 5",
        conn
    )
)

conn.close()


print("\n==============================")
print("PROGRAM COMPLETED")
print("==============================")