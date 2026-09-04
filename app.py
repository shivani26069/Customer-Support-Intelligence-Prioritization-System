from fastapi import FastAPI
import joblib
import sqlite3
from datetime import datetime


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading criticality model...")

criticality_model = joblib.load(
    "criticality_model.pkl"
)

print("Criticality model loaded successfully!")


# ============================================================
# DATABASE SETUP
# ============================================================

def setup_database():

    conn = sqlite3.connect(
        "support_cases.db"
    )

    cursor = conn.cursor()

    # Check existing columns
    cursor.execute(
        "PRAGMA table_info(cases)"
    )

    columns = {
        row[1]
        for row in cursor.fetchall()
    }

    # Add Gmail metadata columns if they don't exist
    if "message_id" not in columns:
        cursor.execute(
            "ALTER TABLE cases ADD COLUMN message_id TEXT"
        )

    if "sender" not in columns:
        cursor.execute(
            "ALTER TABLE cases ADD COLUMN sender TEXT"
        )

    if "received_at" not in columns:
        cursor.execute(
            "ALTER TABLE cases ADD COLUMN received_at TEXT"
        )

    conn.commit()
    conn.close()

    print("Database schema verified successfully!")


setup_database()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Customer Support Intelligence API",
    description="Analyzes customer support emails and assigns priority.",
    version="1.0"
)


# ============================================================
# PRIORITY CALCULATION
# ============================================================

def calculate_priority(criticality, text):

    # --------------------------------------------------------
    # Base score from ML-predicted criticality
    # --------------------------------------------------------

    if criticality == "high":
        score = 60

    elif criticality == "medium":
        score = 35

    else:
        score = 15


    # --------------------------------------------------------
    # Operational urgency
    # --------------------------------------------------------

    text = text.lower()

    urgency_score = 0


    # Production environment
    if "production" in text:
        urgency_score += 10


    # Service / API outage
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


    # Prevent urgency rules from dominating ML
    urgency_score = min(
        urgency_score,
        40
    )

    score += urgency_score

    return min(
        score,
        100
    )


# ============================================================
# PRIORITY LEVEL
# ============================================================

def get_priority_level(score):

    if score >= 80:
        return "CRITICAL"

    elif score >= 60:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Customer Support Intelligence API is running"
    }


# ============================================================
# ANALYZE EMAIL
# ============================================================

@app.post("/analyze")
def analyze_email(
    subject: str,
    message_body: str,
    sender: str = "",
    message_id: str = "",
    received_at: str = ""
):

    # --------------------------------------------------------
    # Combine subject + body
    # --------------------------------------------------------

    text = subject + " " + message_body


    # --------------------------------------------------------
    # Criticality prediction
    # --------------------------------------------------------

    criticality = criticality_model.predict(
        [text]
    )[0]


    # --------------------------------------------------------
    # Priority calculation
    # --------------------------------------------------------

    priority_score = calculate_priority(
        criticality,
        text
    )

    priority = get_priority_level(
        priority_score
    )


    # --------------------------------------------------------
    # Save to SQLite
    # --------------------------------------------------------

    conn = sqlite3.connect(
        "support_cases.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO cases (
            message_id,
            sender,
            received_at,
            subject,
            message_body,
            email_criticality,
            priority_score,
            priority
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            sender,
            received_at,
            subject,
            message_body,
            criticality,
            priority_score,
            priority
        )
    )

    conn.commit()
    conn.close()


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "message_id": message_id,

        "sender": sender,

        "subject": subject,

        "criticality": criticality,

        "priority_score": priority_score,

        "priority": priority,

        "timestamp": datetime.now().isoformat()
    }