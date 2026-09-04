# Customer Support Intelligence & Prioritization System

An ML-powered support triage system that turns incoming customer emails into an actionable, ranked priority queue.

## Why this project?

Customer support teams deal with high email volume, but not every case deserves the same attention. A question about an account setting and a full production outage can land in the same inbox within minutes of each other. The hard part isn't reading the emails — it's knowing which ones to look at first.

This project addresses that directly: it combines a trained ML classifier with operational business rules to automatically triage incoming support emails and rank them by priority.

## What it does

The system takes a support email and produces:

```
Case Criticality  →  Operational Urgency  →  Priority Score  →  Priority Level
```

**Example:**

> "Our production API is completely down for all customers."

```
Criticality      HIGH
Priority Score   100 / 100
Priority         CRITICAL
```

The result is an ordered support queue where high-impact cases surface first — instead of being handled in whatever order they happened to arrive.

## How it works

```
                 Incoming Email
                       │
                       ▼
                  Gmail API
                       │
                       ▼
                Gmail Reader
                       │
                       ▼
                    FastAPI
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Criticality Model    Urgency Rules
             │                   │
             └─────────┬─────────┘
                       ▼
                Priority Score
                       │
                       ▼
                    SQLite
                       │
                       ▼
              Streamlit Dashboard
```

### 1. Criticality Classification

A TF-IDF + Logistic Regression model classifies each support case into **High**, **Medium**, or **Low** criticality, using the email subject and message body as input. The model was trained on 671 case-level examples (thread-aggregated from 2,259 raw emails), with hyperparameters selected via cross-validation on the training set only — the held-out test set was used exactly once, for final evaluation, to keep the reported score honest.

### 2. Operational Urgency

Criticality alone isn't the full picture — a message can be worded calmly and still describe something serious, or worded urgently about something minor. To catch that, the system also scans for operational signals such as:

- Production impact
- Service/API outages
- Customers unable to access the service
- Deployment or server failures
- Explicit urgency language

Each detected signal contributes to the urgency score. *(Fill in your actual weighting here — e.g. "each signal adds a fixed point value, capped at N points" — so the exact scoring logic is documented, not just the signal list.)*

### 3. Priority Scoring

Criticality and Urgency combine into a single 0–100 priority score:

| Score  | Priority     |
|--------|-------------|
| 80–100 | 🔴 CRITICAL |
| 60–79  | 🟠 HIGH     |
| 40–59  | 🟡 MEDIUM   |
| 0–39   | 🟢 LOW      |

## Machine Learning Results

The dataset contains 2,259 support emails across 671 threads. For modeling, the problem was formulated at the case/thread level (671 examples), aggregating each email reply-chain into one training instance and filtering to customer-originated messages only.

| Metric                    | Result |
|----------------------------|--------|
| Test Accuracy              | 76.3%  |
| Macro F1                   | 0.74   |
| High-Criticality Recall    | 94%    |
| Majority-Class Baseline    | 37.6%  |
| 5-Fold CV Macro F1         | 73.5%  |

The model performs strongest on high-criticality cases (94% recall) — this matters because failing to surface a genuinely serious issue is operationally far more costly than over-prioritizing a routine one. The model is more than double the majority-class baseline (37.6%), and error analysis shows the weakest class is Medium, which is expected: it's the boundary category between High and Low, and even human triage would disagree on borderline cases here.

## Dashboard

The Streamlit dashboard gives an operational view of the support workload:

- Total cases
- Critical and high-priority case counts
- Average priority score
- Priority distribution
- Criticality distribution
- Criticality vs. priority analysis
- Priority score distribution
- Top priority queue
- Case filtering and search

*(Add 1–3 screenshots of the live dashboard here once finalized — a visual of the priority queue and the distribution charts makes this section far more convincing than the bullet list alone.)*

## A Design Decision: Why Not Sentiment?

Sentiment analysis was initially considered as part of the prioritization system. However, emotional tone isn't reliably correlated with technical severity.

A customer can write:

> "Hi Support, our production API is currently unavailable. Please investigate when possible."

— completely professional and emotionally neutral, while the underlying issue is severe. Conversely, an angry customer may be reporting something minor.

Because of this mismatch, sentiment was removed from the final prioritization pipeline. The system instead focuses on signals directly relevant to support operations:

```
Criticality + Operational Urgency → Priority
```

## Tech Stack

**Machine Learning:** Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression
**Backend:** FastAPI, SQLite
**Integration:** Gmail API, OAuth 2.0
**Dashboard:** Streamlit, Plotly

## Project Structure

```
customer-support-intelligence/
│
├── main.py                 # Model training & evaluation
├── app.py                  # FastAPI inference API
├── gmail_reader.py         # Gmail email ingestion
├── dashboard.py            # Streamlit dashboard
│
├── criticality_model.pkl   # Trained ML pipeline
├── support_cases.db        # SQLite database
├── dataset.csv             # Training dataset
│
└── README.md
```

## Running the Project

**1. Install dependencies**
```bash
pip install pandas scikit-learn fastapi uvicorn requests streamlit plotly joblib
pip install google-api-python-client google-auth google-auth-oauthlib
```

**2. Train the model**
```bash
python main.py
```

**3. Start the API**
```bash
uvicorn app:app --reload
```

**4. Start Gmail ingestion**
```bash
python gmail_reader.py
```
*(Requires a Gmail API OAuth credentials file — see Google's Gmail API quickstart for setup. Do not commit credentials to the repo.)*

**5. Launch the dashboard**
```bash
streamlit run dashboard.py
```

## Limitations

This is an MVP, not a production support platform:

- The ML dataset is relatively small at the case level (671 examples).
- Criticality labels contain some ambiguity — a handful of threads showed inconsistent labels across the same conversation.
- Operational urgency is currently rule-based, not learned from data.
- Gmail ingestion is manually triggered, not a continuously running service.
- The system prioritizes cases; it does not automatically resolve or assign them.

## Future Improvements

- Background Gmail polling or webhook-based ingestion (instead of manual triggering)
- SLA-aware prioritization
- Resolution-time prediction
- Automatic support-team assignment
- Learning operational urgency weights from historical outcomes instead of fixed rules
- Exploring transformer-based classification if dataset size grows enough to justify it

## The Goal

This project isn't about replacing support agents. It answers a simpler operational question:

> "Out of all the emails we received, which ones should we look at first?"

---
Built with Python, Scikit-learn, FastAPI, Gmail API, SQLite, and Streamlit.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/77ddf7ed-e6da-4b50-b78e-d3c1eae78524" />

