Customer Support Intelligence & Prioritization System

An ML-powered support triage system that turns incoming customer emails into an actionable priority queue.
Why this project?

Customer support teams often deal with a large volume of emails, but not every case requires the same level of attention.
A question about an account setting and a production outage can arrive in the same inbox. The challenge is identifying which cases need attention first.
This project explores that problem by combining machine learning with operational business rules to automatically assess incoming support emails and assign them a priority.

What it does

The system takes a support email and produces:
Case Criticality → Operational Urgency → Priority Score → Priority Level
For example:
"Our production API is completely down for all customers."

The system identifies the case as:

Criticality      HIGH
Priority Score   100 / 100
Priority         CRITICAL

The result is an ordered support queue where high-impact cases can be surfaced first.

How it works
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
1. Criticality Classification

A TF-IDF + Logistic Regression model classifies support cases into:

High
Medium
Low

The model uses the email subject and message body as input.

2. Operational Urgency

Criticality alone isn't enough.

The system also looks for operational signals such as:

Production impact
Service/API outages
Customers being unable to access the service
Deployment or server failures
Explicit urgency

These signals increase the priority score.

3. Priority Scoring

The two components are combined into a 0–100 priority score.

Score	Priority
80–100	🔴 CRITICAL
60–79	🟠 HIGH
40–59	🟡 MEDIUM
0–39	🟢 LOW
Machine Learning Results

The dataset contains 2,259 support emails across 671 threads.

For modeling, the problem was formulated at the case/thread level, resulting in 671 case-level examples.

Test Performance
Metric	Result
Accuracy	76.3%
Macro F1	0.74
High-Criticality Recall	94%
Majority-Class Baseline	37.6%
5-Fold CV Macro F1	73.5%

The model's strongest performance is on high-criticality cases, where recall is particularly important because failing to surface a serious support issue can have a much larger operational impact than over-prioritizing a routine case.

Dashboard

The Streamlit dashboard provides an operational view of the support workload, including:

Total cases
Critical and high-priority cases
Average priority score
Priority distribution
Criticality distribution
Criticality vs. priority analysis
Priority score distribution
Top priority queue
Case filtering and search
A Design Decision: Why Not Sentiment?

Sentiment analysis was initially considered as part of the prioritization system.

However, emotional tone isn't necessarily correlated with technical severity.

A customer can write:

"Hi Support, our production API is currently unavailable. Please investigate when possible."
The email may sound completely professional and emotionally neutral, while the underlying issue is extremely serious.
Conversely, an angry customer may be reporting a relatively minor issue.
Because of this, sentiment was removed from the final prioritization pipeline.
The final system focuses on signals that are directly relevant to support operations:

Criticality + Operational Urgency → Priority

Tech Stack

Machine Learning

Python
Pandas
Scikit-learn
TF-IDF
Logistic Regression

Backend

FastAPI
SQLite

Integration

Gmail API
OAuth 2.0

Dashboard

Streamlit
Plotly
Project Structure
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
Running the Project
Install dependencies
pip install pandas scikit-learn fastapi uvicorn requests streamlit plotly joblib
pip install google-api-python-client google-auth google-auth-oauthlib
Train the model
python main.py
Start the API
uvicorn app:app --reload
Start Gmail ingestion
python gmail_reader.py
Launch the dashboard
streamlit run dashboard.py
Limitations

This is an MVP rather than a production support platform.

Current limitations include:

The ML dataset is relatively small at the case level.
Criticality labels contain some ambiguity.
Operational urgency is currently rule-based.
Gmail ingestion is manually triggered.
The system prioritizes cases but does not automatically resolve or assign them.
Future Improvements

Possible extensions include:

Automatic support-email classification
Background Gmail polling or webhook-based ingestion
SLA-aware prioritization
Resolution-time prediction
Automatic support-team assignment
Learning operational urgency from historical outcomes
More advanced NLP models
The Goal


The purpose of this project isn't to replace support agents.
It's to answer a much simpler operational question:

"Out of all the emails we received, which ones should we look at first?"


