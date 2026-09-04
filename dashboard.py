import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Support Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Overall page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Header */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 750;
        color: #172033;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.8rem;
    }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e8eaf0;
        border-radius: 16px;
        padding: 20px 22px;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(30, 41, 59, 0.05);
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.04em;
    }

    .kpi-number {
        font-size: 2.05rem;
        font-weight: 750;
        margin-top: 6px;
        color: #172033;
    }

    .kpi-description {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 3px;
    }

    .critical-number {
        color: #dc2626;
    }

    .high-number {
        color: #f59e0b;
    }

    .negative-number {
        color: #16a34a;
    }

    /* Section headings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #172033;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #8a93a3;
        font-size: 0.82rem;
        margin-bottom: 0.7rem;
    }

    /* Queue */
    .queue-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #172033;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_cases():

    conn = sqlite3.connect("support_cases.db")

    df = pd.read_sql_query(
        "SELECT * FROM cases",
        conn
    )

    conn.close()

    return df


df = load_cases()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '📊 Customer Support Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'AI-powered customer case analysis and prioritization'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_cases = len(df)

critical_cases = len(
    df[df["priority"] == "CRITICAL"]
)

high_cases = len(
    df[df["priority"] == "HIGH"]
)

average_score = df["priority_score"].mean()


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">TOTAL CASES</div>
            <div class="kpi-number">{total_cases}</div>
            <div class="kpi-description">
                Customer cases analyzed
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🔴 CRITICAL</div>
            <div class="kpi-number critical-number">
                {critical_cases}
            </div>
            <div class="kpi-description">
                Require immediate attention
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🟠 HIGH PRIORITY</div>
            <div class="kpi-number high-number">
                {high_cases}
            </div>
            <div class="kpi-description">
                High-priority cases
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">📈 AVG PRIORITY SCORE</div>
            <div class="kpi-number">
                {average_score:.1f}
            </div>
            <div class="kpi-description">
                Overall operational priority
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Case Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Distribution of cases by operational priority and predicted criticality.'
    '</div>',
    unsafe_allow_html=True
)


chart_col1, chart_col2 = st.columns(2)


# ============================================================
# PRIORITY DONUT
# ============================================================

with chart_col1:

    priority_counts = (
        df["priority"]
        .value_counts()
        .reset_index()
    )

    priority_counts.columns = [
        "priority",
        "count"
    ]

    fig_priority = px.pie(
        priority_counts,
        names="priority",
        values="count",
        hole=0.62,
        title="Priority Distribution",
        color="priority",
        color_discrete_map={
            "CRITICAL": "#ef4444",
            "HIGH": "#f59e0b",
            "MEDIUM": "#3b82f6",
            "LOW": "#22c55e"
        }
    )

    fig_priority.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Cases: %{value}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        )
    )

    fig_priority.update_layout(
        height=390,
        margin=dict(
            t=65,
            b=15,
            l=15,
            r=15
        ),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig_priority,
        use_container_width=True
    )


# ============================================================
# CRITICALITY DONUT
# ============================================================

with chart_col2:

    criticality_counts = (
        df["email_criticality"]
        .value_counts()
        .reindex(["high", "medium", "low"], fill_value=0)
        .reset_index()
    )

    criticality_counts.columns = ["criticality", "count"]

    fig_criticality = px.pie(
        criticality_counts,
        names="criticality",
        values="count",
        hole=0.62,
        title="Criticality Distribution",
        color="criticality",
        color_discrete_map={
            "high": "#ef4444",
            "medium": "#f59e0b",
            "low": "#22c55e"
        }
    )

    fig_criticality.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Cases: %{value}<br>Share: %{percent}<extra></extra>"
    )

    fig_criticality.update_layout(
        height=390,
        margin=dict(t=65, b=15, l=15, r=15),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_criticality, use_container_width=True)


# ============================================================
# CRITICALITY × PRIORITY HEATMAP
# ============================================================

st.markdown(
    '<div class="section-title">Priority Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'How predicted case criticality translates into operational priority.'
    '</div>',
    unsafe_allow_html=True
)

analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    heatmap_data = pd.crosstab(df["email_criticality"], df["priority"])
    heatmap_data = heatmap_data.reindex(
        index=["high", "medium", "low"],
        columns=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        fill_value=0
    )

    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        labels={"x": "Priority", "y": "Criticality", "color": "Cases"}
    )
    fig_heatmap.update_layout(
        title="Criticality vs Priority",
        height=360,
        margin=dict(t=55, b=20, l=20, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with analysis_col2:
    fig_score = px.histogram(
        df,
        x="priority_score",
        nbins=20,
        title="Priority Score Distribution",
        labels={"priority_score": "Priority Score", "count": "Cases"}
    )
    fig_score.update_layout(
        height=360,
        margin=dict(t=55, b=20, l=20, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig_score, use_container_width=True)


# ============================================================
# PRIORITY QUEUE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 Priority Queue'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Cases ranked by priority score. Higher scores appear first.'
    '</div>',
    unsafe_allow_html=True
)


priority_queue = (
    df
    .sort_values(
        "priority_score",
        ascending=False
    )
    .head(8)
    .copy()
)


# Prepare queue for display

queue_display = priority_queue[
    [
        "priority",
        "subject",
        "email_criticality",
        "priority_score"
    ]
].copy()


queue_display.columns = [
    "Priority",
    "Case",
    "Criticality",
    "Score"
]


# Make score integer

queue_display["Score"] = (
    queue_display["Score"]
    .astype(int)
)


# Display queue

st.dataframe(
    queue_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Priority": st.column_config.TextColumn(
            "Priority",
            width="small"
        ),
        "Case": st.column_config.TextColumn(
            "Case",
            width="large"
        ),
        "Criticality": st.column_config.TextColumn(
            "Criticality",
            width="small"
        ),
        "Score": st.column_config.NumberColumn(
            "Score",
            format="%d",
            width="small"
        )
    }
)


# ============================================================
# FILTERS
# ============================================================

st.markdown(
    '<div class="section-title">Filter Cases</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    priority_filter = st.multiselect(
        "Priority",
        options=df["priority"].unique(),
        default=df["priority"].unique()
    )


with col2:

    criticality_filter = st.multiselect(
        "Criticality",
        options=df["email_criticality"].unique(),
        default=df["email_criticality"].unique()
    )


with col3:

    search_text = st.text_input(
        "Search cases",
        placeholder="Search by subject or message..."
    )


# Apply filters

filtered_df = df[
    df["priority"].isin(priority_filter)
    & df["email_criticality"].isin(
        criticality_filter
    )
]

if search_text.strip():
    search = search_text.strip()
    filtered_df = filtered_df[
        filtered_df["subject"].fillna("").str.contains(search, case=False, na=False)
        | filtered_df["message_body"].fillna("").str.contains(search, case=False, na=False)
    ]


# ============================================================
# SUPPORT CASES
# ============================================================

st.markdown(
    '<div class="section-title">Support Cases</div>',
    unsafe_allow_html=True
)


st.dataframe(
    filtered_df[
        [
            "thread_id",
            "subject",
            "email_criticality",
            "priority_score",
            "priority"
        ]
    ],
    use_container_width=True,
    hide_index=True
)