import os
import json
import streamlit as st
import pandas as pd

from app.forecast_engine import predict_default, forecast_liquidity
from app.ollama_agent import ask_ollama


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="🏦 AI-Driven Loan Default & Liquidity Forecast Assistant",
    page_icon="🏦",
    layout="wide",
)

# ============================================================
# Theme toggle in sidebar (LIGHT / DARK)
# ============================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False  # default LIGHT

with st.sidebar:
    st.markdown("### Appearance")
    dark_mode = st.toggle(
        "🌗 Dark theme",
        value=st.session_state.dark_mode,
        help="Turn ON for dark theme.",
    )

st.session_state.dark_mode = dark_mode

# ============================================================
# Build CSS based on THEME (dark_mode) – NO background image
# ============================================================
if dark_mode:
    # Dark theme colors
    base_text_color = "#e5f0ff"
    header_bg = "transparent"
    header_title_color = "#e5f0ff"
    header_subtitle_color = "#cbd5f5"
    card_bg = "rgba(15, 23, 42, 0.98)"
    card_text_color = "#e5e7eb"
    card_border = "rgba(75, 85, 99, 0.8)"
    kpi_bg = "linear-gradient(135deg, #1d4ed8, #4f46e5)"
    kpi_text_main = "#f9fafb"
    kpi_text_sub = "rgba(209, 213, 219, 0.9)"
    surface_bg = "linear-gradient(180deg, #020617 0%, #020617 100%)"
else:
    # Light theme colors
    base_text_color = "#020617"
    header_bg = "transparent"
    header_title_color = "#020617"
    header_subtitle_color = "#64748b"
    card_bg = "#ffffff"
    card_text_color = "#0f172a"
    card_border = "rgba(148, 163, 184, 0.35)"
    kpi_bg = "linear-gradient(135deg, #eff6ff, #e0e7ff)"
    kpi_text_main = "#111827"
    kpi_text_sub = "#6b7280"
    surface_bg = "linear-gradient(180deg, #f4f7ff 0%, #edf2ff 100%)"

accent_color = "#2563eb"
accent_soft = "rgba(37, 99, 235, 0.08)"

# ============================================================
# Global CSS
# ============================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {surface_bg};
        color: {base_text_color};
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Make content span full width */
.block-container {{
    max-width: 100%;
    padding-top: 2.6rem;      /* was 1.3rem – push content down */
    padding-bottom: 3rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    margin: 0;
}}


    /* --- HEADER STRIP (TITLE + SUBTITLE) --- */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0 10px 0;
        margin-bottom: 4px;
        background: {header_bg};
        border-radius: 0;
        border: none;
        box-shadow: none;
        border-bottom: 1px solid rgba(148, 163, 184, 0.45);
    }}
    .header-icon {{
        width: 36px;
        height: 36px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: {accent_soft};
    }}
    .app-title {{
        display: block;
        font-size: 30px;
        font-weight: 700;
        line-height: 1.2;
        color: {header_title_color};
    }}
    .app-subtitle {{
        display: block;
        font-size: 13px;
        margin-top: 2px;
        color: {header_subtitle_color};
    }}

    /* Tabs as pills */
    .stTabs [role="tablist"] {{
        gap: 8px;
        border-bottom: none;
        margin-bottom: 0.5rem;
    }}
    .stTabs [role="tab"] {{
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        border: 1px solid transparent;
        background-color: rgba(148, 163, 184, 0.10);
        color: #64748b;
        font-size: 0.88rem;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background: {accent_color};
        color: #f9fafb;
        border-color: rgba(15, 23, 42, 0.18);
    }}

    /* Generic section card */
    .card {{
        background: {card_bg};
        border-radius: 18px;
        padding: 18px 20px 18px 20px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.10);
        margin-bottom: 18px;
        border: 1px solid {card_border};
        color: {card_text_color};
    }}
    .card-compact {{
        padding: 14px 18px;
        margin-bottom: 14px;
    }}

    .section-title {{
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }}
    .section-caption {{
        font-size: 13px;
        color: #64748b;
        margin-bottom: 1.0rem;
    }}

    /* KPI cards */
    .kpi-row {{
        display: flex;
        gap: 12px;
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
    }}
    .kpi-card {{
        flex: 1;
        background: {kpi_bg};
        border-radius: 16px;
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .kpi-label {{
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {kpi_text_sub};
    }}
    .kpi-value {{
        font-size: 23px;
        font-weight: 700;
        margin-top: 2px;
        color: {kpi_text_main};
    }}
    .kpi-help {{
        font-size: 11px;
        margin-top: 1px;
        color: {kpi_text_sub};
    }}

    /* Pills / badges */
    .pill {{
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        background: rgba(148, 163, 184, 0.06);
        color: #4b5563;
        gap: 4px;
    }}
    .pill-dot {{
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: {accent_color};
    }}
    .pill-demo {{
        background: {accent_soft};
        border-color: rgba(37, 99, 235, 0.5);
        color: {accent_color};
    }}

    /* Filter labels */
    .filter-label {{
        font-size: 12px;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 0.15rem;
    }}
    /* Make chat input box more visible */
    textarea, .stTextInput > div > div > input {{
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.9) !important;
        background-color: #f9fafb !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Header
# ============================================================
with st.container():
    st.markdown(
        """
        <div class="app-header">
            <div class="header-icon">🏦</div>
            <div>
                <span class="app-title">
                    AI-Driven Loan Default & Liquidity Forecast Assistant
                </span>
                <span class="app-subtitle">
                    Monitor credit risk, forecast branch liquidity, and ask an AI assistant for explanations.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")  # spacer

tab1, tab2, tab3 = st.tabs(
    ["📊 Loan Default Risk", "💧 Liquidity Forecast", "💬 Chat Assistant"]
)

# ============================================================
# TAB 1 – Loan Default Risk (demo/upload toggle for DATA)
# ============================================================
with tab1:
    # ---------- Card 1: Data source controls ----------
    st.markdown('<div class="card card-compact">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Loan Default Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">'
        "Upload your own loan portfolio CSV, or use the toggle to switch to the built-in demo dataset."
        "</div>",
        unsafe_allow_html=True,
    )

        # Upload + toggle (full width)
    uploaded_loans = st.file_uploader(
        "Upload loan CSV",
        type=["csv"],
        key="loans_uploader",
        help=(
            "Expected columns: customer_id, age, gender, income, loan_amount, "
            "tenure_months, emi_paid, balance. 'defaulted' is optional."
        ),
    )

    use_demo_loans = st.toggle(
        "Use demo loans data instead of uploaded file",
        value=(uploaded_loans is None),
        help="Turn this ON to ignore the upload and use the built-in demo dataset.",
    )

    # Tip under the controls
    st.markdown(
        """
        <div style="margin-top: 0.4rem; font-size: 12px; color: #64748b;">
            <b>Tip:</b> Start with the demo data to explore the dashboard.  
            Then upload your bank's portfolio to see live PDs and segment filters.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Decide data source
    if use_demo_loans:
        loans_df = pd.read_csv("data/loans.csv")
        source_label = "Demo data · data/loans.csv"
        pill_class = "pill pill-demo"
    else:
        if uploaded_loans is None:
            st.warning("Please upload a loans CSV or turn ON demo data.")
            st.stop()
        loans_df = pd.read_csv(uploaded_loans)
        source_label = "Custom uploaded data"
        pill_class = "pill"

    st.markdown(
        f"""
        <div style="margin-top: 0.4rem;">
            <span class="{pill_class}">
                <span class="pill-dot"></span>
                {source_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # close card 1

    # ---------- Validate required columns ----------
    required_cols = {
        "customer_id",
        "age",
        "gender",
        "income",
        "loan_amount",
        "tenure_months",
        "emi_paid",
        "balance",
    }
    missing = required_cols - set(loans_df.columns)

    if missing:
        st.error(
            "The loan file is missing required columns: "
            + ", ".join(sorted(missing))
        )
        st.stop()

    # ---------- Run prediction ----------
    preds = predict_default(loans_df.copy())

    if isinstance(preds, pd.DataFrame) and "customer_id" in preds.columns:
        full_df = loans_df.merge(
            preds[["customer_id", "default_probability"]], on="customer_id"
        )
    else:
        full_df = loans_df.copy()
        full_df["default_probability"] = preds

    # Create percentage + useful ratios
    full_df["default_probability_pct"] = full_df["default_probability"] * 100.0
    full_df = full_df.assign(
        emi_to_income_pct=lambda d: 100.0 * d["emi_paid"] / d["income"],
        loan_to_income=lambda d: d["loan_amount"] / d["income"],
    )

    # ---------- Quick metrics ----------
    total_customers = len(full_df)
    high_risk_threshold = 5.0  # 5%+
    high_risk_count = (full_df["default_probability_pct"] >= high_risk_threshold).sum()
    avg_pd = full_df["default_probability_pct"].mean()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.4rem;">
            <div style="font-size:16px;font-weight:600;">Portfolio Overview</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-value">{total_customers:,}</div>
                <div class="kpi-help">{source_label}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">High-Risk (≥ {high_risk_threshold:.1f}% PD)</div>
                <div class="kpi-value">{high_risk_count:,}</div>
                <div class="kpi-help">Count above risk threshold</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Default Probability</div>
                <div class="kpi-value">{avg_pd:.2f}%</div>
                <div class="kpi-help">Portfolio-level PD</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
# ---------- Model Performance Metrics ----------
    metrics_path = "models/xgb_default_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.markdown("#### 📈 Loan Default Model Performance")
        st.write(f"ROC AUC: **{metrics['roc_auc']:.3f}**")
    else:
        st.info("Model metrics not found. Please retrain the default model to generate them.")

    # ---------- Filters ----------
    st.markdown(
        '<div style="font-size:15px;font-weight:600;margin-bottom:0.4rem;">Customer Risk Table</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])

    with f1:
        age_min, age_max = int(full_df["age"].min()), int(full_df["age"].max())
        st.markdown('<div class="filter-label">Age range</div>', unsafe_allow_html=True)
        age_range = st.slider(
            "",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
            label_visibility="collapsed",
        )

    with f2:
        genders = sorted(full_df["gender"].unique())
        st.markdown('<div class="filter-label">Gender</div>', unsafe_allow_html=True)
        gender_filter = st.multiselect(
            "",
            options=genders,
            default=genders,
            label_visibility="collapsed",
        )

    with f3:
        max_pd = float(full_df["default_probability_pct"].max())
        st.markdown(
            '<div class="filter-label">Min default probability (%)</div>',
            unsafe_allow_html=True,
        )
        prob_cutoff = st.slider(
            "",
            min_value=0.0,
            max_value=max_pd,
            value=0.0,
            step=0.5,
            label_visibility="collapsed",
        )

    with f4:
        st.markdown('<div class="filter-label">Top N</div>', unsafe_allow_html=True)
        top_n = st.slider(
            "",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
            label_visibility="collapsed",
        )

    # Apply filters
    filt_df = full_df[
        (full_df["age"].between(age_range[0], age_range[1]))
        & (full_df["gender"].isin(gender_filter))
        & (full_df["default_probability_pct"] >= prob_cutoff)
    ]

    # Columns to show
    display_cols = [
        "customer_id",
        "age",
        "gender",
        "income",
        "loan_amount",
        "tenure_months",
        "emi_paid",
        "balance",
        "loan_to_income",
        "emi_to_income_pct",
        "default_probability_pct",
    ]

    table_df = (
        filt_df[display_cols]
        .sort_values("default_probability_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .rename(
            columns={
                "customer_id": "Customer ID",
                "age": "Age",
                "gender": "Gender",
                "income": "Income",
                "loan_amount": "Loan Amount",
                "tenure_months": "Tenure (Months)",
                "emi_paid": "EMI Paid",
                "balance": "Outstanding Balance",
                "loan_to_income": "Loan-to-Income",
                "emi_to_income_pct": "EMI / Income (%)",
                "default_probability_pct": "Default Prob (%)",
            }
        )
    )

    # Style only this wide table
    styled_table = (
        table_df.style.format(
            {
                "Income": "{:,.0f}",
                "Loan Amount": "{:,.0f}",
                "Outstanding Balance": "{:,.0f}",
                "Loan-to-Income": "{:.2f}x",
                "EMI / Income (%)": "{:.1f}%",
                "Default Prob (%)": "{:.2f}%",
            }
        ).background_gradient(cmap="Reds", subset=["Default Prob (%)"])
    )

    st.dataframe(
        styled_table,
        use_container_width=True,
        height=420,
    )

    st.caption(
        "Redder values indicate customers with higher default probability. "
        "Use the filters above to focus on specific segments."
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 Explain Top Risks"):
            with st.spinner("Generating explanation..."):
                summary = ask_ollama(
                    "You are a credit risk analyst. Explain typical reasons why some customers in this "
                    "loan portfolio might have higher default probability, given features like age, "
                    "income, loan amount, tenure, EMI paid, and outstanding balance."
                )
            st.markdown("##### AI Explanation")
            st.write(summary)

    st.markdown("</div>", unsafe_allow_html=True)  # close big card

# ============================================================
# TAB 2 – Liquidity Forecast (demo/upload toggle for DATA)
# ============================================================
with tab2:
    st.markdown('<div class="card card-compact">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Branch Liquidity Forecast (Next 7 Days)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">'
        "Upload a liquidity CSV for branch history, or use the toggle to switch to the demo liquidity data."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_liq = st.file_uploader(
        "Upload liquidity CSV",
        type=["csv"],
        key="liq_uploader",
        help="Expected columns: branch_id, date, inflow, outflow, balance",
    )

    use_demo_liq = st.toggle(
        "Use demo liquidity data instead of uploaded file",
        value=(uploaded_liq is None),
        help="Turn this ON to ignore the upload and use the built-in demo dataset.",
    )

    st.markdown(
        """
        <div style="margin-top: 0.4rem; font-size: 12px; color: #64748b;">
            <b>Tip:</b> Use this to monitor branches that may run into cash shortfall or
            hold excess idle balances.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if use_demo_liq:
        liq_df = pd.read_csv("data/liquidity.csv", parse_dates=["date"])
        liq_source_label = "Demo data · data/liquidity.csv"
        liq_pill_class = "pill pill-demo"
    else:
        if uploaded_liq is None:
            st.warning("Please upload a liquidity CSV or turn ON demo data.")
            st.stop()
        liq_df = pd.read_csv(uploaded_liq, parse_dates=["date"])
        liq_source_label = "Custom uploaded data"
        liq_pill_class = "pill"

    st.markdown(
        f"""
        <div style="margin-top: 0.4rem;">
            <span class="{liq_pill_class}">
                <span class="pill-dot"></span>
                {liq_source_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Validate required columns
    required_liq_cols = {"branch_id", "date", "inflow", "outflow", "balance"}
    missing_liq = required_liq_cols - set(liq_df.columns)

    if missing_liq:
        st.error(
            "The liquidity file is missing required columns: "
            + ", ".join(sorted(missing_liq))
        )
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    branches = sorted(liq_df["branch_id"].unique())
    branch = st.selectbox("Select Branch", branches)

    # Historical stats for the selected branch
    branch_hist = liq_df[liq_df["branch_id"] == branch].sort_values("date")

    if branch_hist.empty:
        st.warning("No records found for this branch in the liquidity file.")
    else:
        last_balance = branch_hist["balance"].iloc[-1]
        last_30 = branch_hist.tail(30)
        min_30 = last_30["balance"].min()
        days_negative = (branch_hist["balance"] < 0).sum()

        # KPI row
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.4rem;">
                <div style="font-size:15px;font-weight:600;">Liquidity Snapshot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-label">Current Balance</div>
                    <div class="kpi-value">{last_balance:,.0f}</div>
                    <div class="kpi-help">{liq_source_label}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Min Balance (last 30 days)</div>
                    <div class="kpi-value">{min_30:,.0f}</div>
                    <div class="kpi-help">Recent stress level</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Days with Negative Balance</div>
                    <div class="kpi-value">{days_negative}</div>
                    <div class="kpi-help">Across full history</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Forecast for next 7 days using your model
        result = forecast_liquidity(branch)  # expected: date + balance, etc.
        result["date"] = pd.to_datetime(result["date"])

        st.markdown("---")
        st.markdown("#### 7-Day Liquidity Forecast")
        st.line_chart(result.set_index("date"))

        st.caption(f"Historical data source: {liq_source_label}")

        if st.button("📈 Explain Forecast Trend"):
            with st.spinner("Generating explanation..."):
                text = ask_ollama(
                    f"You are a treasury analyst. For branch {branch}, explain the liquidity trend over the "
                    f"next 7 days. Indicate if the branch is likely to face a surplus or shortage and give "
                    f"a few possible reasons such as seasonal inflows/outflows or past balance patterns."
                )
            st.markdown("##### AI Explanation")
            st.write(text)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3 – Chat Assistant
# ============================================================
with tab3:
    st.markdown('<div class="card card-compact">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ask Banking AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">'
        "Use this assistant to ask follow-up questions about credit risk, liquidity, or how to interpret the dashboard."
        "</div>",
        unsafe_allow_html=True,
    )

    if "query_submitted" not in st.session_state:
     st.session_state.query_submitted = False

query = st.text_area(
    "Ask your question:",
    placeholder="For example: Why are younger customers appearing more risky?\n"
                "Or: Which branches have the highest liquidity risk this week?",
    key="query_input",
    height=120,
)

ask_clicked = st.button("Ask", type="primary")


if (st.session_state.query_submitted or ask_clicked) and query.strip() != "":
        with st.spinner("Thinking..."):
            response = ask_ollama(
                "You are a banking risk & treasury expert. "
                "Answer clearly but briefly in 3–6 sentences.\n\n"
                f"Question: {query}"
            )
        st.write(response)
        st.session_state.query_submitted = False

st.markdown("</div>", unsafe_allow_html=True)
