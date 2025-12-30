import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HVAI – Forensic Dashboard",
    layout="wide"
)

st.title("AI-Driven Horizontal–Vertical Anomaly Mapper (HVAI)")
st.caption("Financial Statement Analysis & Forensic Auditing – WAI Project")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    income = pd.read_excel("BSE_Financials_HVAI.xlsx", sheet_name="income_statement")
    balance = pd.read_excel("BSE_Financials_HVAI.xlsx", sheet_name="balance_sheet")

    income["company_id"] = income["company_id"].str.strip()
    balance["company_id"] = balance["company_id"].str.strip()

    df = pd.merge(income, balance, on=["company_id", "year"], how="inner")
    df = df.sort_values(by=["company_id", "year"])

    return df

df = load_data()

# -----------------------------
# Ratio Calculations
# -----------------------------
df["net_margin"] = df["net_profit"] / df["revenue"]
df["asset_turnover"] = df["revenue"] / df["total_assets"]
df["debt_equity"] = df["total_debt"] / df["equity"]

# -----------------------------
# Vertical Analysis
# -----------------------------
df["inventory_pct"] = df["inventory"] / df["total_assets"]
df["receivables_pct"] = df["receivables"] / df["total_assets"]

# -----------------------------
# Horizontal Analysis
# -----------------------------
df["revenue_growth"] = df.groupby("company_id")["revenue"].pct_change()
df["inventory_growth"] = df.groupby("company_id")["inventory"].pct_change()

# -----------------------------
# AI Model – Isolation Forest
# -----------------------------
features = df[
    [
        "inventory_pct",
        "receivables_pct",
        "revenue_growth",
        "inventory_growth",
        "net_margin"
    ]
].fillna(0)

iso = IsolationForest(
    n_estimators=100,
    contamination=0.15,
    random_state=42
)

df["HVAI"] = -1 * iso.fit_predict(features) * iso.decision_function(features)

# -----------------------------
# Sidebar
# -----------------------------
company = st.sidebar.selectbox(
    "Select Company",
    df["company_id"].unique()
)

company_df = df[df["company_id"] == company]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Financial Statements", "Ratios", "HV Analysis", "AI Output"]
)

# -----------------------------
# Tab 1: Financial Statements
# -----------------------------
with tab1:
    st.subheader("Income Statement")
    st.dataframe(
        company_df[
            ["year", "revenue", "cogs", "ebitda", "net_profit"]
        ]
    )

    st.subheader("Balance Sheet")
    st.dataframe(
        company_df[
            ["year", "total_assets", "inventory", "receivables", "total_debt", "equity"]
        ]
    )

# -----------------------------
# Tab 2: Ratios
# -----------------------------
with tab2:
    st.subheader("Key Ratios")
    st.dataframe(
        company_df[
            ["year", "net_margin", "asset_turnover", "debt_equity"]
        ]
    )

# -----------------------------
# Tab 3: Horizontal & Vertical Analysis
# -----------------------------
with tab3:
    st.subheader("Vertical Analysis")

    fig, ax = plt.subplots()
    ax.plot(company_df["year"], company_df["inventory_pct"], label="Inventory %")
    ax.plot(company_df["year"], company_df["receivables_pct"], label="Receivables %")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Horizontal Analysis")

    fig2, ax2 = plt.subplots()
    ax2.plot(company_df["year"], company_df["revenue_growth"], label="Revenue Growth")
    ax2.plot(company_df["year"], company_df["inventory_growth"], label="Inventory Growth")
    ax2.legend()
    st.pyplot(fig2)

# -----------------------------
# Tab 4: AI Output
# -----------------------------
with tab4:
    st.subheader("HVAI – Forensic Risk Indicator")
    st.dataframe(
        company_df[
            ["year", "HVAI"]
        ].sort_values("HVAI", ascending=False)
    )

    st.info(
        "HVAI is an early-warning forensic indicator and does not represent fraud confirmation."
    )
