%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Driven HV Anomaly Mapper", layout="wide")

st.title("AI-Driven Horizontal–Vertical Anomaly Mapper")
st.write("Early warning forensic dashboard for BSE firms")

# LOAD DATA
file = "BSE_Financials_HVAI.xlsx"

income = pd.read_excel(file, sheet_name="income_statement")
balance = pd.read_excel(file, sheet_name="balance_sheet")

income["company_id"] = income["company_id"].str.strip()
balance["company_id"] = balance["company_id"].str.strip()

df = pd.merge(income, balance, on=["company_id", "year"], how="inner")
df = df.sort_values(by=["company_id", "year"])

# RATIOS
df["net_margin"] = df["net_profit"] / df["revenue"]
df["asset_turnover"] = df["revenue"] / df["total_assets"]
df["debt_equity"] = df["total_debt"] / df["equity"]

# VERTICAL ANALYSIS
df["inventory_pct"] = df["inventory"] / df["total_assets"]
df["receivables_pct"] = df["receivables"] / df["total_assets"]

# HORIZONTAL ANALYSIS
df["revenue_growth"] = df.groupby("company_id")["revenue"].pct_change()
df["inventory_growth"] = df.groupby("company_id")["inventory"].pct_change()

# AI MODEL
features = df[
    [
        "inventory_pct",
        "receivables_pct",
        "revenue_growth",
        "inventory_growth",
        "net_margin"
    ]
].dropna()

iso = IsolationForest(contamination=0.15, random_state=42)
features["anomaly_flag"] = iso.fit_predict(features)

df.loc[features.index, "anomaly_flag"] = features["anomaly_flag"]

df["HVAI"] = -1 * iso.decision_function(
    df[
        [
            "inventory_pct",
            "receivables_pct",
            "revenue_growth",
            "inventory_growth",
            "net_margin"
        ]
    ].fillna(0)
)

# SIDEBAR
company = st.sidebar.selectbox(
    "Select Company",
    df["company_id"].unique()
)

company_df = df[df["company_id"] == company]

# TABS
tab1, tab2, tab3, tab4 = st.tabs(
    ["Financial Statements", "Ratios", "HV Analysis", "AI Output"]
)

# TAB 1
with tab1:
    st.subheader("Income Statement")
    st.dataframe(company_df[["year", "revenue", "cogs", "ebitda", "net_profit"]])

    st.subheader("Balance Sheet")
    st.dataframe(company_df[["year", "total_assets", "inventory", "receivables", "total_debt", "equity"]])

# TAB 2
with tab2:
    st.subheader("Key Ratios")
    st.dataframe(company_df[["year", "net_margin", "asset_turnover", "debt_equity"]])

# TAB 3
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

# TAB 4
with tab4:
    st.subheader("HVAI – Forensic Risk Indicator")
    st.dataframe(company_df[["year", "HVAI", "anomaly_flag"]].sort_values("HVAI", ascending=False))
