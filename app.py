# app.py — Credit Card Clustering (K=3, BA narrative, no silhouette, no PCA)
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Credit Card Clustering — Context • Problem • Action • Results",
    page_icon="📊",
    layout="wide",
)

# =========================
# Header & Story Framework
# =========================
st.title("📊 Credit Card Customer Clustering")
st.caption("Context • Problem • Actions • Results")

with st.expander("Context & Problem Statement", expanded=True):
    st.markdown("""
**Context**  
AllLife Bank wants to understand its credit-card customers better to improve penetration and upgrade service delivery.

**Problem Statement**  
Identify distinct customer segments based on **credit capacity** and **service behaviors** (branch/online/calls) so Marketing and Ops can:
- run targeted **upsell/loyalty** programs, and  
- reduce servicing cost via **digital/self-service** where appropriate.
""")

# =========================
# Upload
# =========================
uploaded = st.file_uploader(
    "Upload the Credit Card CSV (e.g., Sl_No, Customer Key, Avg_Credit_Limit, Total_Credit_Cards, Total_visits_bank, Total_visits_online, Total_calls_made)",
    type=["csv"],
)
if not uploaded:
    st.info("Please upload your dataset to proceed.")
    st.stop()

# Read + normalize
try:
    df = pd.read_csv(uploaded)
    # Normalize headers (spaces → underscores, uppercase) for consistency
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.upper()
    )
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.success("File loaded successfully ✅")
st.markdown("### Raw Preview")
st.dataframe(df.head())

# =========================
# Actions — Data Preparation
# =========================
with st.expander("Actions — Data Preparation", expanded=True):
    st.markdown("""
**Steps performed:**
1) Drop ID-like columns (`SL_NO`, `CUSTOMER_KEY`) if present  
2) Remove **duplicate** feature rows  
3) Scale numeric features with **StandardScaler** (required for KMeans)  
""")

# 1) drop id columns if present (preserve index for alignment)
drop_cols = [c for c in ["SL_NO", "CUSTOMER_KEY"] if c in df.columns]
work = df.drop(columns=drop_cols, errors="ignore").copy()

# 2) remove duplicate feature rows; keep original index to align labels
dupe_mask = work.duplicated()
work = work.loc[~dupe_mask].copy()

# numeric subset
numeric_work = (
    work.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .copy()
)

# Expected feature set based on your file
FEATS = [
    "AVG_CREDIT_LIMIT",
    "TOTAL_CREDIT_CARDS",
    "TOTAL_VISITS_BANK",
    "TOTAL_VISITS_ONLINE",
    "TOTAL_CALLS_MADE",
]
avail_feats = [c for c in FEATS if c in numeric_work.columns]
if len(avail_feats) < 3:
    st.error(f"Expected at least 3 of {FEATS}, found {avail_feats}. Please verify your headers.")
    st.stop()

# Scale features
scaler = StandardScaler()
X = numeric_work[avail_feats].values
X_scaled = scaler.fit_transform(X)

# =========================
# EDA — Distributions & Correlations
# =========================
st.markdown("### Exploratory Analysis")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Distributions")
    feat = st.selectbox("Choose feature", options=avail_feats, index=0, key="dist_feat")
    fig, ax = plt.subplots()
    sns.histplot(numeric_work[feat], bins=30, ax=ax)
    ax.set_xlabel(feat); ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Correlation Heatmap")
    corr = numeric_work[avail_feats].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="Blues", annot=False, ax=ax)
    st.pyplot(fig)
    st.markdown("""
    **Correlation matrix**  
  - Avg_Credit_Limit is **positively** correlated with **Total_Credit_Cards** and **Total_visits_online** (makes sense).  
  - Avg_Credit_Limit is **negatively** correlated with **Total_calls_made** and **Total_visits_bank**.  
  - **Total_visits_bank**, **Total_visits_online**, **Total_calls_made** are **negatively correlated**, implying most customers use only one of these channels to contact the bank.
    """)

# =========================
# KMeans — Fixed K=3 (no silhouette / no picker / no PCA)
# =========================
st.markdown("### KMeans Clustering (K = 3)")

# Optional transparency: Elbow (SSE vs K) only
with st.expander("Elbow Method (SSE vs K)", expanded=False):
    sse = {}
    K_range = range(1, min(10, max(2, X_scaled.shape[0])))
    for k_ in K_range:
        try:
            sse[k_] = KMeans(n_clusters=k_, n_init=10, random_state=1).fit(X_scaled).inertia_
        except Exception:
            sse[k_] = np.nan
    fig, ax = plt.subplots()
    ax.plot(list(sse.keys()), list(sse.values()), "bx-")
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("SSE (Inertia)")
    st.pyplot(fig)

# Final KMeans with K=3
k = 3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
labels = kmeans.fit_predict(X_scaled)

# Align labels back to original df rows used in clustering
scored = df.copy()
scored["CLUSTER"] = np.nan
scored.loc[work.index, "CLUSTER"] = labels

# =========================
# Results — Descriptions then Table
# =========================
st.markdown("## Results")

# Cluster profiles (means) for labeled rows only
profile = (
    scored.loc[scored["CLUSTER"].notna()]
    .groupby("CLUSTER")[avail_feats]
    .mean()
    .round(2)
)

# ---- Cluster Descriptions (from your brief) ----
st.subheader("Cluster Descriptions")
st.markdown("""
- **Cluster 0**: Individuals with **low average credit limit**, **very few or no credit cards**, and who **use the phone** as the primary method to contact the bank.  
- **Cluster 1**: Individuals with **average credit limits**, **average number of credit cards**, and a **tendency to contact the bank in person** (branch visits).  
- **Cluster 2**: Individuals with **very high average credit limit**, **high count of credit cards**, and a **tendency to use online banking services**.
""")

# ---- Beautified Cluster Profile Table ----
st.subheader("Cluster Profiles (Mean by Feature)")
if not profile.empty:
    # Pretty labels for columns
    nice_names = {
        "AVG_CREDIT_LIMIT": "Avg Credit Limit",
        "TOTAL_CREDIT_CARDS": "Total Credit Cards",
        "TOTAL_VISITS_BANK": "Bank Visits",
        "TOTAL_VISITS_ONLINE": "Online Visits",
        "TOTAL_CALLS_MADE": "Calls Made",
    }
    disp = profile.rename(columns={c: nice_names.get(c, c) for c in profile.columns})

    # Format currency for credit limit; integers for counts
    def _fmt(val, col):
        if col == "Avg Credit Limit":
            return f"${val:,.0f}"
        else:
            return f"{val:,.2f}"

    styler = (
        disp.style
        .format({col: (lambda v, c=col: _fmt(v, c)) for col in disp.columns})
        .background_gradient(cmap="Blues", axis=0)
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#0b1220"), ("color", "white"), ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "right")]},
        ])
    )
    st.dataframe(styler, use_container_width=True)
else:
    st.info("No cluster profiles available (no labeled rows).")

# =========================
# Fixed Observations (your exact text)
# =========================
st.markdown("### Observations")
st.markdown("""
- The distribution of average credit limit is heavily skewed to the right. The median is **$18,000** while the mean is **$34,878**. There is also considerable variation among the individuals' credit limits as the standard deviation is **$37,813**.
- Half of the individuals have between **3 and 6** credit cards.
- Individuals typically make between **1 and 4** total bank visits, with a maximum value of **10**.
- Total online visits also typically range between **1 and 4**, with a maximum value of **15**.
- Individuals typically make between **1 and 5** calls to the bank, with a maximum of **10**.
- A majority of customers have credit limits below **50,000**, with the most frequent range between **10,000 and 25,000**.
- The most common number of credit cards is **4**, followed by **6** and **7**. The data appears slightly left-skewed, with fewer customers having very high or very low numbers of credit cards.
""")

# =========================
# Downloads
# =========================
st.markdown("### Downloads")
c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    st.download_button(
        "⬇️ Download data with cluster labels (CSV)",
        data=scored.to_csv(index=False),
        file_name="clustered_customers.csv",
        mime="text/csv",
    )
with c_dl2:
    # Markdown report
    lines = []
    lines.append("# Credit Card Clustering — Results\n")
    lines.append("- K chosen: 3\n")
    lines.append("\n## Cluster Descriptions\n")
    lines.append("- Cluster 0: low credit limit, few/no cards, phone-first.\n")
    lines.append("- Cluster 1: average credit limit/cards, branch tendency.\n")
    lines.append("- Cluster 2: very high credit limit, many cards, online-first.\n")
    lines.append("\n## Cluster Profiles (Mean)\n")
    if not profile.empty:
        lines.append(profile.to_csv())
    st.download_button(
        "⬇️ Download Results (Markdown)",
        data="".join(lines),
        file_name="clustering_results.md",
        mime="text/markdown",
    )
