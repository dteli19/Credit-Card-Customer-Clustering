# app.py — Credit Card Clustering with Observations (BA narrative)
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Credit Card Clustering — Context • Problem • Action • Results",
                   page_icon="📊", layout="wide")

# =========================
# Header & Story Framework
# =========================
st.title("📊 Credit Card Customer Clustering")
st.caption("Context • Problem • Actions • Results (mirrors your Google Colab workflow)")

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
uploaded = st.file_uploader("Upload the Credit Card CSV (7 columns as in your Colab)", type=["csv"])
if not uploaded:
    st.info("Expected columns: Sl_No, Customer Key, Avg_Credit_Limit, Total_Credit_Cards, Total_visits_bank, Total_visits_online, Total_calls_made")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    # normalize headers for consistent downstream code
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.upper())
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.success("File loaded successfully ✅")
st.markdown("### Raw Preview")
st.dataframe(df.head())

# =========================
# Colab-like Preprocessing
# =========================
with st.expander("Actions — Data Preparation", expanded=True):
    st.markdown("""
**Steps performed (as per the Colab):**
1) Drop ID-like columns (`Sl_No`, `Customer Key`).  
2) Remove duplicate rows (identical features).  
3) Scale numeric features with **StandardScaler** for distance-based clustering.  
""")

# 1) drop id columns if present
drop_cols = [c for c in ["SL_NO", "CUSTOMER_KEY"] if c in df.columns]
work = df.drop(columns=drop_cols, errors="ignore").copy()

# 2) remove duplicates
before = len(work)
work = work[~work.duplicated()]
after = len(work)

# 3) numeric check & scaling
numeric_df = work.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()

# Expected feature set from your dataset
FEATS = [
    "AVG_CREDIT_LIMIT",
    "TOTAL_CREDIT_CARDS",
    "TOTAL_VISITS_BANK",
    "TOTAL_VISITS_ONLINE",
    "TOTAL_CALLS_MADE",
]
avail_feats = [c for c in FEATS if c in numeric_df.columns]
if len(avail_feats) < 3:
    st.error(f"Expected at least 3 of {FEATS}, found {avail_feats}. Please verify your headers.")
    st.stop()

scaler = StandardScaler()
X = numeric_df[avail_feats].values
X_scaled = scaler.fit_transform(X)

col_a, col_b = st.columns(2)
with col_a:
    st.metric("Rows after cleaning", after)
with col_b:
    st.metric("Features used", len(avail_feats))

# =========================
# EDA — Distributions & Correlations
# =========================
st.markdown("### Exploratory Analysis")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Distributions")
    feat = st.selectbox("Choose feature", options=avail_feats, index=0, key="dist_feat")
    fig, ax = plt.subplots()
    sns.histplot(numeric_df[feat], bins=30, ax=ax)
    ax.set_xlabel(feat); ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Correlation Heatmap")
    corr = numeric_df[avail_feats].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="Blues", annot=False, ax=ax)
    st.pyplot(fig)

# =========================
# KMeans — Elbow + Fit
# =========================
st.markdown("### KMeans Clustering")
col_elbow, col_k = st.columns([2, 1])

with col_elbow:
    st.caption("Elbow Method (SSE vs K)")
    sse = {}
    K_range = range(1, 11)
    for k_ in K_range:
        km_ = KMeans(n_clusters=k_, n_init=10, max_iter=1000, random_state=1).fit(X_scaled)
        sse[k_] = km_.inertia_
    fig, ax = plt.subplots()
    ax.plot(list(sse.keys()), list(sse.values()), "bx-")
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("SSE (Inertia)")
    st.pyplot(fig)

with col_k:
    st.caption("Pick K (default 3 as in Colab)")
    k = st.slider("K", min_value=2, max_value=10, value=3, step=1)

# Fit final KMeans
kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
labels = kmeans.fit_predict(X_scaled)

# PCA Projection
pca = PCA(n_components=2, random_state=42)
embed = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
sc = ax.scatter(embed[:, 0], embed[:, 1], c=labels, cmap="tab10")
ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_title("PCA Projection Colored by Cluster")
st.pyplot(fig)

# Attach labels
scored = df.copy()
scored["CLUSTER"] = labels

# =========================
# Results — Cluster Profiles & Observations
# =========================
st.markdown("## Results")

# Summary & metrics
sizes = scored["CLUSTER"].value_counts().sort_index()
try:
    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else np.nan
except Exception:
    sil = np.nan

col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.metric("Chosen K", k)
with col_r2:
    st.metric("Silhouette", "N/A" if pd.isna(sil) else f"{sil:.3f}")
with col_r3:
    st.metric("Segments (sizes)", ", ".join([f"C{int(i)}:{int(v)}" for i, v in sizes.items()]))

# Profiles table (means)
st.subheader("Cluster Profiles (Mean by Feature)")
profile = scored.groupby("CLUSTER")[avail_feats].mean().round(2)
st.dataframe(profile)

# Optional: Boxplots by cluster (mirrors Colab)
with st.expander("Boxplots by Cluster (scaled features)"):
    data_scaled_copy = pd.DataFrame(X_scaled, columns=avail_feats)
    data_scaled_copy["CLUSTER"] = labels
    fig, axes = plt.subplots(1, len(avail_feats), figsize=(4*len(avail_feats), 4), sharey=True)
    if len(avail_feats) == 1:
        axes = [axes]
    for i, f in enumerate(avail_feats):
        sns.boxplot(x="CLUSTER", y=f, data=data_scaled_copy, ax=axes[i])
        axes[i].set_title(f)
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------
# Auto Observations (EDA + Profiles)
# -------------------------
st.subheader("Observations (Auto-generated)")

# From describe
desc = numeric_df[avail_feats].describe().T
obs = []

# Credit limit skew cue
if "AVG_CREDIT_LIMIT" in desc.index:
    mean_lim = desc.loc["AVG_CREDIT_LIMIT", "mean"]
    median_lim = desc.loc["AVG_CREDIT_LIMIT", "50%"]
    if mean_lim > median_lim * 1.3:
        obs.append("Average credit limit is **right-skewed** (mean ≫ median), indicating a small group with very high limits.")

# Typical ranges cue for discrete-ish counts
for col in ["TOTAL_CREDIT_CARDS", "TOTAL_VISITS_BANK", "TOTAL_VISITS_ONLINE", "TOTAL_CALLS_MADE"]:
    if col in desc.index:
        q1 = desc.loc[col, "25%"]; q3 = desc.loc[col, "75%"]
        obs.append(f"**{col}** typically ranges between **{q1:.0f}** and **{q3:.0f}**.")

# Correlation cues
corr_abs = numeric_df[avail_feats].corr(numeric_only=True).abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
top_corr = upper.stack().sort_values(ascending=False).head(5)
if not top_corr.empty:
    txt = "; ".join([f"{a}–{b} ({v:.2f})" for (a, b), v in top_corr.items()])
    obs.append(f"Strongest correlations among features: {txt}.")

# Profile contrasts
for f in avail_feats:
    hi = int(profile[f].idxmax())
    lo = int(profile[f].idxmin())
    if hi != lo:
        obs.append(f"**{f}** is highest in **C{hi}** and lowest in **C{lo}**.")

if obs:
    for o in obs:
        st.markdown(f"- {o}")
else:
    st.markdown("- No notable skews/correlations or profile contrasts detected yet.")

# -------------------------
# Downloadable artifacts
# -------------------------
st.markdown("### Downloads")
c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    st.download_button("⬇️ Download data with cluster labels (CSV)",
                       data=scored.to_csv(index=False),
                       file_name="clustered_customers.csv",
                       mime="text/csv")
with c_dl2:
    # Markdown report
    lines = []
    lines.append("# Credit Card Clustering — Results\n")
    lines.append(f"- K chosen: {k}\n")
    lines.append(f"- Silhouette: {'N/A' if pd.isna(sil) else f'{sil:.3f}'}\n")
    lines.append("- Cluster sizes: " + ", ".join([f"C{int(i)}:{int(v)}" for i, v in sizes.items()]) + "\n\n")
    lines.append("## Cluster Profiles (Mean)\n")
    lines.append(profile.to_csv())
    lines.append("\n## Observations\n")
    for o in obs:
        lines.append(f"- {o}\n")
    st.download_button("⬇️ Download Results (Markdown)",
                       data="".join(lines),
                       file_name="clustering_results.md",
                       mime="text/markdown")

