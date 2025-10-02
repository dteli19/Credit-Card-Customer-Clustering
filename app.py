# app.py — Credit Card Clustering with Auto-K, Observations, and Cluster Descriptions (no PCA, no K picker)
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# 1) drop id columns if present (do not change order or index)
drop_cols = [c for c in ["SL_NO", "CUSTOMER_KEY"] if c in df.columns]
work = df.drop(columns=drop_cols, errors="ignore").copy()

# 2) remove duplicate feature rows; keep index to align labels later
before = len(work)
dupe_mask = work.duplicated()
work = work.loc[~dupe_mask].copy()
after = len(work)

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

# 3) scale
scaler = StandardScaler()
X = numeric_work[avail_feats].values
X_scaled = scaler.fit_transform(X)

cA, cB = st.columns(2)
with cA:
    st.metric("Rows after cleaning", after)
with cB:
    st.metric("Features used", len(avail_feats))

# =========================
# Exploratory Analysis
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

# =========================
# KMeans — Auto-pick K (no UI)
# =========================
st.markdown("### KMeans Clustering (Auto-selected K)")

# Auto-select k via silhouette over a reasonable range
def auto_k_selection(Xs, k_min=2, k_max=8):
    best_k, best_sil = None, -1
    ks = []
    sils = []
    upper = min(k_max, max(k_min, Xs.shape[0] - 1))  # avoid invalid k >= n_samples
    for k in range(k_min, max(upper + 1, k_min + 1)):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=1)
            labels = km.fit_predict(Xs)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(Xs, labels)
            else:
                sil = -1
        except Exception:
            sil = -1
        ks.append(k); sils.append(sil)
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k or 3, ks, sils, best_sil if best_sil >= 0 else None

k, ks, sils, best_sil = auto_k_selection(X_scaled, 2, 8)

col_elbow, col_sil = st.columns(2)
with col_elbow:
    st.caption("SSE (Elbow) is computed internally for stability; showing silhouette for transparency.")
    # Compute SSE for display only
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
    ax.set_title("Elbow (SSE vs K)")
    st.pyplot(fig)

with col_sil:
    # Silhouette chart for ks tried
    fig, ax = plt.subplots()
    ax.plot(ks, sils, "go-")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette vs K")
    st.pyplot(fig)

# Final fit with chosen k (no PCA)
kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
labels = kmeans.fit_predict(X_scaled)

# =========================
# Results — Safe label alignment (fixes your error)
# =========================
st.markdown("## Results")

# Create scored with CLUSTER aligned by index to deduped 'work'
scored = df.copy()
scored["CLUSTER"] = np.nan  # placeholder
scored.loc[work.index, "CLUSTER"] = labels  # assign labels only to rows used in clustering

# Summary & metrics
sizes = pd.Series(labels).value_counts().sort_index()
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.metric("Chosen K", k)
with col_r2:
    st.metric("Silhouette", "N/A" if best_sil is None else f"{best_sil:.3f}")
with col_r3:
    st.metric("Segments (sizes)", ", ".join([f"C{int(i)}:{int(v)}" for i, v in sizes.items()]))

# Cluster profile table (means on available features) — only rows with cluster labels
profile = (
    scored.loc[scored["CLUSTER"].notna()]
    .groupby("CLUSTER")[avail_feats]
    .mean()
    .round(2)
)
st.subheader("Cluster Profiles (Mean by Feature)")
st.dataframe(profile)

# =========================
# Cluster Descriptions (BA-style)
# =========================
st.subheader("Cluster Descriptions")

# z-score across clusters per feature (relative comparison)
if not profile.empty:
    prof_z = (profile - profile.mean()) / (profile.std(ddof=0).replace(0, 1))

    def describe_cluster(cid, row):
        desc = []
        # Credit capacity & penetration
        if "AVG_CREDIT_LIMIT" in row.index and row["AVG_CREDIT_LIMIT"] > 0.5:
            desc.append("high average credit limit")
        if "TOTAL_CREDIT_CARDS" in row.index and row["TOTAL_CREDIT_CARDS"] > 0.5:
            desc.append("multiple cards / deeper product penetration")
        # Channel behaviors
        if "TOTAL_VISITS_ONLINE" in row.index and row["TOTAL_VISITS_ONLINE"] > 0.5:
            desc.append("digitally active (more online visits)")
        if "TOTAL_VISITS_BANK" in row.index and row["TOTAL_VISITS_BANK"] > 0.5:
            desc.append("branch-heavy (more bank visits)")
        if "TOTAL_CALLS_MADE" in row.index and row["TOTAL_CALLS_MADE"] > 0.5:
            desc.append("service-heavy (more calls)")
        if not desc:
            # fallback: top 2 extremes
            top2 = row.abs().sort_values(ascending=False).head(2).index.tolist()
            desc.append("not strongly differentiated; highest on " + ", ".join(top2))
        return " · ".join(desc)

    for cid, row in prof_z.iterrows():
        st.markdown(f"**Cluster C{int(cid)}** — {describe_cluster(cid, row)}")
else:
    st.info("No non-null cluster labels to describe (check data after dedupe).")

# =========================
# Auto Observations
# =========================
st.subheader("Observations (Auto-generated)")

obs = []
# From describe
desc_stats = numeric_work[avail_feats].describe().T

# Credit limit skew cue
if "AVG_CREDIT_LIMIT" in desc_stats.index:
    mean_lim = desc_stats.loc["AVG_CREDIT_LIMIT", "mean"]
    median_lim = desc_stats.loc["AVG_CREDIT_LIMIT", "50%"]
    if median_lim > 0 and mean_lim > median_lim * 1.3:
        obs.append("Average credit limit appears **right-skewed** (mean ≫ median), suggesting a small group with very high limits.")

# Typical ranges for counts
for col in ["TOTAL_CREDIT_CARDS", "TOTAL_VISITS_BANK", "TOTAL_VISITS_ONLINE", "TOTAL_CALLS_MADE"]:
    if col in desc_stats.index:
        q1 = desc_stats.loc[col, "25%"]; q3 = desc_stats.loc[col, "75%"]
        obs.append(f"**{col}** typically spans **{q1:.0f}–{q3:.0f}** (IQR).")

# Correlation cues
corr_abs = numeric_work[avail_feats].corr(numeric_only=True).abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
top_corr = upper.stack().sort_values(ascending=False).head(5)
if not top_corr.empty:
    txt = "; ".join([f"{a}–{b} ({v:.2f})" for (a, b), v in top_corr.items()])
    obs.append(f"Strongest correlations among features: {txt}.")

# Profile contrasts
if not profile.empty:
    for f in avail_feats:
        hi = int(profile[f].idxmax())
        lo = int(profile[f].idxmin())
        if hi != lo:
            obs.append(f"**{f}** is highest in **C{hi}** and lowest in **C{lo}**.")

if obs:
    for o in obs:
        st.markdown(f"- {o}")
else:
    st.markdown("- No notable skews/correlations or profile contrasts detected.")

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
    lines.append(f"- Chosen K: {k}\n")
    lines.append(f"- Silhouette: {'N/A' if best_sil is None else f'{best_sil:.3f}'}\n")
    lines.append("- Cluster sizes: " + ", ".join([f"C{int(i)}:{int(v)}" for i, v in sizes.items()]) + "\n\n")
    lines.append("## Cluster Profiles (Mean)\n")
    if not profile.empty:
        lines.append(profile.to_csv())
    else:
        lines.append("(no profiles — no labels)\n")
    lines.append("\n## Cluster Descriptions\n")
    if not profile.empty:
        for cid, row in prof_z.iterrows():
            lines.append(f"- C{int(cid)} — {describe_cluster(cid, row)}\n")
    lines.append("\n## Observations\n")
    for o in obs:
        lines.append(f"- {o}\n")
    st.download_button(
        "⬇️ Download Results (Markdown)",
        data="".join(lines),
        file_name="clustering_results.md",
        mime="text/markdown",
    )
