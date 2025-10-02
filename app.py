import streamlit as st
import pandas as pd
import numpy as np

# Viz & ML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Page & Sidebar
# ---------------------------
st.set_page_config(page_title="Credit Card Segmentation", page_icon="📊", layout="wide")

with st.sidebar:
    st.header("About")
    st.caption("Interactive UI built from my Colab notebook.")
    st.markdown("**Contact**")
    st.markdown("- 📧 [dteli@umass.edu](mailto:dteli@umass.edu)")
    st.markdown("- 🔗 [LinkedIn](https://www.linkedin.com/in/dhwani-teli/)")
    st.markdown("- 💼 [GitHub](https://github.com/)")

st.title("📊 Credit Card Customer Segmentation")
st.write("Upload your CSV to explore distributions, correlations, and interactive clustering (KMeans + PCA).")

# ---------------------------
# Upload
# ---------------------------
uploaded = st.file_uploader("Upload a CSV to analyze", type=["csv"])

if not uploaded:
    st.info("No file yet. Upload a CSV with customer features (e.g., BALANCE, PURCHASES, CREDIT_LIMIT...).")
    st.stop()

# Read data
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.success("File loaded successfully ✅")
st.subheader("Preview")
st.dataframe(df.head())

# ---------------------------
# Basic handling
# ---------------------------
numeric_df = df.select_dtypes(include=[np.number]).copy()
if numeric_df.empty:
    st.error("No numeric columns found. Please upload a file with numeric features for clustering.")
    st.stop()

st.markdown(f"**Detected numeric features:** {', '.join(numeric_df.columns[:20])}{' ...' if len(numeric_df.columns)>20 else ''}")

# Optional cleanup: replace inf with nan then fill
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---------------------------
# 1) Histogram (Feature Distribution)
# ---------------------------
st.subheader("Feature Distribution")
feature = st.selectbox("Choose a numeric column:", numeric_df.columns)
fig, ax = plt.subplots()
sns.histplot(numeric_df[feature], bins=30, ax=ax)
ax.set_xlabel(feature)
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------------------
# 2) Correlation Heatmap
# ---------------------------
st.subheader("Correlation Heatmap")
if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="Blues", annot=False, ax=ax)
    st.pyplot(fig)
else:
    st.info("Need at least 2 numeric columns to compute correlations.")

# ---------------------------
# 3) KMeans Clustering + PCA Visualization
# ---------------------------
st.subheader("Customer Segments (via KMeans + PCA)")

# Allow user to choose features used for clustering
default_feats = list(numeric_df.columns)
selected_feats = st.multiselect(
    "Select features for clustering (2+):",
    options=list(numeric_df.columns),
    default=default_feats[: min(8, len(default_feats))]  # cap default to first 8 for speed
)

if len(selected_feats) < 2:
    st.warning("Select at least 2 features to run clustering.")
    st.stop()

X = numeric_df[selected_feats].values

# Scale features (recommended for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Number of clusters (k)", 2, 10, 4)
km = KMeans(n_clusters=k, n_init="auto", random_state=42)
clusters = km.fit_predict(X_scaled)

# PCA to 2D for plotting
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="tab10")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("PCA Projection Colored by Cluster")
st.pyplot(fig)

# Attach clusters to original df for downstream use / download
scored = df.copy()
scored["Cluster"] = clusters

# ---------------------------
# 4) Cluster Profiles (Bar Chart of Means)
# ---------------------------
st.subheader("Cluster Profiles")

# Preferred columns; fallback to first 3 numeric if missing
preferred_cols = [c for c in ["BALANCE", "PURCHASES", "CREDIT_LIMIT"] if c in numeric_df.columns]
if len(preferred_cols) < 3:
    # fallback: pick first 3 numeric columns used for clustering (or from numeric_df)
    fallback_cols = selected_feats[:3] if len(selected_feats) >= 3 else list(numeric_df.columns)[:3]
    profile_cols = fallback_cols
else:
    profile_cols = preferred_cols

st.caption(f"Showing mean values by cluster for: {', '.join(profile_cols)}")

profile_df = scored.groupby("Cluster")[profile_cols].mean().round(2)
st.bar_chart(profile_df)

# ---------------------------
# Download results
# ---------------------------
st.markdown("### Download Scored Data")
st.download_button(
    "Download data with cluster labels (CSV)",
    data=scored.to_csv(index=False),
    file_name="clustered_customers.csv",
    mime="text/csv",
)

# =============================
# Executive Summary & Insights
# =============================

import io
from sklearn.metrics import silhouette_score

st.markdown("---")
st.header("🧭 Executive Summary & Insights")

summary = {}
takeaways = []
observations = []

# --- High-level dataset summary ---
summary["rows"] = len(df)
summary["cols_total"] = df.shape[1]
summary["cols_numeric"] = numeric_df.shape[1]
summary["missing_pct"] = round(100 * (df.isna().mean().mean()), 2)

# --- Clustering summary (if available) ---
summary["k"] = k
summary["cluster_sizes"] = dict(pd.Series(clusters).value_counts().sort_index())
summary["largest_cluster"] = int(pd.Series(clusters).value_counts().idxmax())
summary["silhouette"] = None
try:
    # compute silhouette on the scaled features used for clustering (if >= 2 clusters & > samples)
    if len(np.unique(clusters)) > 1 and X_scaled.shape[0] > len(np.unique(clusters)):
        summary["silhouette"] = round(float(silhouette_score(X_scaled, clusters)), 3)
except Exception:
    pass

# --- Correlations (top pairs) ---
top_corr_pairs = []
try:
    corr_m = numeric_df.corr(numeric_only=True).abs()
    corr_ut = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    corr_pairs = corr_ut.stack().sort_values(ascending=False)
    for (c1, c2), v in corr_pairs.head(5).items():
        top_corr_pairs.append((c1, c2, round(float(v), 3)))
except Exception:
    pass

# --- Skewed features (flag potential transformations) ---
skew_series = numeric_df.skew(numeric_only=True).sort_values(ascending=False)
skewed = [(c, round(float(v), 2)) for c, v in skew_series.head(5).items() if abs(v) >= 1]

# --- Outlier-ish columns (IQR rule of thumb) ---
def iqr_outlier_rate(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return float(((series < lower) | (series > upper)).mean())

outlier_rates = {}
for col in numeric_df.columns[:30]:  # cap for performance
    try:
        outlier_rates[col] = round(iqr_outlier_rate(numeric_df[col].dropna()), 3)
    except Exception:
        pass
top_outlier_cols = sorted(outlier_rates.items(), key=lambda x: x[1], reverse=True)[:5]

# --- Cluster profiles on selected columns (or fallback) ---
profile_cols = profile_df.columns.tolist() if "profile_df" in locals() else []
cluster_profiles = {}
try:
    cluster_profiles = scored.groupby("Cluster")[profile_cols].mean().round(2).to_dict(orient="index") if profile_cols else {}
except Exception:
    pass

# ---------------------------
# Build takeaways (bullet points)
# ---------------------------
# Data health
takeaways.append(f"Data contains **{summary['rows']}** rows and **{summary['cols_total']}** columns "
                 f"({summary['cols_numeric']} numeric). Overall missingness: **{summary['missing_pct']}%**.")

# Clustering quality
if summary["silhouette"] is not None:
    takeaways.append(f"KMeans with **k={summary['k']}** produced a silhouette score of **{summary['silhouette']}**, "
                     "indicating cluster separation quality.")

# Cluster distribution
sizes_text = ", ".join([f"C{cid}: {cnt}" for cid, cnt in summary["cluster_sizes"].items()])
takeaways.append(f"Cluster distribution — {sizes_text}. Largest cluster: **C{summary['largest_cluster']}**.")

# Correlations
if top_corr_pairs:
    corr_text = "; ".join([f"**{a}–{b}** ({v})" for a,b,v in top_corr_pairs])
    takeaways.append(f"Top correlated feature pairs: {corr_text}.")

# Skewness
if skewed:
    skew_text = ", ".join([f"**{c}** (skew={v})" for c, v in skewed[:5]])
    takeaways.append(f"Skewed features detected: {skew_text}. Consider transforms (e.g., log) before modeling.")

# Outliers
if top_outlier_cols:
    out_text = ", ".join([f"**{c}** ({r*100:.1f}% outliers by IQR)" for c, r in top_outlier_cols])
    takeaways.append(f"Potential outlier-heavy columns: {out_text}.")

# Cluster profile narrative (if we have it)
if cluster_profiles and profile_cols:
    # Identify extremes per feature across clusters
    try:
        prof_df = scored.groupby("Cluster")[profile_cols].mean()
        for f in profile_cols:
            max_c = int(prof_df[f].idxmax())
            min_c = int(prof_df[f].idxmin())
            observations.append(f"For **{f}**, **C{max_c}** is highest and **C{min_c}** is lowest.")
    except Exception:
        pass

# ---------------------------
# Show in the app
# ---------------------------
st.subheader("Executive Summary")
st.markdown(
    f"""
- Rows: **{summary['rows']}**, Columns: **{summary['cols_total']}** (numeric: **{summary['cols_numeric']}**)
- Missingness (overall avg): **{summary['missing_pct']}%**
- Clustering: **k = {summary['k']}**, Largest cluster: **C{summary['largest_cluster']}**
{"- Silhouette score: **" + str(summary['silhouette']) + "**" if summary['silhouette'] is not None else ""}
"""
)

st.subheader("Key Takeaways")
for t in takeaways:
    st.markdown(f"- {t}")

st.subheader("Notable Observations")
if observations:
    for o in observations:
        st.markdown(f"- {o}")
else:
    st.markdown("- No extreme cluster differences detected on the selected profile features yet. Try changing the feature set or **k**.")

# ---------------------------
# Download a Markdown report
# ---------------------------
report_lines = []
report_lines.append("# Executive Summary\n")
report_lines.append(f"- Rows: {summary['rows']}, Columns: {summary['cols_total']} (numeric: {summary['cols_numeric']})\n")
report_lines.append(f"- Missingness (overall avg): {summary['missing_pct']}%\n")
report_lines.append(f"- Clustering: k = {summary['k']}, Largest cluster: C{summary['largest_cluster']}\n")
if summary['silhouette'] is not None:
    report_lines.append(f"- Silhouette score: {summary['silhouette']}\n")
report_lines.append("\n# Key Takeaways\n")
for t in takeaways:
    report_lines.append(f"- {t}\n")
report_lines.append("\n# Notable Observations\n")
if observations:
    for o in observations:
        report_lines.append(f"- {o}\n")
else:
    report_lines.append("- (none)\n")

report_md = "".join(report_lines)
st.download_button("⬇️ Download Summary (Markdown)", data=report_md, file_name="summary_report.md", mime="text/markdown")


# ---------------------------
# Notebook & Repo links (optional)
# ---------------------------
with st.expander("Notebook & Repo"):
    st.markdown("- View notebook on GitHub: https://github.com/<your-username>/<your-repo>/blob/main/your_notebook.ipynb")
    st.markdown("- Repo root: https://github.com/<your-username>/<your-repo>")
