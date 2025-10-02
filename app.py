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

# ---------------------------
# Notebook & Repo links (optional)
# ---------------------------
with st.expander("Notebook & Repo"):
    st.markdown("- View notebook on GitHub: https://github.com/<your-username>/<your-repo>/blob/main/your_notebook.ipynb")
    st.markdown("- Repo root: https://github.com/<your-username>/<your-repo>")
