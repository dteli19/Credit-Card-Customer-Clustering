import streamlit as st
import pandas as pd

st.set_page_config(page_title="Your Project", page_icon="📊", layout="wide")
st.title("📊 Your Project — Streamlit Frontend")

st.markdown("This app wraps the core analysis from my Colab notebook into an interactive UI.")

# --- Data input (adjust for your project) ---
uploaded = st.file_uploader("Upload a CSV to analyze", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    # Example: simple summary
    st.subheader("Summary")
    st.write(df.describe())

# --- Links to your notebook in the repo (viewer-friendly) ---
st.markdown("#### Notebook")
st.markdown("- View on GitHub: https://github.com/<your-username>/<your-repo>/blob/main/your_notebook.ipynb")
st.markdown("- View on nbviewer (clean render): https://nbviewer.org/github/<your-username>/<your-repo>/blob/main/your_notebook.ipynb")
