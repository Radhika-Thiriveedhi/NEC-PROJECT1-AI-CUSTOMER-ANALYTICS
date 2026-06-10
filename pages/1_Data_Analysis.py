import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data

st.title("📊 Data Analysis")

# Load Data
df = load_data()
df = preprocess_data(df)

# -------------------------------
# Dataset Overview
# -------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Column Information
# -------------------------------

st.header("Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(column_info)

# -------------------------------
# Missing Values
# -------------------------------

st.header("Missing Values Analysis")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(missing_df)

fig = px.bar(
    missing_df,
    x="Column",
    y="Missing Values",
    title="Missing Values by Column"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Statistical Summary
# -------------------------------

st.header("Statistical Summary")

st.dataframe(df.describe())

# -------------------------------
# Correlation Heatmap
# -------------------------------

st.header("Correlation Heatmap")

numeric_df = df.select_dtypes(include=["number"])

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False,
    ax=ax
)

st.pyplot(fig)

# -------------------------------
# Income Distribution
# -------------------------------

st.header("Income Distribution")

fig = px.histogram(
    df,
    x="Income",
    nbins=30,
    title="Income Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Age Distribution
# -------------------------------

st.header("Age Distribution")

fig = px.histogram(
    df,
    x="Age",
    nbins=25,
    title="Age Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Recency Distribution
# -------------------------------

st.header("Recency Distribution")

fig = px.histogram(
    df,
    x="Recency",
    nbins=20,
    title="Recency Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Campaign Response Analysis
# -------------------------------

st.header("Campaign Response Analysis")

response_counts = df["Response"].value_counts().reset_index()
response_counts.columns = ["Response", "Count"]

fig = px.pie(
    response_counts,
    names="Response",
    values="Count",
    title="Campaign Response Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Education Distribution
# -------------------------------

st.header("Education Distribution")

education_counts = (
    df["Education"]
    .value_counts()
    .reset_index()
)

education_counts.columns = ["Education", "Count"]

fig = px.bar(
    education_counts,
    x="Education",
    y="Count",
    title="Education Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Marital Status Distribution
# -------------------------------

st.header("Marital Status Distribution")

marital_counts = (
    df["Marital_Status"]
    .value_counts()
    .reset_index()
)

marital_counts.columns = ["Marital Status", "Count"]

fig = px.bar(
    marital_counts,
    x="Marital Status",
    y="Count",
    title="Marital Status Distribution"
)

st.plotly_chart(fig, use_container_width=True)