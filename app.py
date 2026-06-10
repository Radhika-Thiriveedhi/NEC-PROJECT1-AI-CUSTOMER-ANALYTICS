import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

df = load_data()
df = preprocess_data(df)

# =====================================
# TITLE
# =====================================

st.title("📊 AI-Driven Customer Intelligence and Behavior Analytics Platform")

st.write(
    """
    Welcome to the AI-powered customer analytics platform.
    Explore customer behavior, segmentation, churn prediction,
    purchase prediction and intelligent recommendations.
    """
)

st.divider()

# =====================================
# KPI CARDS
# =====================================

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        df.shape[0]
    )

with col2:
    st.metric(
        "Average Income",
        f"${df['Income'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Average Spending",
        round(df["Total_Spending"].mean(), 1)
    )

with col4:
    st.metric(
        "Campaign Responders",
        int(df["Response"].sum())
    )

st.divider()

# =====================================
# FEATURE CARDS
# =====================================

st.subheader("Platform Features")

c1, c2, c3 = st.columns(3)

with c1:
    st.success(
        """
        📈 Customer Analytics

        • Customer behavior analysis

        • Purchase trends

        • Spending insights
        """
    )

with c2:
    st.info(
        """
        🎯 Machine Learning

        • Segmentation

        • Churn Prediction

        • Purchase Prediction
        """
    )

with c3:
    st.warning(
        """
        🤖 AI Intelligence

        • Recommendation Engine

        • Similar Customers

        • Business Insights
        """
    )

st.divider()

# =====================================
# AI BUSINESS INSIGHTS
# =====================================

st.subheader("🤖 AI Business Insights")

product_totals = {
    "Wine": df["MntWines"].sum(),
    "Fruits": df["MntFruits"].sum(),
    "Meat": df["MntMeatProducts"].sum(),
    "Fish": df["MntFishProducts"].sum(),
    "Sweets": df["MntSweetProducts"].sum(),
    "Gold": df["MntGoldProds"].sum()
}

top_product = max(product_totals, key=product_totals.get)

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        f"""
        Top Revenue Category

        {top_product}
        """
    )

with col2:
    st.info(
        f"""
        Average Web Visits

        {round(df['NumWebVisitsMonth'].mean(),1)}
        """
    )

with col3:
    st.warning(
        """
        Customer segmentation enables
        targeted marketing campaigns.
        """
    )

st.divider()

# =====================================
# SUMMARY GRAPH 1
# =====================================

st.subheader("Income Distribution")

fig = px.histogram(
    df,
    x="Income",
    nbins=30
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# SUMMARY GRAPH 2
# =====================================

st.subheader("Product Category Spending")

product_df = pd.DataFrame({
    "Category": [
        "Wine",
        "Fruits",
        "Meat",
        "Fish",
        "Sweets",
        "Gold"
    ],
    "Amount": [
        df["MntWines"].sum(),
        df["MntFruits"].sum(),
        df["MntMeatProducts"].sum(),
        df["MntFishProducts"].sum(),
        df["MntSweetProducts"].sum(),
        df["MntGoldProds"].sum()
    ]
})

fig = px.bar(
    product_df,
    x="Category",
    y="Amount",
    color="Category"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# SUMMARY GRAPH 3
# =====================================

st.subheader("Campaign Response Distribution")

response_df = (
    df["Response"]
    .value_counts()
    .reset_index()
)

response_df.columns = [
    "Response",
    "Count"
]

response_df["Response"] = response_df["Response"].replace(
    {
        0: "No Response",
        1: "Responded"
    }
)

fig = px.pie(
    response_df,
    names="Response",
    values="Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# FOOTER
# =====================================

st.divider()

st.success(
    """
    Final Year Major Project

    AI-Driven Customer Intelligence and Behavior Analytics Platform
    """
)