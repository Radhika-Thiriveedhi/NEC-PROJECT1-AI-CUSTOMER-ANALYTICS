import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.segmentation import perform_segmentation

# ==========================================
# PAGE TITLE
# ==========================================

st.title("📈 Enterprise Customer Dashboard")

# ==========================================
# LOAD DATA
# ==========================================

df = load_data()
df = preprocess_data(df)

df, _ = perform_segmentation(df)

# ==========================================
# KPI SECTION
# ==========================================

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

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

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Average Purchases",
        round(df["Total_Purchases"].mean(), 1)
    )

with col5:
    st.metric(
        "Campaign Responders",
        int(df["Response"].sum())
    )

with col6:
    st.metric(
        "Churn Customers",
        int(df["Churn"].sum())
    )

st.divider()

# ==========================================
# AI BUSINESS INSIGHTS
# ==========================================

st.subheader("🤖 AI Business Insights")

highest_product = {
    "Wine": df["MntWines"].sum(),
    "Fruits": df["MntFruits"].sum(),
    "Meat": df["MntMeatProducts"].sum(),
    "Fish": df["MntFishProducts"].sum(),
    "Sweets": df["MntSweetProducts"].sum(),
    "Gold": df["MntGoldProds"].sum()
}

top_product = max(highest_product, key=highest_product.get)

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        f"""
        Top Product Category

        {top_product}
        """
    )

with col2:
    st.info(
        f"""
        Campaign Responders

        {int(df['Response'].sum())}
        """
    )

with col3:
    st.warning(
        f"""
        Average Web Visits

        {round(df['NumWebVisitsMonth'].mean(),1)}
        """
    )

st.divider()

# ==========================================
# INCOME DISTRIBUTION
# ==========================================

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

# ==========================================
# SPENDING DISTRIBUTION
# ==========================================

st.subheader("Total Spending Distribution")

fig = px.histogram(
    df,
    x="Total_Spending",
    nbins=30
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# CAMPAIGN RESPONSE
# ==========================================

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

# ==========================================
# PRODUCT CATEGORY ANALYSIS
# ==========================================

st.subheader("Product Category Analysis")

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

# ==========================================
# WEB PURCHASE CHANNEL ANALYSIS
# ==========================================

st.subheader("Purchase Channel Analysis")

purchase_df = pd.DataFrame({
    "Channel": [
        "Web",
        "Catalog",
        "Store"
    ],
    "Purchases": [
        df["NumWebPurchases"].sum(),
        df["NumCatalogPurchases"].sum(),
        df["NumStorePurchases"].sum()
    ]
})

fig = px.bar(
    purchase_df,
    x="Channel",
    y="Purchases",
    color="Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# TOP CUSTOMERS
# ==========================================

st.subheader("Top 10 High Value Customers")

top_customers = (
    df[
        [
            "ID",
            "Income",
            "Total_Spending",
            "Total_Purchases"
        ]
    ]
    .sort_values(
        by="Total_Spending",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_customers,
    use_container_width=True
)