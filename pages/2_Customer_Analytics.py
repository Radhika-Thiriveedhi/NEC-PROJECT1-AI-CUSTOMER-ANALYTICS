import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data

# ======================
# PAGE TITLE
# ======================

st.title("👥 Customer Analytics")

# ======================
# LOAD DATA
# ======================

df = load_data()
df = preprocess_data(df)

# ======================
# KPI SECTION
# ======================

st.subheader("Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Age",
    round(df["Age"].mean(), 1)
)

col2.metric(
    "Average Income",
    f"${df['Income'].mean():,.0f}"
)

col3.metric(
    "Average Spending",
    round(df["Total_Spending"].mean(), 1)
)

col4.metric(
    "Average Purchases",
    round(df["Total_Purchases"].mean(), 1)
)

st.divider()

# ======================
# CUSTOMER INSIGHTS
# ======================

st.subheader("AI Customer Insights")

top_category = {
    "Wine": df["MntWines"].sum(),
    "Fruits": df["MntFruits"].sum(),
    "Meat": df["MntMeatProducts"].sum(),
    "Fish": df["MntFishProducts"].sum(),
    "Sweets": df["MntSweetProducts"].sum(),
    "Gold": df["MntGoldProds"].sum()
}

top_product = max(top_category, key=top_category.get)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        f"""
        **Top Spending Category**

        {top_product}
        """
    )

with col2:
    st.success(
        f"""
        **Most Common Education**

        {df['Education'].mode()[0]}
        """
    )

with col3:
    st.warning(
        f"""
        **Average Web Visits**

        {round(df['NumWebVisitsMonth'].mean(),1)}
        """
    )

st.divider()

# ======================
# INTERACTIVE FILTER
# ======================

st.subheader("Customer Explorer")

income_threshold = st.slider(
    "Minimum Income",
    int(df["Income"].min()),
    int(df["Income"].max()),
    int(df["Income"].median())
)

filtered_df = df[df["Income"] >= income_threshold]

st.write(
    f"Customers matching filter : {filtered_df.shape[0]}"
)

st.dataframe(
    filtered_df[
        [
            "ID",
            "Age",
            "Income",
            "Total_Spending",
            "Total_Purchases"
        ]
    ].head(20)
)

st.divider()

# ======================
# GRAPHS
# ======================

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

# ---------------------

st.subheader("Age Distribution")

fig = px.histogram(
    df,
    x="Age",
    nbins=25
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------

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

# ---------------------

st.subheader("Purchase Channel Comparison")

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
    y="Purchases"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------

st.subheader("Product Spending Comparison")

spending_df = pd.DataFrame({
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
    spending_df,
    x="Category",
    y="Amount",
    color="Category"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------

st.subheader("Recency vs Spending")

fig = px.scatter(
    df,
    x="Recency",
    y="Total_Spending",
    color="Income"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================
# HIGH VALUE CUSTOMERS
# ======================

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