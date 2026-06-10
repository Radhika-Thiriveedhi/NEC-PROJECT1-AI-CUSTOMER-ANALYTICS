import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.segmentation import perform_segmentation

# =====================================
# PAGE TITLE
# =====================================

st.title("🤖 AI Recommendation Engine")

# =====================================
# LOAD DATA
# =====================================

df = load_data()
df = preprocess_data(df)

df, _ = perform_segmentation(df)

# =====================================
# SEGMENT NAMES
# =====================================

cluster_summary = (
    df.groupby("Cluster")["Total_Spending"]
    .mean()
    .sort_values()
)

cluster_order = cluster_summary.index.tolist()

cluster_names = {
    cluster_order[0]: "Budget Customers",
    cluster_order[1]: "Regular Customers",
    cluster_order[2]: "High Potential Customers",
    cluster_order[3]: "Premium Customers"
}

df["Segment"] = df["Cluster"].map(cluster_names)

# =====================================
# INPUT
# =====================================

st.subheader("Customer Recommendation")

customer_id = st.number_input(
    "Enter Customer ID",
    min_value=int(df["ID"].min()),
    max_value=int(df["ID"].max()),
    value=int(df["ID"].iloc[0])
)

# =====================================
# BUTTON
# =====================================

if st.button("Generate Recommendation"):

    customer = df[df["ID"] == customer_id]

    if customer.empty:

        st.error("Customer ID not found.")

    else:

        customer = customer.iloc[0]

        income = customer["Income"]
        spending = customer["Total_Spending"]
        purchases = customer["Total_Purchases"]
        segment = customer["Segment"]

        # =====================================
        # CUSTOMER DETAILS
        # =====================================

        st.subheader("Customer Details")

        col1, col2, col3 = st.columns(3)

        col1.metric("Income", round(income))
        col2.metric("Total Spending", round(spending))
        col3.metric("Purchases", round(purchases))

        # =====================================
        # SEGMENT
        # =====================================

        st.subheader("Customer Segment")

        st.success(segment)

        # =====================================
        # RECOMMENDATION
        # =====================================

        st.subheader("AI Recommendation")

        if segment == "Premium Customers":

            st.success(
                """
                Premium Membership Recommended

                • VIP Offers

                • Loyalty Rewards

                • Exclusive Campaigns
                """
            )

            priority = "High"

        elif segment == "High Potential Customers":

            st.info(
                """
                Growth Campaign Recommended

                • Personalized Promotions

                • Seasonal Discounts
                """
            )

            priority = "Medium"

        elif segment == "Regular Customers":

            st.warning(
                """
                Engagement Campaign Recommended

                • Coupons

                • Email Marketing
                """
            )

            priority = "Medium"

        else:

            st.error(
                """
                Discount Campaign Recommended

                • Price Offers

                • Re-engagement Promotions
                """
            )

            priority = "Low"

        st.subheader("Priority Level")

        st.metric(
            "Priority",
            priority
        )

        # =====================================
        # SIMILAR CUSTOMERS
        # =====================================

        st.subheader("Similar Customers")

        features = df[
            [
                "Income",
                "Total_Spending",
                "Total_Purchases",
                "Recency"
            ]
        ]

        similarity_matrix = cosine_similarity(features)

        index = customer.name

        similarities = similarity_matrix[index]

        similar_indices = (
            similarities.argsort()[-6:-1][::-1]
        )

        similar_customers = df.iloc[
            similar_indices
        ][
            [
                "ID",
                "Segment",
                "Income",
                "Total_Spending"
            ]
        ]

        st.dataframe(
            similar_customers,
            use_container_width=True
        )

# =====================================
# SEGMENT DISTRIBUTION
# =====================================

st.subheader("Segment Distribution")

segment_df = (
    df["Segment"]
    .value_counts()
    .reset_index()
)

segment_df.columns = [
    "Segment",
    "Count"
]

fig = px.bar(
    segment_df,
    x="Segment",
    y="Count",
    color="Segment"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# SPENDING BY SEGMENT
# =====================================

st.subheader("Average Spending by Segment")

spending_df = (
    df.groupby("Segment")["Total_Spending"]
    .mean()
    .reset_index()
)

fig = px.bar(
    spending_df,
    x="Segment",
    y="Total_Spending",
    color="Segment"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# INCOME BY SEGMENT
# =====================================

st.subheader("Average Income by Segment")

income_df = (
    df.groupby("Segment")["Income"]
    .mean()
    .reset_index()
)

fig = px.bar(
    income_df,
    x="Segment",
    y="Income",
    color="Segment"
)

st.plotly_chart(
    fig,
    use_container_width=True
)