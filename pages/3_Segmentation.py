import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.segmentation import perform_segmentation

# =================================
# PAGE TITLE
# =================================

st.title("🎯 Customer Segmentation")

# =================================
# LOAD DATA
# =================================

df = load_data()
df = preprocess_data(df)

# =================================
# BUTTON
# =================================

if st.button("Perform Segmentation"):

    df, kmeans = perform_segmentation(df)

    # ===============================
    # Cluster Names
    # ===============================

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

    # ===============================
    # KPIs
    # ===============================

    st.subheader("Segmentation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        df.shape[0]
    )

    col2.metric(
        "Number of Segments",
        4
    )

    col3.metric(
        "Premium Customers",
        (df["Segment"] == "Premium Customers").sum()
    )

    st.divider()

    # ===============================
    # AI Insights
    # ===============================

    st.subheader("AI Insights")

    st.success(
        """
        Premium Customers

        High spending and high income customers.
        Suitable for premium membership and loyalty programs.
        """
    )

    st.info(
        """
        High Potential Customers

        Moderate spending customers with growth opportunities.
        """
    )

    st.warning(
        """
        Regular Customers

        Average customers requiring engagement campaigns.
        """
    )

    st.error(
        """
        Budget Customers

        Low spending customers suitable for discounts and offers.
        """
    )

    st.divider()

    # ===============================
    # Segment Statistics
    # ===============================

    st.subheader("Segment Statistics")

    stats_df = (
        df.groupby("Segment")[
            [
                "Income",
                "Total_Spending",
                "Total_Purchases",
                "Recency"
            ]
        ]
        .mean()
        .round(2)
    )

    st.dataframe(
        stats_df,
        use_container_width=True
    )

    st.divider()

    # ===============================
    # Graphs
    # ===============================

    st.subheader("Customer Count by Segment")

    count_df = (
        df["Segment"]
        .value_counts()
        .reset_index()
    )

    count_df.columns = [
        "Segment",
        "Count"
    ]

    fig = px.bar(
        count_df,
        x="Segment",
        y="Count",
        color="Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------

    st.subheader("Segment Distribution")

    fig = px.pie(
        count_df,
        names="Segment",
        values="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------

    st.subheader("Income vs Spending")

    fig = px.scatter(
        df,
        x="Income",
        y="Total_Spending",
        color="Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------

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

    # ------------------------------

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

    # ------------------------------

    st.subheader("Average Purchases by Segment")

    purchase_df = (
        df.groupby("Segment")["Total_Purchases"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        purchase_df,
        x="Segment",
        y="Total_Purchases",
        color="Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===============================
    # Customer Table
    # ===============================

    st.subheader("Segmented Customers")

    st.dataframe(
        df[
            [
                "ID",
                "Income",
                "Total_Spending",
                "Total_Purchases",
                "Segment"
            ]
        ].head(20),
        use_container_width=True
    )

else:

    st.info(
        "Click 'Perform Segmentation' to generate customer segments."
    )