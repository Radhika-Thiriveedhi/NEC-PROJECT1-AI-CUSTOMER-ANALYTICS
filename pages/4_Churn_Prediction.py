import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.churn import train_churn_model

# =====================================
# PAGE TITLE
# =====================================

st.title("⚠️ Customer Churn Prediction")

# =====================================
# LOAD DATA
# =====================================

df = load_data()
df = preprocess_data(df)

model, metrics, X_test, y_test, predictions = train_churn_model(df)

# =====================================
# INPUT SECTION
# =====================================

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Income",
        min_value=0,
        value=int(df["Income"].median())
    )

    spending = st.number_input(
        "Total Spending",
        min_value=0,
        value=int(df["Total_Spending"].median())
    )

    purchases = st.number_input(
        "Total Purchases",
        min_value=0,
        value=int(df["Total_Purchases"].median())
    )

with col2:

    web_visits = st.number_input(
        "Web Visits Per Month",
        min_value=0,
        value=int(df["NumWebVisitsMonth"].median())
    )

    recency = st.number_input(
        "Recency",
        min_value=0,
        value=int(df["Recency"].median())
    )

# =====================================
# BUTTON
# =====================================

if st.button("Predict Churn"):

    input_df = pd.DataFrame({
        "Income": [income],
        "Recency": [recency],
        "Total_Spending": [spending],
        "Total_Purchases": [purchases],
        "NumWebVisitsMonth": [web_visits]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if probability < 0.3:

        st.success(
            f"""
            Low Risk Customer

            Churn Probability: {probability:.2%}
            """
        )

    elif probability < 0.7:

        st.warning(
            f"""
            Medium Risk Customer

            Churn Probability: {probability:.2%}
            """
        )

    else:

        st.error(
            f"""
            High Risk Customer

            Churn Probability: {probability:.2%}
            """
        )

    # =====================================
    # AI INSIGHTS
    # =====================================

    st.subheader("AI Insights")

    if probability > 0.7:

        st.error(
            """
            Customer is likely to leave.

            Recommended Actions:

            • Loyalty Program

            • Discount Coupons

            • Personalized Offers
            """
        )

    elif probability > 0.3:

        st.warning(
            """
            Customer requires engagement.

            Recommended Actions:

            • Promotional Campaigns

            • Email Marketing
            """
        )

    else:

        st.success(
            """
            Customer is stable.

            Maintain customer relationship.
            """
        )

# =====================================
# MODEL PERFORMANCE
# =====================================

st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.2%}"
)

c2.metric(
    "Precision",
    f"{metrics['precision']:.2%}"
)

c3.metric(
    "Recall",
    f"{metrics['recall']:.2%}"
)

c4.metric(
    "F1 Score",
    f"{metrics['f1_score']:.2%}"
)

# =====================================
# FEATURE IMPORTANCE
# =====================================

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    color="Feature"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# CONFUSION MATRIX
# =====================================

st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]

fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted Active", "Predicted Churn"],
    y=["Actual Active", "Actual Churn"],
    colorscale="Blues"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# CHURN DISTRIBUTION
# =====================================

st.subheader("Customer Churn Distribution")

count_df = (
    df["Churn"]
    .value_counts()
    .reset_index()
)

count_df.columns = [
    "Churn",
    "Count"
]

count_df["Churn"] = count_df["Churn"].replace(
    {
        0: "Active",
        1: "Churn"
    }
)

fig = px.pie(
    count_df,
    names="Churn",
    values="Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)