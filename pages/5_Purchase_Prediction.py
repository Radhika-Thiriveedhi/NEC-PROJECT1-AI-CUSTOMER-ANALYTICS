import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from models.purchase import train_purchase_model

# ======================================
# PAGE TITLE
# ======================================

st.title("🛒 Purchase Response Prediction")

# ======================================
# LOAD DATA
# ======================================

df = load_data()
df = preprocess_data(df)

model, metrics, X_test, y_test, predictions = train_purchase_model(df)

# ======================================
# INPUT SECTION
# ======================================

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

# ======================================
# BUTTON
# ======================================

if st.button("Predict Response"):

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

    if prediction == 1:

        st.success(
            f"""
            Customer is likely to respond.

            Response Probability: {probability:.2%}
            """
        )

    else:

        st.error(
            f"""
            Customer is unlikely to respond.

            Response Probability: {probability:.2%}
            """
        )

    # ======================================
    # AI Recommendation
    # ======================================

    st.subheader("Marketing Recommendation")

    if probability > 0.7:

        st.success(
            """
            Premium Campaign Recommended

            • Personalized Email

            • Loyalty Rewards

            • Special Offers
            """
        )

    elif probability > 0.4:

        st.warning(
            """
            Moderate Response Probability

            • Promotional Discounts

            • SMS Campaign

            • Seasonal Offers
            """
        )

    else:

        st.error(
            """
            Low Response Probability

            Avoid expensive campaigns.
            """
        )

# ======================================
# MODEL PERFORMANCE
# ======================================

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

# ======================================
# FEATURE IMPORTANCE
# ======================================

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": abs(model.coef_[0])
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

# ======================================
# CONFUSION MATRIX
# ======================================

st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]

fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted No", "Predicted Yes"],
    y=["Actual No", "Actual Yes"],
    colorscale="Greens"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================
# RESPONSE DISTRIBUTION
# ======================================

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