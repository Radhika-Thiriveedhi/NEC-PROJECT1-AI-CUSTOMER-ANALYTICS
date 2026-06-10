from datetime import datetime


def preprocess_data(df):
    """
    Perform data preprocessing and feature engineering
    """

    # Handle missing income values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Age Calculation
    current_year = 2026
    df["Age"] = current_year - df["Year_Birth"]

    # Total Spending
    df["Total_Spending"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    # Total Purchases
    df["Total_Purchases"] = (
        df["NumWebPurchases"]
        + df["NumCatalogPurchases"]
        + df["NumStorePurchases"]
    )

    # Churn Label
    df["Churn"] = df["Recency"].apply(
        lambda x: 1 if x > 60 else 0
    )

    return df