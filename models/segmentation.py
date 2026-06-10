from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def perform_segmentation(df):
    """
    Perform customer segmentation using K-Means
    """

    features = df[
        [
            "Income",
            "Total_Spending",
            "Recency",
            "Total_Purchases"
        ]
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(scaled_features)

    return df, kmeans