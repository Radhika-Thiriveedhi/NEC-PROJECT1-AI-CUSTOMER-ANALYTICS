from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_purchase_model(df):
    """
    Train purchase response prediction model
    """

    features = [
        "Income",
        "Recency",
        "Total_Spending",
        "Total_Purchases",
        "NumWebVisitsMonth"
    ]

    X = df[features]
    y = df["Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions)
    }

    return model, metrics, X_test, y_test, predictions