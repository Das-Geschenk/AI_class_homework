"""Standard scikit-learn logistic regression model."""

from sklearn.linear_model import LogisticRegression


MODEL_NAME = "LogisticRegression"


def build_model(random_state=42, **params):
    return LogisticRegression(
        C=params.pop("C", 1.0),
        max_iter=params.pop("max_iter", 1000),
        random_state=random_state,
        **params,
    )
