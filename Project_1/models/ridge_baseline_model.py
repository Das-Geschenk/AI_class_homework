"""Standard scikit-learn RidgeClassifier baseline."""

from sklearn.linear_model import RidgeClassifier


MODEL_NAME = "RidgeClassifier-baseline"


def build_model(random_state=42, **params):
    return RidgeClassifier(alpha=1.0, **params)
