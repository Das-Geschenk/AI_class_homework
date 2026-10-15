"""Multi-layer perceptron model with an accessible training loss curve."""

from sklearn.neural_network import MLPClassifier


MODEL_NAME = "MLPClassifier"


def build_model(random_state=42, **params):
    return MLPClassifier(
        hidden_layer_sizes=params.pop("hidden_layer_sizes", (50,)),
        activation=params.pop("activation", "relu"),
        alpha=params.pop("alpha", 0.0001),
        max_iter=params.pop("max_iter", 100),
        # 用训练数据内部划出的验证数据早停，300 次是上限。
        early_stopping=True,
        n_iter_no_change=10,
        random_state=random_state,
        **params,
    )
