"""Shared settings: Ridge benchmark; tune Logistic Regression and MLP."""

RANDOM_STATE = 42

VALID_SIZE = 0.2

TFIDF_CONFIG = {'max_features': 5000}

MODEL_MODULES = {'logistic': 'logistic_regression_model',
 'logistic_regression': 'logistic_regression_model',
 'mlp': 'mlp_model'}

# 只搜索逻辑回归和 MLP；Ridge 基准不参与调参。
PARAM_GRIDS = {'logistic_regression_model': {'C': [0.1, 1.0, 10.0, 100.0],
                               'max_iter': [2000]},
 'mlp_model': {'hidden_layer_sizes': [(50,), (100,), (200,), (100, 50)],
               'alpha': [0.0001],
               'activation': ['relu'],
               'max_iter': [300]}}

