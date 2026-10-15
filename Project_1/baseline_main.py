"""Run one reproducible text-classification experiment.

To switch algorithms, change MODEL_MODULE to another file in models/.
The data split, TF-IDF transformation, evaluation, and result format stay
unchanged across model comparisons.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存图像，不弹出绘图窗口。
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "train_data.csv"
RESULTS_DIR = PROJECT_DIR / "results"

# Ridge 仅作为 benchmark，最终模型从逻辑回归和 MLP 中选择。
MODEL_MODULE = "ridge_baseline_model"

RANDOM_STATE = 42
VALID_SIZE = 0.2
TFIDF_CONFIG = {
    "max_features": 5000,
}


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    return train_df["text"].astype(str), train_df["target"]


def save_loss_curve(model, output_dir):
    """Save an iteration-loss curve when the selected model exposes one."""
    loss_curve = getattr(model, "loss_curve_", None)
    status_path = output_dir / "loss_curve_status.txt"

    if loss_curve is None:
        message = (
            f"{type(model).__name__} does not expose per-iteration loss history. "
            "No loss curve was generated."
        )
        status_path.write_text(message, encoding="utf-8")
        return None

    pd.DataFrame(
        {"epoch": range(1, len(loss_curve) + 1), "loss": loss_curve}
    ).to_csv(output_dir / "loss_curve.csv", index=False)
    epochs = range(1, len(loss_curve) + 1)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, loss_curve, color="#2563eb", linewidth=2)
    axis.set_title(f"Training Loss - {type(model).__name__}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()

    output_path = output_dir / "loss_curve.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    status_path.write_text(
        f"Loss curve saved with {len(loss_curve)} iterations.",
        encoding="utf-8",
    )
    return output_path


def main():
    texts, labels = load_data()
    # 分层划分保持类别比例；固定种子便于复现。
    x_train, x_valid, y_train, y_valid = train_test_split(
        texts,
        labels,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # 只在训练子集学习词表和 IDF，避免验证集信息泄漏。
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_valid_tfidf = vectorizer.transform(x_valid)

    model_module = importlib.import_module(f"models.{MODEL_MODULE}")
    model = model_module.build_model(random_state=RANDOM_STATE)

    start_time = time.perf_counter()
    model.fit(x_train_tfidf, y_train)
    fit_seconds = time.perf_counter() - start_time
    y_pred = model.predict(x_valid_tfidf)

    report = classification_report(
        y_valid,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "model_module": MODEL_MODULE,
        "model_name": model_module.MODEL_NAME,
        "random_state": RANDOM_STATE,
        "train_samples": len(x_train),
        "valid_samples": len(x_valid),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "fit_seconds": round(fit_seconds, 3),
        "accuracy": round(accuracy_score(y_valid, y_pred), 6),
        "macro_f1": round(f1_score(y_valid, y_pred, average="macro"), 6),
        "weighted_f1": round(f1_score(y_valid, y_pred, average="weighted"), 6),
        "tfidf_config": json.dumps(TFIDF_CONFIG, ensure_ascii=False),
    }

    model_results_dir = RESULTS_DIR / MODEL_MODULE
    model_results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(model_results_dir / "metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_valid, y_pred),
        index=[f"true_{label}" for label in sorted(labels.unique())],
        columns=[f"pred_{label}" for label in sorted(labels.unique())],
    ).to_csv(model_results_dir / "confusion_matrix.csv")
    (model_results_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    loss_curve_path = save_loss_curve(model, model_results_dir)

    print(f"模型: {model_module.MODEL_NAME}")
    print(f"训练集: {len(x_train)} 条，验证集: {len(x_valid)} 条")
    print(f"TF-IDF 词表大小: {len(vectorizer.vocabulary_)}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro-F1: {summary['macro_f1']:.4f}")
    print(f"Weighted-F1: {summary['weighted_f1']:.4f}")
    if loss_curve_path is None:
        print("该模型没有可供绘制的逐轮 loss 记录。")
    else:
        print(f"Loss 曲线: {loss_curve_path}")
    print(f"结果已保存到: {model_results_dir}")


if __name__ == "__main__":
    main()
