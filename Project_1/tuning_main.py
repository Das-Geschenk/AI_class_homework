"""Run a parameter search for one standard model at a time.

Examples:
    python tuning_main.py --model logistic_regression
    python tuning_main.py --model mlp
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存图像，不弹出绘图窗口。
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from experiment_config import (
    MODEL_MODULES,
    PARAM_GRIDS,
    RANDOM_STATE,
    TFIDF_CONFIG,
    VALID_SIZE,
)


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "train_data.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "tuning"


def load_split_and_features():
    data = pd.read_csv(TRAIN_PATH)
    texts = data["text"].astype(str)
    labels = data["target"]
    # 所有参数组合使用相同的分层划分和随机种子。
    x_train, x_valid, y_train, y_valid = train_test_split(
        texts,
        labels,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    # 仅在训练子集拟合 TF-IDF，再转换验证集。
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_valid_tfidf = vectorizer.transform(x_valid)
    return x_train_tfidf, x_valid_tfidf, y_train, y_valid, len(vectorizer.vocabulary_)


def parameter_combinations(grid):
    names = list(grid)
    values = [grid[name] for name in names]
    # 笛卡尔积生成全部搭配；每次提供一个参数字典。
    for combination in itertools.product(*values):
        yield dict(zip(names, combination))


def save_loss_curve(model, output_dir):
    loss_curve = getattr(model, "loss_curve_", None)
    if loss_curve is None:
        return
    pd.DataFrame(
        {"iteration": range(1, len(loss_curve) + 1), "loss": loss_curve}
    ).to_csv(output_dir / "loss_curve.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(range(1, len(loss_curve) + 1), loss_curve, linewidth=2)
    axis.set_title("Training Loss - MLPClassifier")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_model(model_module_name, params, x_train, x_valid, y_train, y_valid, vocabulary_size, run_index):
    module = importlib.import_module(f"models.{model_module_name}")
    model = module.build_model(random_state=RANDOM_STATE, **params)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    predictions = model.predict(x_valid)

    run_dir = RESULTS_DIR / model_module_name / f"run_{run_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_module": model_module_name,
        "model_name": module.MODEL_NAME,
        "run_index": run_index,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "random_state": RANDOM_STATE,
        "train_samples": len(y_train),
        "valid_samples": len(y_valid),
        "vocabulary_size": vocabulary_size,
        "fit_seconds": round(fit_seconds, 3),
        "accuracy": round(accuracy_score(y_valid, predictions), 6),
        "macro_f1": round(f1_score(y_valid, predictions, average="macro"), 6),
        "weighted_f1": round(f1_score(y_valid, predictions, average="weighted"), 6),
        "tfidf_config": json.dumps(TFIDF_CONFIG, ensure_ascii=False, sort_keys=True),
    }
    save_loss_curve(model, run_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tune one standard model.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_MODULES))
    args = parser.parse_args()

    model_module_name = MODEL_MODULES[args.model]
    x_train, x_valid, y_train, y_valid, vocabulary_size = load_split_and_features()
    rows = []
    for run_index, params in enumerate(parameter_combinations(PARAM_GRIDS[model_module_name]), start=1):
        print(f"[{run_index}] {model_module_name}: {params}")
        row = run_model(
            model_module_name,
            params,
            x_train,
            x_valid,
            y_train,
            y_valid,
            vocabulary_size,
            run_index,
        )
        rows.append(row)
        print(f"  Accuracy={row['accuracy']:.4f}, Macro-F1={row['macro_f1']:.4f}")

    # 按验证集 Macro-F1 排序，第一行是本模型的最佳配置。
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    model_dir = RESULTS_DIR / model_module_name
    model_dir.mkdir(parents=True, exist_ok=True)
    # CSV 保存全部调参结果，JSON 保存最佳参数与指标。
    model_csv_path = RESULTS_DIR / f"{model_module_name.replace('_model', '')}_tuning_results.csv"
    results.to_csv(model_csv_path, index=False)
    best = results.iloc[0].to_dict()
    (RESULTS_DIR / f"{model_module_name.replace('_model', '')}_best_result.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n最佳参数: {best['params']}")
    print(f"最佳 Macro-F1: {best['macro_f1']:.4f}")
    print(f"汇总 CSV: {model_csv_path}")


if __name__ == "__main__":
    main()
