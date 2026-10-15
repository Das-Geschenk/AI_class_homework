"""Train the selected model on all labeled data and create predictions.csv."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from models.mlp_model import build_model
from experiment_config import RANDOM_STATE, TFIDF_CONFIG

# 根据调参结果手动固定最终配置；此处不会自动读取最佳参数 JSON。
FINAL_MODEL_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "alpha": 0.0001,
    "max_iter": 300,
}
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
def write_model_comparison():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/metrics.csv")):
        row = pd.read_csv(path).iloc[0]
        rows.append({"model": row["model_module"], "experiment": "baseline", "params": "default", "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "weighted_f1": row["weighted_f1"], "fit_seconds": row["fit_seconds"]})
    for path in sorted((RESULTS_DIR / "tuning").glob("*_best_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"model": row["model_module"], "experiment": "tuned_best", "params": row["params"], "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "weighted_f1": row["weighted_f1"], "fit_seconds": row["fit_seconds"]})
    if rows:
        try:
            pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(
                RESULTS_DIR / "model_comparison.csv", index=False
            )
        except PermissionError:
            print("警告：无法写入 results/model_comparison.csv，已跳过该汇总文件。")
def main():
    train = pd.read_csv(PROJECT_DIR / "train_data.csv")
    test = pd.read_csv(PROJECT_DIR / "test_data_unlabeled.csv")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("输入数据包含缺失值。")
    # 最终阶段重新使用全部有标签数据拟合 TF-IDF。
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    x_train = vectorizer.fit_transform(train["text"].astype(str))
    x_test = vectorizer.transform(test["text"].astype(str))
    # 沿用调参时的 MLP 配置，包括训练数据内部的早停划分。
    model = build_model(random_state=RANDOM_STATE, **FINAL_MODEL_PARAMS)
    model.fit(x_train, train["target"])
    predictions = model.predict(x_test)
    if len(predictions) != 2457 or not set(predictions).issubset(set(range(10))):
        raise ValueError("预测结果必须是 2457 个 0 到 9 的标签。")
    # 预测顺序与测试文本一致，提交文件不含表头和索引。
    pd.DataFrame(predictions).to_csv(PROJECT_DIR / "predictions.csv", index=False, header=False)
    write_model_comparison()
    print(f"最终模型: MLPClassifier, 参数: {FINAL_MODEL_PARAMS}")
    print(f"已生成 predictions.csv，共 {len(predictions)} 行。")
if __name__ == "__main__":
    main()
