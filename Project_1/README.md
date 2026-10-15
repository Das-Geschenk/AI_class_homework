# 实验一：英文新闻文本分类

本项目使用 TF-IDF 表示英文新闻文本，对 10 个类别进行分类。RidgeClassifier 作为额外 benchmark，正式调参与最终模型选择仅涉及逻辑回归和 MLP；最终采用验证集 Macro-F1 较高的 MLP。



## 1. 环境与依赖

代码使用 Python 3.10 及以上语法，需安装：

```powershell
python -m pip install numpy pandas scikit-learn matplotlib
```



## 2. 文件说明

| 文件或目录 | 用途 |
|---|---|
| `train_data.csv` | 7368 条有标签文本，包含 `text`、`target` |
| `test_data_unlabeled.csv` | 2457 条无标签文本，仅包含 `text` |
| `experiment_config.py` | 随机种子、验证集比例、TF-IDF 配置与调参范围 |
| `models/` | Ridge、逻辑回归与 MLP 的模型构建函数 |
| `baseline_main.py` | 训练并评价 Ridge benchmark |
| `tuning_main.py` | 对逻辑回归或 MLP 进行参数搜索 |
| `final_predict.py` | 重新训练最终 MLP，生成预测与比较汇总 |
| `plot_mlp_loss_points.py` | 根据最佳 MLP 原始 CSV 绘制逐点损失曲线 |
| `plot_report.py` | 根据结果 CSV 绘制参数比较和损失曲线组合图 |
| `results/` | 验证指标、调参记录与图表 |
| **`predictions.csv`** | **位于根目录下；最终测试集预测标签** |

## 3. 实验设置

- 将有标签数据按 80%/20% 分层划分为训练子集 5894 条、验证集 1474 条，随机种子为 42。
- 使用 `TfidfVectorizer(max_features=5000)`，采用默认小写转换和分词规则，不额外删除停用词、做词干提取或清洗邮件头。
- 验证阶段只在训练子集拟合 TF-IDF，再转换验证集，避免外部验证信息泄漏。
- 所有候选配置使用相同外部划分和特征设置；主要依据验证集 Macro-F1 选择，辅以 Accuracy、Weighted-F1 和模型 fit 时间。
- 测试集没有真实标签，不用于调参，也无法计算测试准确率。

| 模型 | 参数设置 |
|---|---|
| Ridge benchmark | `alpha=1.0`，不参与调参及最终候选选择 |
| 逻辑回归 | `C=[0.1, 1, 10, 100]`，`max_iter=2000` |
| MLP | 隐藏层 `[(50,), (100,), (200,), (100, 50)]`；`activation='relu'`、`alpha=0.0001`、`max_iter=300` |

MLP 开启 `early_stopping=True`、`n_iter_no_change=10`，从训练子集内部再划出验证数据决定停止，外部验证集用于配置比较。300 是最大迭代次数，不保证每次训练满 300 轮。

## 4. ==运行方法==

先进入本目录后，依次运行：

```cmd
REM 1. 运行 Ridge 基准模型
python baseline_main.py

REM 2. 逻辑回归调参
python tuning_main.py --model logistic_regression

REM 3. MLP 调参
python tuning_main.py --model mlp

REM 4. 训练最终 MLP 并生成 predictions.csv
python final_predict.py
```

本目录已包含此前完整实验结果，无需仅为查看结果重新训练。重新执行会覆盖对应输出。



## 5. 输出与最佳参数

- `results/ridge_baseline_model/`：基准指标 `metrics.csv`、分类报告 `classification_report.json`、混淆矩阵 `confusion_matrix.csv`。Ridge 不提供逐轮 loss，因此不生成损失曲线。
- `results/tuning/logistic_regression_tuning_results.csv`、`mlp_tuning_results.csv`：对应模型的全部参数比较结果。
- `results/tuning/logistic_regression_best_result.json`、`mlp_best_result.json`：按验证集 Macro-F1 选出的各模型最佳参数和指标。
- `results/tuning/mlp_model/run_001/` 至 `run_004/`：按配置顺序保存 MLP 原始 `loss_curve.csv` 和曲线图片。
- `results/model_comparison.csv`：执行 `final_predict.py` 后汇总 Ridge 与两个调参模型的最佳验证结果。
- `predictions.csv`：2457 行，无表头、无索引，每行一个 0–9 标签，与测试文本逐行对应。

