# 调参说明

本实验只调节逻辑回归和 MLP，Ridge 作为 benchmark。

逻辑回归：比较 C=[0.1, 1, 10, 100]，max_iter=2000。C 越小，正则化越强。

MLP：比较 (50,)、(100,)、(200,) 和 (100, 50)，固定 alpha=0.0001、activation='relu'、max_iter=300。开启早停，n_iter_no_change=10，实际训练可以提前结束。内部早停验证集来自训练子集，外部验证集用于参数比较。

配置见 experiment_config.py，早停设置见 models/mlp_model.py，完整运行命令见 README.md。