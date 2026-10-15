"""Generate the report figure from recorded experimental CSV files."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parent
TUNING = ROOT / 'results' / 'tuning'
OUTPUT = ROOT / 'results' / 'figures'


def main():
    # Read actual results; locate the loss data for the best validation run.
    lr = pd.read_csv(TUNING / 'logistic_regression_tuning_results.csv')
    lr['C'] = lr['params'].map(lambda p: json.loads(p)['C'])
    lr = lr.sort_values('C')
    mlp = pd.read_csv(TUNING / 'mlp_tuning_results.csv').sort_values('run_index')
    labels = mlp['params'].map(lambda p: str(tuple(json.loads(p)['hidden_layer_sizes'])))
    best = mlp.loc[mlp['macro_f1'].idxmax()]
    loss_path = TUNING / 'mlp_model' / f"run_{int(best['run_index']):03d}" / 'loss_curve.csv'
    loss = pd.read_csv(loss_path)
    if loss.empty:
        raise ValueError('The best MLP run has no recorded loss data.')
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']:
        if name in installed:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    else:
        raise RuntimeError('A Chinese font is required for this figure.')
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.15), layout='constrained')
    for ax in axes:
        ax.grid(axis='y', alpha=0.22)
        ax.set_axisbelow(True)
    ax = axes[0]
    ax.plot(lr['C'], lr['macro_f1'], marker='o', color='#2563eb', linewidth=1.8)
    ax.set_xscale('log')
    ax.set_xticks(lr['C'], [f'{c:g}' for c in lr['C']])
    ax.set_ylim(0.83, 0.932)
    ax.set_title('(a) 逻辑回归参数对比', fontsize=11)
    ax.set_xlabel('C（对数刻度）')
    ax.set_ylabel('验证集 Macro-F1')
    for c, score in zip(lr['C'], lr['macro_f1']):
        ax.annotate(f'{score:.4f}', (c, score), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=8)
    ax = axes[1]
    colors = ['#16a34a' if idx == best.name else '#94a3b8' for idx in mlp.index]
    ax.bar(range(len(mlp)), mlp['macro_f1'], color=colors, width=0.62)
    ax.set_xticks(range(len(mlp)), labels)
    # Match both comparison panels; the truncated F1 axis is shown explicitly.
    ax.set_ylim(0.83, 0.932)
    ax.set_title('(b) MLP 隐藏层结构对比', fontsize=11)
    ax.set_xlabel('隐藏层结构（绿色为最佳）')
    ax.set_ylabel('验证集 Macro-F1')
    for i, score in enumerate(mlp['macro_f1']):
        ax.text(i, score + 0.003, f'{score:.4f}', ha='center', fontsize=8)
    ax = axes[2]
    ax.plot(loss['iteration'], loss['loss'], color='#dc2626', linewidth=1.8)
    ax.set_title('(c) 最佳 MLP 训练损失', fontsize=11)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('训练损失')
    ax.set_ylim(bottom=0)
    ax.annotate(f"{len(loss)} 轮；最终损失 {loss['loss'].iloc[-1]:.4f}",
                xy=(0.97, 0.92), xycoords='axes fraction', ha='right', fontsize=8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        path = OUTPUT / f'report_comparison.{suffix}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()