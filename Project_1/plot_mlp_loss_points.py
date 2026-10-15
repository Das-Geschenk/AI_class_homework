"""Plot every recorded run_002 loss value without smoothing."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results' / 'tuning' / 'mlp_model' / 'run_002'

def main():
    data = pd.read_csv(RUN / 'loss_curve.csv')
    if data.empty or data[['iteration', 'loss']].isna().any().any():
        raise ValueError('Loss data are empty or contain missing values.')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Plot all CSV values directly: no averaging, interpolation or smoothing.
    ax.plot(data['iteration'], data['loss'], marker='o', markersize=4,
            linewidth=1.1, color='#2563eb', label='Raw loss (each iteration)')
    ax.set_title('Training Loss - MLP (100,)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training loss')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(RUN / f'loss_curve_points.{extension}', dpi=300, bbox_inches='tight')
    # Refresh the original report image with the same raw point-and-line plot.
    fig.savefig(RUN / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Plotted all {len(data)} raw loss values; no smoothing applied.')

if __name__ == '__main__':
    main()