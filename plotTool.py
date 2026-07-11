import json
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrices(path, title, labels, matrix):
    matrix = np.array(matrix)

    # Plot raw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{title} - Raw Counts')
    plt.tight_layout()
    plt.savefig(f'{path}_raw.png')
    plt.close()

    # Row-normalized (percentages by actual class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_by_row = (100 * matrix) / row_sums
    annot_row = np.char.mod('%.1f%%', normalized_by_row)

    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_by_row, annot=annot_row, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{title} - Row Normalized (%)')
    plt.tight_layout()
    plt.savefig(f'{path}_row_normalized.png')
    plt.close()

    # Total-normalized (percentages over whole matrix)
    total_sum = matrix.sum()
    normalized_total = (100 * matrix) / total_sum
    annot_total = np.char.mod('%.1f%%', normalized_total)

    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_total, annot=annot_total, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{title} - Total Normalized (%)')
    plt.tight_layout()
    plt.savefig(f'{path}_total_normalized.png')
    plt.close()

    # Hybrid heatmap: Raw counts + Row-normalized colors
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_by_row = matrix / row_sums  # scale for coloring

    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_by_row, annot=matrix, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{title} - Raw Counts w/ Row-Normalized Color')
    plt.tight_layout()
    plt.savefig(f'{path}_hybrid_row_normalized.png')
    plt.close()




def plot_threshold_curve(path, title, sweep, best_balanced, best_kpi):
    """Two-panel operating curve: detection vs false-alarm, and balance vs threshold."""
    thr = np.array([s['threshold'] for s in sweep])
    det = np.array([s['detected'] for s in sweep]) * 100.0
    fa  = np.array([s['false_alarm'] for s in sweep]) * 100.0
    bal = det - fa

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    lim = max(10.0, fa.max() * 1.1)
    ax1.plot([0, lim], [0, lim], ls=':', color='gray', lw=1)
    ax1.plot(fa, det, color='#2a78d6', lw=2)
    if best_balanced['threshold'] == best_kpi['threshold']:
        points = ((best_balanced, '#2a78d6', 'balanced+KPI', -4),)
    else:
        points = ((best_balanced, '#2a78d6', 'balanced', -4),
                  (best_kpi, '#e34948', 'KPI', -10))
    for best, col, name, dy in points:
        bx, by = best['false_alarm'] * 100.0, best['detected'] * 100.0
        ax1.scatter([bx], [by], s=50, color=col, zorder=4)
        ax1.annotate(f"{name}: thr {best['threshold']:.2f}", (bx, by),
                     xytext=(bx + lim * 0.03 * (-1 if bx > lim * 0.7 else 1), by + dy),
                     fontsize=9, ha='right' if bx > lim * 0.7 else 'left')
    ax1.set_xlabel('False alarms on clean tiles (%)')
    ax1.set_ylabel('Defect tiles detected (%)')
    ax1.set_title(f'{title} - Operating Curve')
    ax1.grid(alpha=0.3)

    ax2.plot(thr, bal, color='#2a78d6', lw=2)
    bb = int(np.argmax(bal))
    ax2.scatter([thr[bb]], [bal[bb]], s=50, color='#2a78d6', zorder=4)
    ax2.set_xlabel('Confidence threshold')
    ax2.set_ylabel('Balance = detected% - false-alarm%')
    ax2.set_title(f'{title} - Balance vs Threshold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{path}_curve.png')
    plt.close()
    print(f'Wrote {path}_curve.png')

def main():
    if len(sys.argv) < 2:
        print("Usage: python confusion_matrix_plotter.py <file1.json> [<file2.json> ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            title = data['title']
            path   = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Processing '{filepath}' -> {path} / '{title}'")
            if 'sweep' in data:
                plot_threshold_curve(path, title, data['sweep'],
                                     data['best_balanced'], data['best_kpi'])
            else:
                plot_confusion_matrices(path, title, data['labels'], data['matrix'])
        except Exception as e:
            print(f"Error processing '{filepath}': {e}")

if __name__ == "__main__":
    main()
