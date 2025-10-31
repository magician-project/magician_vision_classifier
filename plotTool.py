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



def main():
    if len(sys.argv) < 2:
        print("Usage: python confusion_matrix_plotter.py <file1.json> [<file2.json> ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            title = data['title']
            labels = data['labels']
            matrix = data['matrix']
            path   = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Processing '{filepath}' -> {path} / '{title}'")
            plot_confusion_matrices(path, title, labels, matrix)
        except Exception as e:
            print(f"Error processing '{filepath}': {e}")

if __name__ == "__main__":
    main()
