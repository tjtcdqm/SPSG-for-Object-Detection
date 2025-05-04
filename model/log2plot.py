import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def plot_log(log_path, save_path=None):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    df = pd.read_csv(log_path, sep='\t')
    df = df[df['split'] == 'test']

    epochs = df['epoch'].astype(int)
    mAP_gt_50 = pd.to_numeric(df['GT_mAP@0.5'], errors='coerce')
    mAP_bb_50 = pd.to_numeric(df['BB_mAP@0.5'], errors='coerce')
    mAP_gt_95 = pd.to_numeric(df.get('GT_mAP@.50:.95', None), errors='coerce')
    mAP_bb_95 = pd.to_numeric(df.get('BB_mAP@.50:.95', None), errors='coerce')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mAP_gt_50, label='GT mAP@0.5', marker='o', color='blue')
    plt.plot(epochs, mAP_bb_50, label='BB mAP@0.5', marker='x', color='green')

    if mAP_gt_95.notnull().any():
        plt.plot(epochs, mAP_gt_95, label='GT mAP@.50:.95', marker='s', linestyle='--', color='orange')
    if mAP_bb_95.notnull().any():
        plt.plot(epochs, mAP_bb_95, label='BB mAP@.50:.95', marker='d', linestyle='--', color='red')

    plt.xlabel('Epoch')
    plt.ylabel('mAP (%)')
    plt.title('Proxy mAP@0.5 and mAP@[.50:.95] (GT & BB)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mAP logs")
    parser.add_argument('--log', type=str, required=True, help='Path to log file')
    parser.add_argument('--out', type=str, help='Save figure to path (e.g., output.png)')
    args = parser.parse_args()

    plot_log(args.log, args.out)
