import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_wb_labels(folder_path):
    csv_path = os.path.join(folder_path, 'labels.csv')
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"'labels.csv' not found in folder: {folder_path}")
    
    df = pd.read_csv(csv_path)
    
    expected_cols = {'frame', 'red', 'green', 'blue'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"'labels.csv' must contain columns: {expected_cols}")
    
    return df

def plot_illuminant_timeline(df, save_path=None):
    df_sorted = df.copy()
    df_sorted['frame_index'] = df_sorted['frame'].str.extract(r'(\d+)').astype(int)
    df_sorted = df_sorted.sort_values('frame_index')

    wb_array = df_sorted[['red', 'green', 'blue']].values
    rgb_colors = wb_array / wb_array.max(axis=1, keepdims=True)  # normalize per frame

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow([rgb_colors], aspect='auto')
    ax.set_xticks(np.linspace(0, len(df_sorted)-1, 10))
    ax.set_xticklabels(np.linspace(0, len(df_sorted)-1, 10, dtype=int))
    ax.set_yticks([])
    ax.set_title("Estimated Illuminant Chromaticity Over Frames")
    ax.set_xlabel("Frame Index")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
        plt.close()
    else:
        plt.show()

def process_all_folders(root_folder):
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, 'labels.csv')
            if os.path.isfile(csv_path):
                try:
                    df = load_wb_labels(subfolder_path)
                    save_path = os.path.join(subfolder_path, 'illuminant_timeline.png')
                    plot_illuminant_timeline(df, save_path)
                except Exception as e:
                    print(f"❌ Failed to process {subfolder_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate illuminant timeline plots for all folders containing labels.csv.")
    parser.add_argument("root_folder", type=str, help="Path to the top-level folder containing video subfolders.")
    args = parser.parse_args()

    if not os.path.isdir(args.root_folder):
        print(f"Error: '{args.root_folder}' is not a valid directory.")
        return
    
    process_all_folders(args.root_folder)

if __name__ == "__main__":
    main()