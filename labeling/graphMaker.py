import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import argparse
import math

def load_wb_labels(folder_path):
    csv_path = os.path.join(folder_path, 'labels.csv')
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"'labels.csv' not found in folder: {folder_path}")
    
    df = pd.read_csv(csv_path)
    
    expected_cols = {'frame', 'red', 'green', 'blue'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"'labels.csv' must contain columns: {expected_cols}")
    
    return df

def angular_distance_degrees(vec1, vec2):
    # Both vec1 and vec2 should be normalized vectors
    dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def plot_illuminant_timeline(df, save_path=None, save_with_jumps_path=None, jump_threshold_deg=10.0):
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
        print(f"[OK] Saved plot to: {save_path}")
        plt.close()
    else:
        plt.show()


    normalized = wb_array / np.linalg.norm(wb_array, axis=1, keepdims=True)
    jumps = []
    for i in range(1, len(normalized)):
        angle = angular_distance_degrees(normalized[i - 1], normalized[i])
        if angle > jump_threshold_deg:
            jumps.append(i)

    # Recreate plot with red lines for jumps
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow([rgb_colors], aspect='auto')
    ax.set_yticks([])
    ax.set_title(f"Illuminant Changes (> {jump_threshold_deg}°)")
    ax.set_xlabel("Frame Index")

    for x in jumps:
        ax.axvline(x=x, color='red', linewidth=1)

    ax.set_xticks(np.linspace(0, len(df_sorted)-1, 10))
    ax.set_xticklabels(np.linspace(0, len(df_sorted)-1, 10, dtype=int))
    plt.tight_layout()

    if save_with_jumps_path:
        plt.savefig(save_with_jumps_path, dpi=150)
        print(f"[OK] Saved plot with jump markers to: {save_with_jumps_path}")
        plt.close()
    else:
        plt.show()
        
def plot_illuminant_drift_from_first(df, save_path=None):
    df_sorted = df.copy()
    df_sorted['frame_index'] = df_sorted['frame'].str.extract(r'(\d+)').astype(int)
    df_sorted = df_sorted.sort_values('frame_index')

    wb_array = df_sorted[['red', 'green', 'blue']].values
    normalized = wb_array / np.linalg.norm(wb_array, axis=1, keepdims=True)

    reference = normalized[0]  # First frame
    angles = [0.0]

    for i in range(1, len(normalized)):
        angle = angular_distance_degrees(reference, normalized[i])
        angles.append(angle)

    # Compute display RGB colors (normalize for display)
    rgb_colors = wb_array / wb_array.max(axis=1, keepdims=True)

    # Make segments for LineCollection
    x = np.arange(len(angles))
    y = np.array(angles)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = rgb_colors[:-1]  # color each segment based on the leading point

    # Create colored line
    lc = LineCollection(segments, colors=segment_colors, linewidths=2)

    # Y-axis range: round up to nearest 5
    y_max = int(math.ceil(max(y) / 5.0) * 5)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 5))
    ax.set_title("Angular Distance from First Frame's Illuminant (Colored by Illuminant)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Angular Distance (degrees)")
    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Saved color-coded drift plot to: {save_path}")
        plt.close()
    else:
        plt.show()
        
def plot_drift_from_reference_white(df, reference_white, save_path=None):
    df_sorted = df.copy()
    df_sorted['frame_index'] = df_sorted['frame'].str.extract(r'(\d+)').astype(int)
    df_sorted = df_sorted.sort_values('frame_index')

    wb_array = df_sorted[['red', 'green', 'blue']].values
    normalized = wb_array / np.linalg.norm(wb_array, axis=1, keepdims=True)

    # Compute angular distance to reference white
    angles = [angular_distance_degrees(reference_white, n) for n in normalized]

    # Normalize colors for RGB display
    rgb_colors = wb_array / wb_array.max(axis=1, keepdims=True)

    # Prepare segments
    x = np.arange(len(angles))
    y = np.array(angles)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = rgb_colors[:-1]

    # Create colored line
    lc = LineCollection(segments, colors=segment_colors, linewidths=2)

    # Y-axis range: round up to nearest 5
    y_max = int(math.ceil(max(y) / 5.0) * 5)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 180)
    ax.set_yticks(np.arange(0, 181, 15))
    ax.set_title("Angular Distance from Reference White (Colored by Illuminant)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Angular Distance (degrees)")
    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Saved drift-from-white plot to: {save_path}")
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
                    jump_save_path = os.path.join(subfolder_path, 'illuminant_timeline_with_jumps.png')
                    plot_illuminant_timeline(df, save_path, jump_save_path, 5.0)
                    plot_illuminant_drift_from_first(
                    df,
                    save_path=os.path.join(subfolder_path, 'illuminant_drift_from_first.png')
                    )
                    reference_white = np.array([1.0, 1.0, 1.0])/math.sqrt(3)  # Normalized [1, 1, 1] white vector
                    plot_drift_from_reference_white(
                        df,
                        reference_white,
                        save_path=os.path.join(subfolder_path, 'drift_from_reference_white.png')
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process {subfolder_path}: {e}")

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