import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ColorChartLocalization.annotate import annotate
import os
import torch
import torchvision
from multiprocessing import Pool, cpu_count



def slerp(v0,v1,t):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    dot = np.clip(np.dot(v0,v1), -1.0,1.0)
    omega = np.arccos(dot)
    if omega < 1e-6:
        return v0
    sin_omega = np.sin(omega)
    return (np.sin((1-t) * omega)/sin_omega) * v0 + (np.sin(t*omega) / sin_omega) * v1

def worker(args):
    from rawPNGChartLabeler import process_folder  # if needed to avoid circular imports or global issues
    folder_path, segment_background, device, get_viz, max_hole_size = args
    try:
        print(f"[INFO] Starting: {folder_path}")
        process_folder(folder_path, segment_background, device, get_viz, max_hole_size)
        print(f"[INFO] Finished: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Failed on {folder_path}: {e}")


def label_image(image_path, out_csv_file=None, segment_background=False, device='cpu', get_viz=False, out_image_path=None, wb_override=None):
    # Load and normalize the image
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]

    if wb_override is not None:
        wbParameters = np.array(wb_override, dtype=np.float32)
    else:
        try:
            annotate(image_path, out_csv_file, segment_background, device, get_viz)
        except RuntimeError as e:
            print(f"[ERROR] Error during annotation: {e}")
            return None

        # Read the output csv file
                # Read the output csv file
        if out_csv_file is not None and os.path.exists(out_csv_file):
            df = pd.read_csv(out_csv_file)
        else:
            print(f"[ERROR] Missing CSV output for image: {image_path}")
            return None

        # Filter out valid white entries
        if "White" not in df.columns:
            print(f"[ERROR] 'White' column missing in CSV for image: {image_path}")
            return None

        white_triplets = []
        for val in df["White"].dropna():
            try:
                triplet = np.array([float(x)/255 for x in val.strip('[]').split(',')], dtype=np.float32)
                if triplet.shape == (3,):
                    white_triplets.append(triplet)
            except Exception as e:
                print(f"[WARN] Skipping invalid white triplet '{val}' in {image_path}: {e}")

        if not white_triplets:
            print(f"[ERROR] No valid white triplets found in CSV for image: {image_path}")
            return None

        white_triplet = np.mean(white_triplets, axis=0)


        if get_viz and "White_coords" in df.columns:
            try:
                dot_img = rgb.copy()

                for coord_str in df["White_coords"].dropna():
                    try:
                        x_center, y_center = [int(float(x)) for x in coord_str.strip("[]").split(",")]
                        cv2.circle(dot_img, (x_center, y_center), 5, (0, 0, 255), -1)  # Red dot
                    except Exception as coord_e:
                        print(f"[WARN] Skipping invalid coordinate '{coord_str}': {coord_e}")

                parent_folder = os.path.dirname(out_image_path)
                filename = os.path.basename(out_image_path)
                circle_folder = os.path.join(parent_folder, "circle_marked")
                os.makedirs(circle_folder, exist_ok=True)

                out_circle_path = os.path.join(circle_folder, filename)
                cv2.imwrite(out_circle_path, (dot_img * 255).astype(np.uint8))
            except Exception as e:
                print(f"[WARN] Failed to draw viz circles: {e}")

        wbParameters = white_triplet / np.linalg.norm(white_triplet)

    # Apply white balancing
    white_balanced_rgb = rgb / (wbParameters + 1e-6)
    white_balanced_rgb = np.clip(white_balanced_rgb, 0, 1)

    if out_image_path is not None:
        cv2.imwrite(out_image_path, (white_balanced_rgb * 255).astype(np.uint8))

    return wbParameters.tolist()


def process_frame(image_path, out_csv_file=None, segment_background=False, device='cpu', get_viz=False, out_image_path=None, label_file=None):
    """
    Processes a single frame and saves the white balance triplet in the video label csv file
    """
    print(f"[INFO] Processing: {image_path}")
    try:
        white_triplet = label_image(image_path, out_csv_file, segment_background, device, get_viz, out_image_path)
        # if white_triplet is not None and label_file is not None:
        #     with open(label_file, 'a') as f:
        #         relative_path = os.path.relpath(image_path, start=os.path.dirname(label_file))
        #         f.write(f"{relative_path},{white_triplet[0]},{white_triplet[1]},{white_triplet[2]}\n")
        return white_triplet
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None
    
def process_folder(folder_path, segment_background=False, device='cpu', get_viz=False, max_hole_size=60):
    raw_png_folder = os.path.join(folder_path, "rgb_png")
    output_base_folder = os.path.join(folder_path, "processed_labeled_RGB_png")
    output_image_folder = os.path.join(output_base_folder, "frames")
    output_csv_folder = os.path.join(output_base_folder, "CSVs")

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_csv_folder, exist_ok=True)

    video_label_file = os.path.join(folder_path, "labels.csv")

    frame_files = sorted([f for f in os.listdir(raw_png_folder) if f.endswith(".png")])
    frame_labels = []

    for filename in frame_files:
        image_path = os.path.join(raw_png_folder, filename)
        frame_name = filename.replace(".png", "")
        out_image_path = os.path.join(output_image_folder, f"{frame_name}.png")
        out_csv_file = os.path.join(output_csv_folder, f"{frame_name}.csv")

        triplet = process_frame(
            image_path,
            out_csv_file=out_csv_file,
            segment_background=segment_background,
            device=device,
            get_viz=get_viz,
            out_image_path=out_image_path
        )
        frame_labels.append((filename, triplet))

    # SLERP interpolation with hole size check
    interpolated_labels = []
    i = 0
    while i < len(frame_labels):
        frame, label = frame_labels[i]
        if label is not None:
            interpolated_labels.append((frame, label))
            i += 1
            continue

        # Start of a hole
        start_idx = i - 1
        while i < len(frame_labels) and frame_labels[i][1] is None:
            i += 1
        end_idx = i
        num_missing = end_idx - start_idx - 1

        if start_idx >= 0 and end_idx < len(frame_labels) and num_missing <= max_hole_size:
            label_start = np.array(frame_labels[start_idx][1])
            label_end = np.array(frame_labels[end_idx][1])
            for j in range(num_missing):
                t = (j + 1) / (num_missing + 1)
                interp_label = slerp(label_start, label_end, t)
                interp_frame = frame_labels[start_idx + j + 1][0]
                image_path = os.path.join(raw_png_folder, interp_frame)
                out_image_path = os.path.join(output_image_folder, interp_frame)
                label_image(image_path, wb_override=interp_label, out_image_path=out_image_path)
                interpolated_labels.append((interp_frame, interp_label.tolist()))
        else:
            # Hole too large or at boundary — discard
            for j in range(start_idx + 1, end_idx):
                skipped_frame = frame_labels[j][0]
                print(f"[WARN] Skipping frame '{skipped_frame}' due to hole > {max_hole_size} or boundary.")
                interpolated_labels.append((skipped_frame, None))

        if i < len(frame_labels):
            interpolated_labels.append((frame_labels[i][0], frame_labels[i][1]))
            i += 1

    # Save final CSV
    with open(video_label_file, 'w') as f:
        f.write("frame,red,green,blue\n")
        for frame, label in interpolated_labels:
            if label is not None:
                f.write(f"{frame},{label[0]},{label[1]},{label[2]}\n")

            
            
            
def process_all_folders_in_directory(root_path, segment_background=False, device='cpu', get_viz=False, max_hole_size = 60):
    """
    Finds all folders in the root_path and applies process_folder() to each one in parallel.
    """
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    args_list = [
        (folder_path, segment_background, device, get_viz,max_hole_size)
        for folder_path in folder_paths
    ]

    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        pool.map(worker, args_list)
    
    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python rawPNGChartLabeler.py <path_to_base_directory>")
        sys.exit(1)
        
    filename = sys.argv[1]
    process_all_folders_in_directory(filename, device='cuda', get_viz=True, segment_background=False, max_hole_size= 60)
    
