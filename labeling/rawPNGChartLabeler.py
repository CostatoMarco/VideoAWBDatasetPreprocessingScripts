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



def worker(args):
    from rawPNGChartLabeler import process_folder  # if needed to avoid circular imports or global issues
    folder_path, segment_background, device, get_viz = args
    try:
        print(f"[INFO] Starting: {folder_path}")
        process_folder(folder_path, segment_background, device, get_viz)
        print(f"[INFO] Finished: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Failed on {folder_path}: {e}")


def label_image(image_path, out_csv_file=None, segment_background=False, device='cpu', get_viz=False, out_image_path=None):
    # Load and normalize the image
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Extract the patches into a csv file
    try:
        annotate(image_path, out_csv_file, segment_background, device, get_viz)
    except RuntimeError as e:
        print(f"[ERROR] Error during annotation: {e}")
        return None

    # Read the output csv file
    if out_csv_file is not None:
        df = pd.read_csv(out_csv_file)
    else:
        return None

    # Select the "White" column from the dataframe
    white_triplet = df["White"].values[0]
    white_triplet = np.array([float(x)/255 for x in white_triplet.strip('[]').split(',')], dtype=np.float32)
    
    if get_viz:
        coords_str = df["White_coords"].values[0]
        x_center, y_center = [int(float(x)) for x in coords_str.strip("[]").split(",")]
        

        dot_img = rgb.copy()
        cv2.circle(dot_img, (x_center, y_center), 5, (0, 0, 255), -1)  # Red dot
        
        
        parent_folder = os.path.dirname(out_image_path)
        filename = os.path.basename(out_image_path)
        circle_folder = os.path.join(parent_folder, "circle_marked")
        os.makedirs(circle_folder, exist_ok=True)
        
        out_circle_path = os.path.join(circle_folder, filename)
        cv2.imwrite(out_circle_path, (dot_img * 255).astype(np.uint8))

    # Compute white balance parameters
    # wbParameters = np.divide(np.mean(white_triplet), white_triplet, out=np.zeros_like(white_triplet, dtype=np.float32), where=np.array(white_triplet) != 0)
    # wbParameters = np.divide(white_triplet, np.linalg.norm(white_triplet) , out=np.zeros_like(white_triplet, dtype=np.float32), where=np.array(white_triplet) != 0)
    wbParameters = white_triplet / np.linalg.norm(white_triplet)

    # Apply white balancing
    white_balanced_rgb = rgb / (wbParameters+1e-6)

    # Clip the values to [0, 1]
    white_balanced_rgb = np.clip(white_balanced_rgb, 0, 1)

    # Save the white balanced image if path is provided
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
        if white_triplet is not None and label_file is not None:
            with open(label_file, 'a') as f:
                relative_path = os.path.relpath(image_path, start=os.path.dirname(label_file))
                f.write(f"{relative_path},{white_triplet[0]},{white_triplet[1]},{white_triplet[2]}\n")
        return white_triplet
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None
    
def process_folder(folder_path, segment_background=False, device='cpu', get_viz=False):
    """
    Processes every frame in the folder frame by frame, extracting white balance triplets and saving them to a CSV file.
    """
    raw_png_folder = os.path.join(folder_path, "rgb_png")
    output_base_folder = os.path.join(folder_path, "processed_labeled_RGB_png")
    output_image_folder = os.path.join(output_base_folder, "frames")
    output_csv_folder = os.path.join(output_base_folder, "CSVs")
    
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_csv_folder, exist_ok=True)
    
    video_label_file = os.path.join(folder_path, "labels.csv")
   
    with open(video_label_file, 'w') as f:
        f.write("frame,red,green,blue\n")  # CSV header
        
        
    for filename in os.listdir(raw_png_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(raw_png_folder, filename)
            
            frame_name = filename.replace(".png", "")
            out_image_path = os.path.join(output_image_folder, f"{frame_name}.png")
            out_csv_file = os.path.join(output_csv_folder, f"{frame_name}.csv")
            process_frame(
                image_path, 
                out_csv_file=out_csv_file, 
                segment_background=segment_background, 
                device=device,
                get_viz=get_viz,
                out_image_path=out_image_path,
                label_file=video_label_file)
            
            
            
def process_all_folders_in_directory(root_path, segment_background=False, device='cpu', get_viz=False):
    """
    Finds all folders in the root_path and applies process_folder() to each one in parallel.
    """
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    args_list = [
        (folder_path, segment_background, device, get_viz)
        for folder_path in folder_paths
    ]

    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        pool.map(worker, args_list)
    
    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python rawPNGChartLabeler.py <path_to_base_directory>")
        sys.exit(1)
        
    filename = sys.argv[1]
    process_all_folders_in_directory(filename, device='cuda', get_viz=True, segment_background=False)
    

    
        
    
    

