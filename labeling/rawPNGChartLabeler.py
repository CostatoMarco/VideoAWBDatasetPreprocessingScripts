import cv2
import numpy as np
import pandas as pd
from ColorChartLocalization.annotate import annotate
import os
import torch
import torchvision





def label_image(image_path, out_csv_file=None, segment_background=False, device='cpu', get_viz=False, out_image_path=None):
    """
    Labels an image with the correct white balancing triplets
    """
    #load the image
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    #extract the patches into a csv file
    try:
        annotate(image_path, out_csv_file, segment_background, device, get_viz)
    except RuntimeError as e:
        print(f"Error during annotation: {e}")
        return None
    # read the output csv file
    if out_csv_file is not None:
        df = pd.read_csv(out_csv_file)
    else:
        return None
    #select the "White" column from the dataframe
    white_triplet = df["White"].values[0]
    #convert the triplet to a list
    white_triplet = [float(x) for x in white_triplet.strip('[]').split(',')]
    #compute the white balance
    white_balanced_rgb = rgb / np.array(white_triplet, dtype=np.float32)
    #clip the values to [0, 1]
    white_balanced_rgb = np.clip(white_balanced_rgb, 0, 1)
    #save the white balanced image if out_image_path is provided
    if out_image_path is not None:
        cv2.imwrite(out_image_path, (white_balanced_rgb * 255).astype(np.uint8))
    white_triplet_np = np.array(white_triplet, dtype=np.float32)
    reciprocal = np.divide(1.0, white_triplet_np, out=np.zeros_like(white_triplet_np), where=white_triplet_np != 0)
    return reciprocal

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
    
    
if __name__ == "__main__":
    filename = "C:\\Users\\marco\\Desktop\\TesiAWB\\VideoProvaFrames\\008-VIDEO_25mm-250430_103600.0"
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    process_folder(filename, device='cuda', get_viz=False)
    

    
        
    
    

