import argparse 
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, './ultralytics')
from ultralytics import YOLO
import matplotlib.pyplot as plt
from find_patches.findpatches import findpatches, get_triplets
from find_patches.det_and_seg import detect, segment
import pandas as pd
import warnings

import ipdb


def annotate(image_path, out_file, segment_background, device, get_viz):
    patches_names = ["Dark Skin", "Light Skin", "Blue Sky", "Foliage", "Blue Flower", "Bluish Green", 
                     "Orange", "Purplish Blue", "Moderate Red", "Purple", "Yellow Green", "Orange Yellow", 
                     "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", 
                     "White","Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black"]


    out_file_name = out_file if out_file is not None else image_path.split(".")[0] + ".csv"
    
    detector = YOLO('ultralytics/runs/detect/train/weights/best.pt') # Initialize the detection model
    segmenter = YOLO('ultralytics/runs/segment/train/weights/best.pt') if segment_background else None # Initialize the segmentation model if needed

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    crops = detect(img, detector, device) # Get the predictions of the detection model

    if len(crops) == 0:
        raise RuntimeError("Error finding patches")
    
    crops = [crops[0]] # For now, only use the first prediction 


    out = []
    for c in crops:
        crop = c[0]
        displacement = c[1]

        try:
            mask = segment(crop, segmenter, device) if segment_background else None # Segment the image if needed
            # # Find the patches
            patches, radius, viz = findpatches(crop, mask=mask, get_viz = get_viz, t1=80, t2=170, tolerance=np.pi/9) # Find the patches in the image

            if get_viz:
                plt.imshow(viz[:,:,::-1])
                plt.show()
        
        except:
            raise RuntimeError("Error finding patches")

        r  = radius*np.max(crop.shape[:2])/256
        patches = [p["coords"]*np.max(crop.shape[:2])/256 + np.array(displacement) for p in patches] # Get the coordinates of the patches and convert them to the original image coordinates
        
        rgb_triplets = get_triplets(img, patches, r) # Get the rgb triplets of the patches

        rgb_triplets = np.array(rgb_triplets).reshape(-1, 3) # Reshape the rgb triplets to a 2D array
        row = {}

        for i, triplet in enumerate(rgb_triplets):
            row[f"{patches_names[i]}"] = triplet.tolist()
        
        out.append(row) # Append the row to the output list

    df = pd.DataFrame(out) # Create a dataframe from the output list
    df.to_csv(out_file_name, index=False) # Save the dataframe to a csv file

if __name__ == "__main__":
    
    # USAGE EXAMPLE: # python annotate.py path/to/image.jpg --segment_background --device cuda --get_viz
    
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description="Annotate an image with patches")
    parser.add_argument("image_path", type=str, help="Path to the image to annotate")
    parser.add_argument("--segment_background", action="store_true", help="Segment the background of the image")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for the models (cpu or cuda)")
    parser.add_argument("--get_viz", action="store_true", help="Get visualization of the patches")
    parser.add_argument("--out_file",  default=None, help="Output file to save the annotations")
    args = parser.parse_args()
    
    rgb_triplets = annotate(args.image_path, args.out_file, args.segment_background, args.device, args.get_viz)
    



        


    