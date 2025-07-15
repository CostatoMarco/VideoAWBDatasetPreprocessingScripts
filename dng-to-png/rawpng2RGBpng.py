import json
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2
import csv
from collections import defaultdict
WB_METHODS = ["as_shot_neutral", "GW", "WP", "hist_5", "hist_15"]


def load_bayer_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return img

def downsample_demosaic_gbrg(bayer):
    R = bayer[1::2, 0::2]
    B = bayer[0::2, 1::2]
    G1 = bayer[0::2, 0::2]
    G2 = bayer[1::2, 1::2]
    G = (G1 + G2) / 2
    rgb = np.stack([R, G, B], axis=2)
    return rgb





def grayworld_white_balance(rgb, frame_name, method, wb_param):

    gw = rgb.mean(axis=(0,1), keepdims=True)
    gw = gw / np.linalg.norm(gw)
    wb_param[method][frame_name] = gw.flatten().tolist()
    gw_corrected = (rgb / gw).clip(0,1)
    return gw_corrected

def white_patch_white_balance(rgb, frame_name, method, wb_param):
    img = rgb.astype(np.float32)
    max_rgb = np.max(img, axis=(0, 1))
    wb_param[method][frame_name] = max_rgb.flatten().tolist()
    scale = 1.0 / (max_rgb + 1e-8)
    img *= scale
    return np.clip(img, 0.0, 1.0)

def histogram_white_balance(rgb, frame_name, method, wb_param, top_percent=5.0, ):
    img = rgb.astype(np.float32)
    h, w, _ = img.shape
    flat_img = img.reshape(-1, 3)

    # Compute luminance for each pixel
    luminance = np.mean(flat_img, axis=1)

    # Determine the threshold to keep only top X% brightest pixels
    threshold = np.percentile(luminance, 100 - top_percent)

    # Mask the top X% brightest pixels
    top_pixels = flat_img[luminance >= threshold]

    if len(top_pixels) == 0:
        print("[WARN] No pixels selected for white balancing.")
        return rgb

    # Calculate average RGB of selected bright pixels
    avg_rgb = np.mean(top_pixels, axis=0)
    wb_param[method][frame_name] = avg_rgb.flatten().tolist()
    scale = 1.0 / (avg_rgb + 1e-8)

    # Apply scaling
    balanced = img * scale
    return np.clip(balanced, 0.0, 1.0)


def apply_as_shot_neutral_white_balance(rgb, image_path, frame_name, method, wb_param):

    def dng_name_from_png(png_name):
        base = os.path.splitext(png_name)[0]
        if base.endswith('_raw'):
            base = base[:-4]
        return base + '.dng'

    # Load metadata
    dir = os.path.dirname(image_path)
    json_path = os.path.join(dir, "raw_metadata.json")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    target_dng = dng_name_from_png(os.path.basename(image_path))
    entry = next((m for m in metadata if m["frame"] == target_dng), None)
    if entry is None:
        raise ValueError(f"Metadata for {target_dng} not found in {json_path}.")
    
    neutral = entry.get("as_shot_neutral")
    if neutral is None:
        print(f"[WARN] AsShotNeutral not found for {target_dng}. Skipping white balance.")
        return rgb  # fallback: no correction
    
    neutral = np.array(neutral, dtype=np.float32)
    wb_param[method][frame_name] = neutral.flatten().tolist()
    # Normalize relative to green (middle channel)
    scale = neutral[1] / neutral  # G / (R, G, B)
    
    img = rgb.astype(np.float32)
    img[:, :, 0] *= scale[0]  # R
    img[:, :, 1] *= scale[1]  # G
    img[:, :, 2] *= scale[2]  # B

    return np.clip(img, 0.0, 1.0)


def white_black_normalization(rgb, image_path):
    def dng_name_from_png(png_name):
        base = os.path.splitext(png_name)[0]
        if base.endswith('_raw'):
            base = base[:-4]
        return base + '.dng'
    
    dir = os.path.dirname(image_path)
    json_path = os.path.join(dir, "raw_metadata.json")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    target_dng = dng_name_from_png(os.path.basename(image_path))
    entry = next((m for m in metadata if m["frame"] == target_dng), None)
    if entry is None:
        raise ValueError(f"Metadata for {target_dng} not found in {json_path}.")
    black = np.array(entry["black_level_per_channel"], dtype = np.float32)
    white = np.array(entry["white_level_per_channel"], dtype = np.float32)
    R_bl, G_bl, B_bl = black[0], black[1], black[2]
    R_wh, G_wh, B_wh = white[0], white[1], white[2]
    
    raw = rgb.astype(np.float32)
    h, w = raw.shape
    norm = np.zeros_like(raw, dtype=np.float32)
    #print (f"[INFO] Normalizing {target_dng} with black {black} and white {white}")
    norm[0::2, 0::2] = (raw[0::2, 0::2] - G_bl) / (G_wh - G_bl)
    norm[0::2, 1::2] = (raw[0::2, 1::2] - B_bl) / (B_wh - B_bl)
    norm[1::2, 0::2] = (raw[1::2, 0::2] - R_bl) / (R_wh - R_bl)
    norm[1::2, 1::2] = (raw[1::2, 1::2] - G_bl) / (G_wh - G_bl)
    
    return np.clip(norm, 0.0,1.0)



def normalize_brightness(img, target_brightness=0.5, percentile=80):
    img = img.astype(np.float32)
    luminance = np.mean(img, axis=2)
    current = np.percentile(luminance, percentile)
    
    if current < 1e-5:
        scale = 1.0  # prevent division by zero
    else:
        scale = target_brightness / current

    print(f"[INFO]Current brightness (p{percentile}): {current:.4f}, Scale: {scale:.3f}")
    
    img_scaled = np.clip(img * scale, 0, 1.0)
    return img_scaled


def process_bayer_image(rgb,path, frame_name, method, wb_param):
    if method == "GW":
        rgb = grayworld_white_balance(rgb, frame_name, method, wb_param)
    elif method == "WP":
        rgb = white_patch_white_balance(rgb, frame_name, method, wb_param)
    elif method == "hist_5":
        rgb = histogram_white_balance(rgb, frame_name, method,wb_param, top_percent=5.0)
    elif method == "hist_15":
        rgb = histogram_white_balance(rgb, frame_name, method,wb_param, top_percent=15.0)
    elif method == "as_shot_neutral":
        rgb = apply_as_shot_neutral_white_balance(rgb, path, frame_name, method, wb_param)
    else:
        raise ValueError(f"Unsupported WB method: {method}")
    return rgb


def save_rgb_image(rgb, output_path):
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)
    img.save(output_path)  # Overwrites if file exists
    print(f"[OK] Saved: {output_path}")
    
def write_wb_csv(output_folder, method, params):
    if not params:
        print(f"[WARN] No WB data for {method} in {output_folder}")
        return
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, f"white_balance_values_{method}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "red", "green", "blue"])
        for frame, scale in params.items():
            writer.writerow([frame] + scale)
    print(f"[INFO] WB CSV saved to: {csv_path}")
    

def process_single_image(image_path):
    print(f"[INFO] Processing: {image_path} on process {os.getpid()}")
    wb_params = {method: {} for method in WB_METHODS}
    try:
        frame_name = os.path.basename(image_path).replace("_raw", "")
        raw = load_bayer_image(image_path)
        raw = white_black_normalization(raw, image_path)
        rgb_base = downsample_demosaic_gbrg(raw)
        raw_png_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(raw_png_dir)

        for method in WB_METHODS:
            rgb = rgb_base.copy()
            rgb = process_bayer_image(rgb, image_path, frame_name, method, wb_params)
            output_dir = os.path.join(parent_dir, f"processed_{method}_RGB_png")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path).replace("_raw", "")
            output_path = os.path.join(output_dir, filename)
            save_rgb_image(rgb, output_path)

        return (parent_dir, frame_name, wb_params)

    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None




    


def process_all_images(parent_dir):
    tasks = []
    for subdir, dirs, files in os.walk(parent_dir):
        if os.path.basename(subdir) == "raw_png":
            png_files = [f for f in files if f.endswith(".png")]
            for png in png_files:
                full_path = os.path.join(subdir, png)
                tasks.append(full_path)

    # Run in parallel
    with Pool() as pool:
        raw_results = pool.map(process_single_image, tasks)

    # Aggregate white balance values
    grouped_wb_params = defaultdict(lambda: defaultdict(dict))  # video_dir -> method -> frame -> [r, g, b]
    for result in raw_results:
        if result is None:
            continue
        video_dir, frame_name, wb_params = result
        for method, frame_dict in wb_params.items():
            if frame_name in frame_dict:
                grouped_wb_params[video_dir][method][frame_name] = frame_dict[frame_name]

    # Write CSVs
    for video_dir, methods in grouped_wb_params.items():
        for method, frame_data in methods.items():
            output_dir = os.path.join(video_dir, f"processed_{method}_RGB_png")
            write_wb_csv(output_dir, method, frame_data)



# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rawpng2RGBpng.py <path_to_base_directory>")
        sys.exit(1)
    parent_directory = sys.argv[1]
    process_all_images(parent_directory)

