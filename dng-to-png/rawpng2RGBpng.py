import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool

def load_bayer_image(path):
    img = Image.open(path).convert("L")  # Ensure grayscale
    return np.array(img, dtype=np.float32)

def downsample_demosaic_gbrg(bayer):
    R = bayer[1::2, 0::2]
    B = bayer[0::2, 1::2]
    G1 = bayer[0::2, 0::2]
    G2 = bayer[1::2, 1::2]
    G = (G1 + G2) / 2
    rgb = np.stack([R, G, B], axis=2)
    return rgb

def apply_dynamic_gamma(img, min_gamma=0.5, max_gamma=5.0):
    luminance = np.mean(img, axis=2)
    mean_luminance = np.mean(luminance)

    if mean_luminance <= 0:
        gamma = 1.0  # fallback
    else:
        gamma = np.log(0.5) / np.log(mean_luminance + 1e-8)
        gamma = np.clip(gamma, min_gamma, max_gamma)

    print(f"[INFO]Dynamic gamma applied: {gamma:.3f}")
    return np.clip(np.power(img, 1.0 / gamma), 0.0, 1.0)



def apply_percentile_gamma(img, target=0.95, percentile=90, min_gamma=0.5, max_gamma=8.0):
    luminance = np.mean(img, axis=2)
    p = np.percentile(luminance, percentile)

    if p <= 0:
        gamma = 1.0  # fallback
    else:
        gamma = np.log(target) / np.log(p + 1e-8)
        gamma = np.clip(gamma, min_gamma, max_gamma)

    print(f"[INFO]Percentile luminance: {p:.4f}, Gamma applied: {gamma:.3f}")
    return np.clip(np.power(img, 1.0 / gamma), 0.0, 1.0)


def apply_brightening_gamma(img, target_brightness=0.5, min_gamma=0.5, max_gamma=5.0):
    luminance = np.mean(img, axis=2)
    median_luminance = np.median(luminance)

    # If image is very dark, apply stronger gamma boost
    if median_luminance <= 0:
        gamma = 1.0
    else:
        gamma = np.log(target_brightness) / np.log(median_luminance + 1e-8)
        gamma = np.clip(gamma, min_gamma, max_gamma)

    print(f"[INFO]Median luminance: {median_luminance:.4f}, Gamma applied: {gamma:.3f}")
    return np.clip(np.power(img, 1.0 / gamma), 0.0, 1.0)


def apply_auto_brightening_gamma(img, target_percentile=90, target_output=0.8, min_gamma=0.5, max_gamma=3.0):
    luminance = np.mean(img, axis=2)
    p_val = np.percentile(luminance, target_percentile)

    # If the image is very dark, p_val will be low, which should cause gamma < 1 (brightening)
    if p_val <= 0:
        gamma = 1.0
    else:
        gamma = np.log(target_output) / np.log(p_val + 1e-8)

    gamma = np.clip(gamma, min_gamma, max_gamma)

    print(f"[INFO]90th percentile luminance: {p_val:.4f}, Gamma applied: {gamma:.3f}")

    return np.clip(np.power(img, 1.0 / gamma), 0.0, 1.0)



def grayworld_white_balance(rgb):
    img = rgb.astype(np.float32)
    avg_rgb = np.mean(img, axis=(0, 1))
    gray_avg = np.mean(avg_rgb)
    scale = gray_avg / (avg_rgb + 1e-8)
    corrected = img * scale
    return np.clip(corrected, 0.0, 1.0)

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
    norm[0::2, 0::2] = (raw[0::2, 0::2] - G_bl) / (G_wh - G_bl)
    norm[0::2, 1::2] = (raw[0::2, 1::2] - B_bl) / (B_wh - B_bl)
    norm[1::2, 0::2] = (raw[1::2, 0::2] - R_bl) / (R_wh - R_bl)
    norm[1::2, 1::2] = (raw[1::2, 1::2] - G_bl) / (G_wh - G_bl)
    
    return np.clip(norm, 0.0,1.0)



def normalize_brightness(img, target_brightness=0.5, percentile=80):
    """
    Normalize image brightness by scaling pixel values so that
    a given percentile of luminance hits the target_brightness.
    """
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


def process_bayer_image(path):
    bayer = load_bayer_image(path)
    bayer = white_black_normalization(bayer, path)
    rgb = downsample_demosaic_gbrg(bayer)
    rgb = grayworld_white_balance(rgb)
    rgb = normalize_brightness(rgb)
   # rgb = apply_auto_brightening_gamma(rgb)
    return rgb

def save_rgb_image(rgb, output_path):
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)
    img.save(output_path)  # Overwrites if file exists
    print(f"[OK] Saved: {output_path}")
    

def process_single_image(image_path):
    print(f"[INFO] Processing: {image_path} on process {os.getpid()}")
    try:
        rgb = process_bayer_image(image_path)
        raw_png_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(raw_png_dir)
        output_dir = os.path.join(parent_dir, "processed_RGB_png")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path).replace("_raw", "")
        output_path = os.path.join(output_dir, filename)
        save_rgb_image(rgb, output_path)
        #print(f"Saved: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")

def process_all_images(parent_dir):
    tasks = []
    for subdir, dirs, files in os.walk(parent_dir):
        if os.path.basename(subdir) == "raw_png":
            png_files = [f for f in files if f.endswith(".png")]
            for png in png_files:
                full_path = os.path.join(subdir, png)
                tasks.append(full_path)

    with Pool() as pool:
        pool.map(process_single_image, tasks)

# Entry point
if __name__ == "__main__":
    parent_directory = "C:\\Users\\marco\\Desktop\\TesiAWB\\VideoProvaFrames"
    process_all_images(parent_directory)

