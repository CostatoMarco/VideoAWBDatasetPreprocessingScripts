import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_bayer_image(path):
    img = Image.open(path).convert("L")  # Ensure grayscale
    return np.array(img, dtype=np.float32)

def downsample_demosaic_gbrg(bayer):
    h, w = bayer.shape
    h2, w2 = h // 2, w // 2

    # Extract 2x2 blocks
    R = bayer[1::2, 0::2]
    B = bayer[0::2, 1::2]
    G1 = bayer[0::2, 0::2]
    G2 = bayer[1::2, 1::2]

    G = (G1 + G2) / 2

    rgb = np.stack([R, G, B], axis=2)
    return rgb

def min_max_normalize(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)

def apply_gamma(img, gamma=2.2):
    return np.power(img, 1.0 / gamma)
def grayworld_white_balance(rgb):
    img = rgb.astype(np.float32)
    avg_rgb = np.mean(img, axis=(0, 1))
    gray_avg = np.mean(avg_rgb)
    #gray_avg = [0.5, 0.5, 0.5]  # Target gray average
    scale = gray_avg / (avg_rgb + 1e-8)
    corrected = img*scale
    print(scale)
    return np.clip(corrected, 0.0, 1.0)

def white_point_white_balance(rgb):
    # Scale each channel so that the max becomes 1.0
    max_vals = np.max(rgb, axis=(0, 1))
    scale = 1.0 / (max_vals + 1e-8)  # Avoid division by zero
    print(scale)
    return rgb * scale



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
    
        
    
    

def process_bayer_image(path):
    bayer = load_bayer_image(path)
    bayer = white_black_normalization(bayer, path)
    rgb = downsample_demosaic_gbrg(bayer)
    rgb = grayworld_white_balance(rgb)
    #rgb = white_point_white_balance(rgb)
    rgb = apply_gamma(rgb, 2.0)
    return rgb

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.title("Processed Image")
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "C:\\Users\\marco\\Desktop\\TesiAWB\\VideoProvaFrames\\008-VIDEO_25mm-250430_103600.0\\raw_png\\frame_000080_raw.png"  # Replace with your actual image path
    processed = process_bayer_image(image_path)
    show_image(processed)
