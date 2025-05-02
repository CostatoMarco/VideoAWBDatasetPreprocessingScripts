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
    # Compute average for each channel
    avg = np.mean(rgb, axis=(0, 1))
    gray_avg = np.mean(avg)
    scale = gray_avg / (avg + 1e-8)  # Avoid division by zero
    return rgb * scale
def white_point_white_balance(rgb):
    # Scale each channel so that the max becomes 1.0
    max_vals = np.max(rgb, axis=(0, 1))
    scale = 1.0 / (max_vals + 1e-8)  # Avoid division by zero
    return rgb * scale

def process_bayer_image(path):
    bayer = load_bayer_image(path)
    rgb = downsample_demosaic_gbrg(bayer)
    rgb = grayworld_white_balance(rgb)
    #rgb = white_point_white_balance(rgb)
    rgb = min_max_normalize(rgb)
    rgb = apply_gamma(rgb, 1.0)
    return rgb

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.title("Processed Image")
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "C:\\Users\\marco\\Desktop\\TesiAWB\\VideoProvaFrames\\008-VIDEO_25mm-250430_103600.0\\raw_png\\frame_000100_raw.png"  # Replace with your actual image path
    processed = process_bayer_image(image_path)
    show_image(processed)
