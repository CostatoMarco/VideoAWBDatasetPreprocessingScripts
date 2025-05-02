import os
import sys
import rawpy
import imageio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(input_path, output_path):
    try:
        if os.path.exists(output_path):
            return f"[SKIP] Already exists: {os.path.basename(output_path)}"

        with rawpy.imread(input_path) as raw:
            raw_image = raw.raw_image_visible
            raw_image_uint16 = raw_image.astype(np.uint16)
            imageio.imwrite(output_path, raw_image_uint16)

        return f"[OK] Saved: {os.path.basename(output_path)}"
    except Exception as e:
        return f"[ERROR] Failed to process {os.path.basename(input_path)}: {e}"

def process_directory(base_path, max_workers=8):
    # check if it's a valid directory
    if not os.path.isdir(base_path):
        print(f"[ERROR] {base_path} is not a valid directory.")
        return

    # Initialize the futures list for concurrent processing
    futures = []
    # Use ThreadPoolExecutor for concurrent file processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Walk through the directory tree and find all .dng files
        for root, dirs, files in os.walk(base_path):
            dng_files = [f for f in files if f.lower().endswith('.dng')]
            if not dng_files:
                continue
            # Create the output directory for raw PNG files    
            output_dir = os.path.join(root, 'raw_png')
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n[INFO] Processing directory: {root}")
            print(f"[INFO] Found {len(dng_files)} .dng files.")
            #Start a processing job for each .dng file
            for dng_file in dng_files:
                input_path = os.path.join(root, dng_file)
                output_filename = os.path.splitext(dng_file)[0] + '_raw.png'
                output_path = os.path.join(output_dir, output_filename)

                futures.append(executor.submit(process_file, input_path, output_path))
        # Wait for all futures to complete and print results
        print("\n[INFO] All files submitted for processing. Waiting for results...")
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_dng_to_raw_png.py <path_to_base_directory>")
        sys.exit(1)

    base_path = sys.argv[1]
    process_directory(base_path)
