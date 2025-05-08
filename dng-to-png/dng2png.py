import json
import os
import sys
import rawpy
import imageio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(input_path, output_path):
    try:
        # if os.path.exists(output_path):
        #     return f"[SKIP] Already exists: {os.path.basename(output_path)}"

        with rawpy.imread(input_path) as raw:
            raw_image = raw.raw_image_visible
            raw_image_uint16 = raw_image.astype(np.uint16)
            imageio.imwrite(output_path, raw_image_uint16)
            
            #extract metadata
            black = raw.black_level_per_channel
            white = (raw.camera_white_level_per_channel 
                     if raw.camera_white_level_per_channel is not None 
                     else [raw.white_level]*4)
            metadata = {
                "frame": os.path.basename(input_path),
                "black_level_per_channel": black,
                "white_level_per_channel": white
            }
            
        return metadata
    except Exception as e:
        print(f"[ERROR] Failed to process {os.path.basename(input_path)}: {e}")
        return None

def process_directory(base_path, max_workers=8):
    # check if it's a valid directory
    if not os.path.isdir(base_path):
        print(f"[ERROR] {base_path} is not a valid directory.")
        return

    # Initialize the futures list for concurrent processing

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
            metadata_list = []
            futures = []    
            for dng_file in dng_files:
                input_path = os.path.join(root, dng_file)
                output_filename = os.path.splitext(dng_file)[0] + '_raw.png'
                output_path = os.path.join(output_dir, output_filename)

                futures.append(executor.submit(process_file, input_path, output_path))
                
            print("\n[INFO] All files submitted for processing. Waiting for results...") 
            # Wait for all futures to complete and print results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    metadata_list.append(result)
                    print(f"[OK] Saved: {result["frame"]}")            
            if metadata_list:
                metadata_path = os.path.join(output_dir, "raw_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata_list, f, indent = 2)
                print(f"[INFO] Metadata saved to {metadata_path}")
            
   

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_dng_to_raw_png.py <path_to_base_directory>")
        sys.exit(1)

    base_path = sys.argv[1]
    process_directory(base_path)
