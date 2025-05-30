# Video AWB Dataset: Preprocessing Scripts

Collection of scripts used to process RAW video frames for the creation of a video dataset for the Automatic White Balancing task.

Part of my master's thesis at the Imaging and Vision Laboratory @ unimib.

How to use:

1. Load the .mcraw files to be decoded into a parent directory
2. Inside MCRawDecoder\decoder\motioncam-decoder, build and run the example file by adding the directory containing the .mcraw files. Instructions on how to do so are contained inside the folder's README.md file.
3. The previous step will produce a series of files containing the .dng frames of each video. Move them inside a common directory
4. Run the /dng-to-png/dng2png.py file by adding the directory containing all the files produced in the previous step.
5. Inside each video folder, a new folder named "raw_png" will be created containing all raw images in .png format and a .json file containing usefull metadata
6. Modifiy the rawpng2RGBpng.py file by including the processing steps needed and select a target output folder
7. Run the rawpng2RGBpng.py file by adding the directory passed in step 4.
