import cv2
import os
from tqdm import tqdm

# Path to the folder containing image frames
frame_folder = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/test_folder6'

# Get a list of all image files in the folder
image_files = [os.path.join(frame_folder, img) for img in os.listdir(frame_folder) if img.endswith(".png")]

# Sort the image files in order
image_files.sort()

# Define the output video file name
output_video = 'output.mp4'

# Open the first image to get dimensions
img = cv2.imread(image_files[0])
height, width, layers = img.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Loop through the image files and write them to the video
for image_file in tqdm(image_files):
    img = cv2.imread(image_file)
    video.write(img)

# Release the video writer
video.release()

print(f"Video saved as {output_video}")