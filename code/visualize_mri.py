import nibabel as nib
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


path = 'dataset/n171_smwp1/smwp10001_T1.nii'
img = nib.load(path).get_fdata()
img.shape

# Define the output video filename
output_video = 'output_video.mp4'

# Get image dimensions
height, width = img.shape[0], img.shape[2]

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 10, (width, height))

# Iterate through the slices along the y-axis and write them to the video
for i in range(img.shape[1]):
    # Normalize the slice values to 0-255
    slice_img = (img[:, i, :] - np.min(img[:, i, :])) / (np.max(img[:, i, :]) - np.min(img[:, i, :])) * 255
    # Convert to uint8
    slice_img = slice_img.astype(np.uint8)
    # Convert to BGR format
    slice_img_bgr = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
    # Write the frame to the video
    out.write(slice_img_bgr)

# Release the VideoWriter object
out.release()

print(f'Video saved as {output_video}')