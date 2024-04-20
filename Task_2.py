# Herath HMKDB
# EG/2019/3601

import numpy as np
import matplotlib.pyplot as plt
import cv2

def region_growing(img, seeds, thresh):
    # Get the dimensions of the image
    height, width = img.shape

    # Initialize the segmented output image
    segmented = np.zeros_like(img, dtype=bool)

    # List of pixels that need to be examined
    pixel_list = list(seeds)

    # Process each pixel in the list
    while pixel_list:
        x, y = pixel_list.pop(0)
        # Mark the current pixel as segmented
        segmented[x, y] = True

        # Check the 4 neighbors (left, right, up, down)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            # Check if the neighbor is within the image bounds
            if 0 <= nx < height and 0 <= ny < width:
                # Check if the neighbor should be added to the region
                if not segmented[nx, ny] and abs(int(img[nx, ny]) - int(img[x, y])) <= thresh:
                    segmented[nx, ny] = True
                    pixel_list.append((nx, ny))

    return segmented

# Load an image in grayscale mode
image_path = 'Image_Task_2.2.jpg' 
gray_img = cv2.imread(image_path, 0) 
cv2.imshow('Original image',gray_img)

# Define seeds and threshold
#seed_points = [(162, 162)] 
#seed_points = [(250, 180)] 
seed_points = [(320, 270)] 
threshold = 2

# Perform region growing
segmented_image = region_growing(gray_img, seed_points, threshold)

# Convert the boolean array to an image to see the result (255 for True, 0 for False)
result_image = (segmented_image * 255).astype(np.uint8)

# We will use matplotlib to show the image here
cv2.imshow('Result image',result_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()