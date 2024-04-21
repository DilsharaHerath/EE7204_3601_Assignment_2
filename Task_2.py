# Herath HMKDB
# EG/2019/3601

# Importing libraries
import numpy as np
import cv2

# Implementing the region growing algorithm for image segmentation for 4 neighbors
# The function takes an image, initial seed points, and a threshold value as input. 

def region_growing_4_neighbor(img, seeds, thresh):
    # Get the dimensions of the image
    height, width = img.shape

    # Initialize the segmented output image and List of pixels that need to be examined
    segmented = np.zeros_like(img, dtype=bool)
    pixel_list = list(seeds)

    # Process each pixel in the list
    while pixel_list:
        x, y = pixel_list.pop(0)
        # Mark the current pixel as segmented
        segmented[x, y] = True

        # Check the 4 neighbors (left, right, up, down)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            # Checking if the neighbor is within the image bounds
            if 0 <= nx < height and 0 <= ny < width:
                # Check if the neighbor should be added to the region
                if not segmented[nx, ny] and abs(int(img[nx, ny]) - int(img[x, y])) <= thresh:
                    segmented[nx, ny] = True
                    pixel_list.append((nx, ny))

    return segmented

# Implementing the region growing algorithm for image segmentation for 8 neighbors

def region_growing_8_neighbor(img, seeds, thresh):
    # Get the dimensions of the image
    height, width = img.shape

    # Initialize the segmented output image and list of pixels that need to be examined
    segmented = np.zeros_like(img, dtype=bool)
    pixel_list = list(seeds)

    # Process each pixel in the list
    while pixel_list:
        x, y = pixel_list.pop(0)
        # Mark the current pixel as segmented
        segmented[x, y] = True

        # Check the 8 neighbors (left, right, up, down, and the 4 diagonals)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy

            # Check if the neighbor is within the image bounds
            if 0 <= nx < height and 0 <= ny < width:
                # Check if the neighbor should be added to the region
                if not segmented[nx, ny] and abs(int(img[nx, ny]) - int(img[x, y])) <= thresh:
                    segmented[nx, ny] = True
                    pixel_list.append((nx, ny))

    return segmented


# Load an image in grayscale mode
gray_img = cv2.imread('Image_Task_2.1.jpg', 0) 
cv2.imshow('Original image',gray_img)

# Define seeds and threshold
seed_points = [(320, 270)] 
#seed_points = [(80, 80)] 
threshold = 2

# Perform region growing
segmented_image = region_growing_4_neighbor(gray_img, seed_points, threshold)

# Converting the boolean array to an image to see the result
result_image = (segmented_image * 255).astype(np.uint8)

# Displaying the result image
cv2.imshow('Result image',result_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()