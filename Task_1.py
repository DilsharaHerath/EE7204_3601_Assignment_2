# Herath HMKDB
# EG/2019/3601

# Importing libraries 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Original Image
img = cv2.imread('Image_Task_1.png', 0)  
img = img/255

# Displaying the Original Image (Gray Scaled)
cv2.imshow('Original Image (Gray Scaled)', img)

# Creating Gaussian noise
x, y = img.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, scale=sigma, size=(x,y))

# Displaying the Gaussian noise
cv2.imshow('Gaussian Noise', n)

# Adding the gaussian noise 
g = img + n

# Plotting the noise added image
cv2.imshow('Noise added image', g)


# Implementing and testing the Otsu’s algorithm

# Converting to uint8 for Otsu's thresholding
g_uint8 = np.uint8(g * 255)

# Applying Otsu's thresholding
ret, otsu_thresholded = cv2.threshold(g_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Displaying the result
cv2.imshow('Otsu Thresholded Image', otsu_thresholded)


# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
