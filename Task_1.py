# Herath HMKDB
# EG/2019/3601

# Importing libraries 
import cv2
import numpy as np
#from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

# Original Image
img = cv2.imread('Task_1_Image.png', 0)  
img = img/255

cv2.imshow('Original Image', img)

#Creating Gaussian noise
x, y = img.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, scale=sigma, size=(x,y))

# Gaussian noise
cv2.imshow('Gaussian Noise', n)

# Adding the gaussian noise 
g = img + n

# Plotting the noise added image
cv2.imshow('Noise added image', g)



# Implementing and testing the Otsu’s algorithm

# Plotting the histogram

# Convert to uint8 for Otsu's thresholding
g_uint8 = np.uint8(g * 255)

# Apply Otsu's thresholding
ret, otsu_thresholded = cv2.threshold(g_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the result
cv2.imshow('Otsu Thresholded Image', otsu_thresholded)


# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

#Plotting the gaussian noise in 3D plot
# x, y = np.meshgrid(np.arange(n.shape[1]), np.arange(n.shape[0]))

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, n, cmap='viridis')

# # Customize plot
# ax.set_title('3D Plot of Gaussian Noise')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Noise')

# # Show plot
# plt.show()