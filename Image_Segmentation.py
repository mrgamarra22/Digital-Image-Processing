# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:17:18 2024

@author: mrgamarra
"""

# Importing necessary libraries
from skimage import data, img_as_float
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import chan_vese
from skimage.color import label2rgb
from skimage.segmentation import slic


#### Segmentation by Thresholding – Manual Input

"""
# Sample Image of scikit-image package
coffee = data.coffee()
gray_coffee = rgb2gray(coffee)
plt.imshow(gray_coffee, cmap = 'gray')

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

for i in range(10):

    # Iterating different thresholds
    binarized_gray = (gray_coffee > i*0.1)*1
    plt.subplot(5,2,i+1)
    
    # Rounding of the threshold
    # value to 1 decimal point
    plt.title("Threshold: >"+str(round(i*0.1,1)))
    
    # Displaying the binarized image
    # of various thresholds
    plt.imshow(binarized_gray, cmap = 'gray')

plt.tight_layout()
"""



"""
### Segmentation by Thresholding  Using skimage.filters module

# Sample Image of scikit-image package
coffee = data.coffee()
gray_coffee = rgb2gray(coffee)
plt.imshow(gray_coffee, cmap = 'gray')
# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))


# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_coffee)

# Computing binarized values using the obtained
# threshold
binarized_coffee = (gray_coffee > threshold)*1
plt.subplot(2,2,1)
plt.title("Threshold: >"+str(threshold))

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap = "gray")

# Computing Ni black's local pixel
# threshold values for every pixel
threshold = filters.threshold_niblack(gray_coffee)

# Computing binarized values using the obtained 
# threshold
binarized_coffee = (gray_coffee > threshold)*1
plt.subplot(2,2,2)
plt.title("Niblack Thresholding")

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap = "gray")

# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_coffee)
plt.subplot(2,2,3)
plt.title("Sauvola Thresholding")

# Displaying the local threshold values
plt.imshow(threshold, cmap = "gray")

# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_coffee = (gray_coffee > threshold)*1
plt.subplot(2,2,4)
plt.title("Sauvola Thresholding - Converting to 0's and 1's")

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap = "gray")
"""


"""
### Active Contour Segmentation

# Sample Image of scikit-image package
astronaut = data.astronaut()
gray_astronaut = rgb2gray(astronaut)
plt.imshow(gray_astronaut, cmap = 'gray')
plt.axis('off')

# Applying Gaussian Filter to remove noise
gray_astronaut_noiseless = gaussian(gray_astronaut, 1)

# Localising the circle's center at 220, 110
x1 = 220 + 100*np.cos(np.linspace(0, 2*np.pi, 500))
x2 = 100 + 100*np.sin(np.linspace(0, 2*np.pi, 500))

# Generating a circle based on x1, x2
snake = np.array([x1, x2]).T

# Computing the Active Contour for the given image
astronaut_snake = active_contour(gray_astronaut_noiseless,
								snake)

fig = plt.figure(figsize=(10, 10))

# Adding subplots to display the markers
ax = fig.add_subplot(111)

# Plotting sample image
ax.imshow(gray_astronaut_noiseless, cmap = 'gray')

# Plotting the face boundary marker
ax.plot(astronaut_snake[:, 0],
		astronaut_snake[:, 1], 
		'-b', lw=5)

# Plotting the circle around face
ax.plot(snake[:, 0], snake[:, 1], '--r', lw=5)
"""


"""
### Chan-Vese Segmentation
fig, axes = plt.subplots(1, 3, figsize=(10, 10))

# Sample Image of scikit-image package
astronaut = data.astronaut()
gray_astronaut = rgb2gray(astronaut)

# Computing the Chan VESE segmentation technique
chanvese_gray_astronaut = chan_vese(gray_astronaut,
									max_iter=100,
									extended_output=True)

ax = axes.flatten()

# Plotting the original image
ax[0].imshow(gray_astronaut, cmap="gray")
ax[0].set_title("Original Image")

# Plotting the segmented - 100 iterations image
ax[1].imshow(chanvese_gray_astronaut[0], cmap="gray")
title = "Chan-Vese segmentation - {} iterations".format(len(chanvese_gray_astronaut[2]))

ax[1].set_title(title)

# Plotting the final level set
ax[2].imshow(chanvese_gray_astronaut[1], cmap="gray")
ax[2].set_title("Final Level Set")
plt.show()
"""


### SLIC segmentation

# Setting the plot size as 15, 15
plt.figure(figsize=(15,15))

# Sample Image of scikit-image package
astronaut = data.astronaut()


# Applying Simple Linear Iterative
# Clustering on the image
# - 50 segments & compactness = 10
astronaut_segments = slic(astronaut,
						n_segments=50,
						compactness=10)
plt.subplot(1,2,1)

# Plotting the original image
plt.imshow(astronaut)
plt.subplot(1,2,2)

# Converts a label image into
# an RGB color image for visualizing
# the labeled regions. 
plt.imshow(label2rgb(astronaut_segments,
					astronaut,
					kind = 'avg'))
