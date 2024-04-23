# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:50:02 2024

@author: Margarita G
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)

##leer una imagen
img_path = r"C:\Users\Margarita G\Desktop\Clases\Uninorte\PDI\Ejem_images" 
files_names = os.listdir(img_path)
ruta_imagen = img_path + "/" + files_names[0];
input_image = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

###Crear un elemento estructural
element = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])
# element = np.array([[1,1,1],
#                     [1,1,1],
#                     [1,1,1]])
plt.imshow(element, cmap='gray');
plt.show()

###Crear una imagen binaria
circle_image = np.zeros((25, 40))
circle_image[disk((12, 12), 8)] = 1
circle_image[disk((12, 28), 8)] = 1
for x in range(20):
   circle_image[np.random.randint(25), np.random.randint(40)] = 1
plt.imshow(circle_image, cmap='gray');

###aplicar erosión y dilatación con el elemento estructural
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].imshow(erosion(circle_image, element), cmap='gray');
ax[0].set_title('Eroded Image')
ax[1].imshow(dilation(circle_image, element), cmap='gray')
ax[1].set_title('Dilated Image')

####Operaciones de opening y closing
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].imshow(opening(circle_image, element), cmap='gray');
ax[0].set_title('Opened Image')
ax[1].imshow(closing(circle_image, element), cmap='gray')
ax[1].set_title('Closed Image')
plt.show()

###convertir a uint8 para utilizar la librería cv2
###Morphological gradient
circle_image=circle_image.astype(np.uint8) 
kernel = element.astype(np.uint8)
gradient = cv2.morphologyEx(circle_image, cv2.MORPH_GRADIENT, kernel)
plt.imshow(gradient, cmap='gray');
plt.show()

tophat = cv2.morphologyEx(circle_image, cv2.MORPH_TOPHAT, kernel)
plt.imshow(tophat, cmap='gray');
plt.show()

blackhat = cv2.morphologyEx(circle_image, cv2.MORPH_BLACKHAT, kernel)
plt.imshow(blackhat, cmap='gray');
plt.show()