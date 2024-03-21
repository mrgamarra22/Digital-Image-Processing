# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:05:33 2024

@author: mrgamarra
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt 
from PIL import Image

##leer una imagen
img_path = r"C:\Users\mrgamarra\Desktop\Clases\PDI\Ejem_imagenes" 
files_names = os.listdir(img_path)
ruta_imagen = img_path + "/" + files_names[0];
input_image = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
#input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  
image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) 

hist1 = cv2.calcHist([image_gray],[0],None,[256],[0,256])
img_2 = cv2.equalizeHist(image_gray)
hist2 = cv2.calcHist([img_2],[0],None,[256],[0,256])
plt.subplot(221),plt.imshow(image_gray, cmap='gray');
plt.axis('off')
plt.subplot(222),plt.plot(hist1);
plt.subplot(223),plt.imshow(img_2, cmap='gray');
plt.axis('off')
plt.subplot(224),plt.plot(hist2);