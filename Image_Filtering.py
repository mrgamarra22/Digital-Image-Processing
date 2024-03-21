# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:36:17 2024

@author: mrgamarra
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt 
from PIL import Image


###Función para mostrar las imágenes
def plot_image(image_1, image_2,title_1,title_2):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.axis('off')
    plt.show()

##leer una imagen
img_path = r"C:\Users\mrgamarra\Desktop\Clases\PDI\Ejem_imagenes" 
files_names = os.listdir(img_path)
ruta_imagen = img_path + "/" + files_names[0];
input_image = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
#input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  


##agregar ruido a una imagen

####RUIDO GAUSSIANO
# Generar ruido gaussiano con las mismas dimensiones de la imagen original
rows, cols,_= input_image.shape
gaussian_noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)

# Agregar el ruido a la imagen
img_noised = input_image + gaussian_noise

#Mostrar las imágenes
plot_image(input_image, img_noised,"Imagen Original","Imagen con ruido gaussiano")

##Filtro promedio
img_blurred = cv2.blur(img_noised, (10, 10), -1)
plot_image(img_noised,img_blurred,"Imagen con ruido gaussiano","Filtro promedio")

##Frequency domain filters
domainFilter = cv2.edgePreservingFilter(input_image, flags=1, sigma_s=60, sigma_r=0.6) 
plot_image(input_image,domainFilter,"Imagen Original","Edge preserving")


gaussBlur = cv2.GaussianBlur(input_image,(5,5),cv2.BORDER_DEFAULT)
plot_image(input_image,gaussBlur,"Imagen Original","Filtro gaussiano")

##cálculo de PSNR
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
 
value_noise = PSNR(input_image, img_noised) 
psnr = cv2.PSNR(input_image, img_noised)
print(f"PSNR value is {value_noise} dB") 
print(psnr)

value_filtered = PSNR(img_blurred, img_noised) 
print(f"PSNR value is {value_filtered} dB") 