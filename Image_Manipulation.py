# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:54:23 2024

@author: mrgamarra
"""

import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

##leer una imagen
img_path = r"C:\Users\mrgamarra\Desktop\Clases\PDI\Ejem_imagenes" 
files_names = os.listdir(img_path)
ruta_imagen = img_path + "/" + files_names[0];
input_image = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  
##mostrar una imagen
plt.imshow(input_image)  
plt.title('Imagen original')                                                   
plt.axis('off')
plt.show()

##cambiar el espacio de color
image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) 
plt.imshow(image_gray, cmap='gray')  
plt.title('Escala de grices')                                                   
plt.axis('off')
plt.show()

##recortar la imagen
height = input_image.shape[0]
width = input_image.shape[1]
number_of_channels = input_image.shape[2]
# [rows, columns] 
crop_image = input_image[0:int(height/2), int(width/2):width]
plt.imshow(crop_image)  
plt.title('Imagen recortada')                                                   
plt.axis('off')
plt.show()

##rotar una imagen
im_PIL = Image.fromarray(input_image)
im_45 = im_PIL.rotate(45)
plt.imshow(im_45)  
plt.title('Imagen rotada')                                                   
plt.axis('off')
plt.show()

##cambiar el tamaño
scale_percent = 60 # percent of original size
height2 = int(image_gray.shape[0] * scale_percent / 100)
width2 = int(image_gray.shape[1] * scale_percent / 100)
dim2 = (width2, height2)
im_resized = cv2.resize(image_gray, dim2, interpolation = cv2.INTER_AREA)
plt.imshow(im_resized)  
plt.title('Cambiar tamaño')                                                   
plt.axis('off')
plt.show()

##guardar la imagen modificada
filename2 = img_path + "/" + "Mod_" + files_names[0]
cv2.imwrite(filename2, im_resized)

