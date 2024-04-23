# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:00:40 2024

@author: Margarita G
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

##leer una imagen
img_path = r"C:\Users\Margarita G\Desktop\Clases\Uninorte\PDI\Ejem_images" 
files_names = os.listdir(img_path)
ruta_imagen = img_path + "/" + files_names[0];
input_image = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) 

###Histograma y equalización del histograma
hist1 = cv2.calcHist([image_gray],[0],None,[256],[0,256])
img_2 = cv2.equalizeHist(image_gray)
hist2 = cv2.calcHist([img_2],[0],None,[256],[0,256])
plt.subplot(221),plt.imshow(image_gray, cmap='gray');
plt.axis('off')
plt.subplot(222),plt.plot(hist1);
plt.subplot(223),plt.imshow(img_2, cmap='gray');
plt.axis('off')
plt.subplot(224),plt.plot(hist2);


# Reading the video 
cap = cv2.VideoCapture(0)
while True:
    # Capturing each frame of the video
    ret,frame = cap.read()
    # Muestra el video original
    cv2.imshow('frame', frame)
    
    image_gray_cam = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    image= cv2.equalizeHist(image_gray_cam)
    #image = cv2.cvtColor(img_2cam, cv2.COLOR_GRAY2RGB)
    
    #Muestra el video con histograma ecualizado
    cv2.imshow('Histogram Equalization',image)
    
    #crear el CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_CLAHE = clahe.apply(image_gray_cam)
    
    #Muestra el video con CLAHE
    cv2.imshow('CLAHE',equalized_CLAHE)
    
    ##aplicar CLAHE al canal V de HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    hsv_adjusted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    cv2.imshow('CLAHE in HSV',hsv_adjusted)
    
    ##Presionar la tecla esc para salir
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()    
cv2.destroyAllWindows() 