#!/usr/bin/env python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import os

face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
rootdir = '../data/'
save_path = '../processed_data/'
if not os.path.exists(save_path):
	os.makedirs(save_path)
for subdir, dirs, files in sorted(os.walk(rootdir)):
    for file in sorted(files):
        path = os.path.join(subdir, file)
        
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
        crop_face = cv2.resize(roi_color, (182, 182))
        crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
        folder_name = subdir.split('/')[-1]
        if not os.path.exists(save_path+folder_name+'/'):
            os.makedirs(save_path+folder_name)
        plt.imsave(save_path+folder_name+'/'+file, crop_face, format="jpeg")




