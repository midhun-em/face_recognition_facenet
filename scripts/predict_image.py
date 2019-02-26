import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
import cv2
import numpy as np
import facenet
import os
import time
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import sys
img_path='../test_data/yl.jpeg'
modeldir = '../models/20170511-185253.pb'
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
train_img="../data"
plt.ion()

emb_array_full = np.load('../models/emb_array.npy')
emb_array_data = emb_array_full[:,0:128]
emb_array_data.shape

font = cv2.FONT_HERSHEY_SIMPLEX
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		image_size = 182
		input_image_size = 160

		HumanNames = os.listdir(train_img)
		HumanNames.sort()
        
		print('Loading feature extraction model')
		facenet.load_model(modeldir)

		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		embedding_size = embeddings.get_shape()[1]


		print('Start Recognition!')
		frame = cv2.imread(img_path,0)
		frame = facenet.to_rgb(frame)          
		emb_array = np.zeros((1, embedding_size))
		image = frame
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = image[y:y+h, x:x+w]
			final_image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		crop_face = cv2.resize(roi_color, (160, 160))
		crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
		crop_face = crop_face.reshape(-1,input_image_size,input_image_size,3)

		crop_face = facenet.flip(crop_face, False)
		crop_face = facenet.prewhiten(crop_face)                 
		feed_dict = {images_placeholder: crop_face, phase_train_placeholder: False}
		emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
		
sess.close()		
HumanNames = os.listdir(train_img)
HumanNames.sort()
emb_array_label = emb_array_full[:,128:129]
final_label = emb_array_label[euclidean_distances(emb_array_data, emb_array).argmin()]
print('label',final_label)			
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.putText(image,HumanNames[int(final_label)], (10,100), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
cv2.imshow('image',image)
if cv2.waitKey(1000000) & 0xFF == ord('q'):
	sys.exit("Thanks")
	cv2.destroyAllWindows()
