Face recognition 
----------------  
Packages used 
----------------------------- 
- Python: '3.7.1' 
- Tensorflow: 1.13.0-rc1 
- Opencv: 4.0.0 
- Architecture used : Facenet davidsandberg github  
Steps 
------  
- 1) Download and extract the zip file 
- 2) Data - Images of 3 person in 3 folder (Andrew Ng,Geoffrey Hinton,YanLecunn) 
3) models - All the models are saved in this folder 
4) scripts - All scripts 
5) test data - This folder contain 3 test image of each person and one unknown person  
6) Run preprocess_image.py - It will detect the face from all images and write that in 'processed_data' folder 
7) Train the model for 3 person using train_model.py from folder scripts 8) Run predict_image.py for testing on images of all 3 person and one unknown person.


Inspired from https://github.com/davidsandberg/facenet
