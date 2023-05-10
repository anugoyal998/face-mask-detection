import pickle
import numpy as np
from keras.models import load_model
import cv2

input_image_path = input('Path of the image to be predicted: ')
model = load_model('facemask.h5')
input_image = cv2.imread(input_image_path)
cv2.imshow('img', input_image)
input_image_resized = cv2.resize(input_image, (128,128))
input_image_scaled = input_image_resized/255
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
print(input_prediction)
input_pred_label = np.argmax(input_prediction)
print(input_pred_label)
if input_pred_label == 1:
  print('The person in the image is wearing a mask')
else:
  print('The person in the image is not wearing a mask')
