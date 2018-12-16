
# coding: utf-8

# Import libraries
import csv
import cv2
import numpy as np

# Import CSV file 
lines=[]
with open ('data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader: 
        lines.append(line)
        
# fill array "image" with center/left/right images and add/substract correction vector to side images
# fill array "measurements" with steering angles
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1] 
        current_path= 'data/IMG/' + filename
        # print(current_path)
        image = cv2.imread(current_path)
        images.append(image)
        if i==0: 
            measurement = float(line[3])+.35
            measurements.append(measurement)
        elif i==1:
            measurement = float(line[3])
            measurements.append(measurement)
        else:
            measurement = float(line[3])-.35
            measurements.append(measurement)

# make tuple with images and steering angles, flip images for augmentation            
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# label dataset and import keras functions    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, core, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

# Architecture of ConvNet with Normalization and cropping  
model = Sequential()
model.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(32,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Adam optimzer with learning rate of 0.0001, regression model, split into train & val set, shuffle
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 1)

# Save model
model.save('model16.h5')
