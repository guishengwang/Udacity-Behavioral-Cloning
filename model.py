import os
import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines=[]
with open("/opt/carnd_p3/data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("# of train samples :",len(train_samples))
print("# of validation samples: ",len( validation_samples))

def generator(samples, batch_size=32,correction=0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '/opt/carnd_p3/data/IMG/'+batch_sample[i].split('/')[-1]
                    image = mpimg.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                         
                    if(i==0):
                        angles.append(angle)
                    elif(i==1):
                        angles.append(angle+correction)
                    elif(i==2):
                        angles.append(angle-correction)
                    
                    images.append(cv2.flip(image,1))
                    if(i==0):
                        angles.append(angle*-1)
                    elif(i==1):
                        angles.append((angle+correction)*-1)
                    elif(i==2):
                        angles.append((angle-correction)*-1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print("X_train.shape:", X_train.shape)
            #print("y_train.shape:", y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

            
# Set our batch size
batch_size=32
correction=0.2
ch, row, col = 3, 160, 320 

# compile and train the model using the generator function
train_generator = generator(train_samples,batch_size,correction)
validation_generator = generator(validation_samples,batch_size,correction) 

    
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Lambda, Dropout, Conv2D, Cropping2D, BatchNormalization, Activation
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model=Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col,ch), output_shape=(row, col,ch)))
model.add(Cropping2D(cropping=((50,20),(0,0)))) 
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
history_object=model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')
print("Model was saved successfully \n\n")

