#!/usr/bin/env python
# coding: utf-8

# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


path = '/datasets/faces/'

def load_train(path):
    labels = pd.read_csv(path + 'labels.csv')
    train_datagen = ImageDataGenerator( validation_split=0.25,
                                        rescale=1/255.#,
                                        #horizontal_flip = True,
                                        #vertical_flip = True,
                                        #rotation_range = 90,
                                        #width_shift_range = 0.2,
                                        #height_shift_range = 0.2
                                      )
    train_gen_flow = train_datagen.flow_from_dataframe(
            dataframe=labels,
            directory= path + 'final_files/',
            x_col='file_name',
            y_col='real_age',
            target_size=(224, 224),
            batch_size=32,
            class_mode='raw',
            subset='training',
            seed=12345) 
    
    return train_gen_flow

def load_test(path):
    labels = pd.read_csv(path + 'labels.csv')
    test_datagen = ImageDataGenerator(validation_split=0.25,
                                      rescale=1/255.)
    test_gen_flow = test_datagen.flow_from_dataframe(
            dataframe=labels,
            directory= path + 'final_files/',
            x_col='file_name',
            y_col='real_age',
            target_size=(224, 224),
            batch_size=32,
            class_mode='raw',
            subset='validation',
            seed=12345)   
    
    return test_gen_flow

def create_model(input_shape):
    optimizer = Adam(lr=0.0005)
    backbone = ResNet50(input_shape=input_shape,
                    weights='imagenet', 
                    include_top=False)

    # замораживаем ResNet50 без верхушки
    # backbone.trainable = False
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100, activation='relu'))

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model 

def train_model(model, train_data, test_data, batch_size=None, epochs=5,
                steps_per_epoch=None, validation_steps=None):
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=(test_data),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model 


# In[ ]:




