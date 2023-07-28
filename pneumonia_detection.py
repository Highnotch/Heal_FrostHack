#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pickle


# In[2]:


IMAGE_SIZE = [224, 224]

train_path = 'train'
valid_path = 'test'


# In[3]:


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[4]:


for layer in vgg.layers:
    layer.trainable = False


# In[5]:


folders = glob('train/*')
x = Flatten()(vgg.output)


# In[6]:


prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()


# In[7]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[8]:


from keras.preprocessing.image import ImageDataGenerator


# In[9]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)


# In[10]:


training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')

valid_set = valid_datagen.flow_from_directory('chest_xray/val',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


# In[ ]:


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(valid_set)
)


# In[ ]:


import tensorflow as tf
from keras.models import load_model

model.save('chest_xray.h5')


# In[ ]:


from keras.models import load_model


# In[ ]:


from keras.preprocessing import image
import tensorflow.keras.utils as utils
from keras.applications.vgg16 import preprocess_input


# In[ ]:


import numpy as np


# In[ ]:


model=load_model('chest_xray.h5')


# In[ ]:


img=utils.load_img('test/NORMAL/NORMAL2-IM-0023-0001.jpeg',target_size=(224,224))
x=utils.img_to_array(img)
x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)


# In[ ]:


result=int(classes[0][0])


# In[ ]:


result


# In[ ]:


if result==0:
    print("Person is affeected by pneumonia")
else:
    print("Result is normal")


# In[ ]:





# In[ ]:





# In[ ]:




