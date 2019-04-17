#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:17 2019

@author: mendel
"""
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.models import load_model


import tensorflow as tf
from tensorflow import keras
Model = keras.models.Model
preprocess_input = keras.applications.inception_v3.preprocess_input
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
from modules.model_zoo_alpha import models_dict
from modules.library import get_img_paths


from modules.Image_Mask_DataGenerator import Image_DataGenerator

# Set some parameters
IMG_WIDTH =  1024
IMG_HEIGHT = 768
IMG_CHANNELS = 3
BATCH_SIZE = 4

TRAIN_PATH   = os.getcwd()+"/dataset/train/"
TEST_PATH = os.getcwd()+"/dataset/test/"




TRAIN_IMAGE_PATH = os.getcwd()+"/dataset/color_memory/original/train/images"
TRAIN_MASK_PATH = os.getcwd()+"/dataset/color_memory/original/train/masks"




TEST_IMAGE_PATH = os.getcwd()+"/dataset/color_memory/original/validate/images"
TEST_MASK_PATH = os.getcwd()+"/dataset/color_memory/original/validate/masks"

PATH   = os.getcwd()+"/temp/"

down_size = 4


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#
#def Preprocess(path ):
#  """Preprocess the image for training
#  It creates trainable numpy arrays and rescaled
#  
#  Parameter: PATH: need a path with 2 Subfolders with names: images, masks
#  
#  """
#  img_height=768
#  img_width=1024
#  img_channels=3 
#  
#  
#  
#  # Get train and test IDs
#  train_ids = [os.path.basename(i) for i in get_img_paths(path+"images")]
#  
#  # Get and resize train images and masks
#  X_train = np.zeros((len(train_ids), img_height, img_width, img_channels), dtype=np.float)
#  Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.float)
#  #Getting and resizing train images and masks
#  
#  sys.stdout.flush()
#  for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#  
#    img = imread(path+"images"  +"/" +id_)[:,:,:img_channels]
#    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
#    X_train[n] = img / 255
#    
#    
#  for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#    mask = imread(path+"masks/" + id_)
#    mask = np.expand_dims(resize(mask, (img_height, img_width), mode='constant', 
#                                      preserve_range=True), axis=-1)
#    Y_train[n] = mask / 255
#  
#  
#  return X_train,Y_train
#


def load_data(path, down_size = 16):
    """
    """
    #loading data
    img_height=768
    img_width=1024
    img_channels=3 
    
    # Get train and test IDs
    train_ids = [os.path.basename(i) for i in get_img_paths(path+"images")]
    
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), int(img_height / down_size), int(img_width/ down_size), img_channels), dtype=np.float)
    Y_train = np.zeros((len(train_ids), int(img_height /down_size), int(img_width/down_size), 1), dtype=np.float)
    i = 0
    
    
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
      try:
        img = imread(path+"images"  +"/" +id_)[:,:,:img_channels]
      except:
        return X_train,Y_train  
      
      
      try:
        mask = imread(path+"masks/" + id_)
      except:
        return X_train,Y_train  
      
      img = resize(img, ( int(img_height / down_size), int(img_width/ down_size)), mode='constant', preserve_range=True)
      mask = np.expand_dims(resize(mask, ( int(img_height / down_size), int(img_width/ down_size)), mode='constant', 
                                        preserve_range=True), axis=-1)
        
      #X_train[n] = img / 255
     # Y_train[n] = mask / 255
      
      X_train[n] = img /255 
      Y_train[n] = mask /255
    
    
    return X_train,Y_train  
  
  
  
down_size = 2


X_train,Y_train = load_data(TRAIN_PATH,down_size)


np.max(Y_train[0])
#train with low dataset

#load the model
img_shape = (int(IMG_HEIGHT / down_size), int(IMG_WIDTH/ down_size),IMG_CHANNELS)
inc_v3_enc_dec_model = models_dict['inc_v3'](img_shape, False, None)



i = 0
for layer in inc_v3_enc_dec_model.layers:
  i += 1
  if i < 31:
    layer.trainable = False
  else:
    layer.trainable = True


##check if layers are trainable
for layer in inc_v3_enc_dec_model.layers:
  print(layer.trainable)
  

inc_v3_enc_dec_model.compile(optimizer='adam',
                             loss='mse', 
                             metrics=['accuracy'])

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('models/unet_model.h5', verbose=1, save_best_only=True)


inc_v3_enc_dec_model.summary()


results = inc_v3_enc_dec_model.fit(X_train, Y_train, validation_split=0.1, batch_size=1, epochs=200, 
                    callbacks=[earlystopper, checkpointer])




#
#
#plt.plot(inc_v3_enc_dec_model.history)
#plt.show()
#
#
#inc_v3_enc_dec_model.save_weights(str(down_size)+'model_weights.h5')
#
#
#X_test,  Y_test  = myGenerator(TEST_PATH, down_size = down_size)
#X_test.shape
#
#len(X_test)
#
#Y_predict = inc_v3_enc_dec_model.predict(X_train)
#
#imshow(X_test[0])
#imshow(np.squeeze(Y_predict[1]))
#plt.show()
#imshow(np.squeeze(Y_test[0]))
#
#
#imshow(X_test[3])
#
#
#for i in range(10):
#  imshow(np.squeeze(Y_predict[i]))
#  plt.show()
#  imshow(np.squeeze(Y_test[i]))
#  plt.show()
#
#
#next(x)[0].shape  
#next(x)[1].shape     
#
#len(next(x))
#
#
#inc_v3_enc_dec_model.fit_generator(x, 
#                                   steps_per_epoch = 9,
#                                   epochs=50
#                                   )

#
#
#plt.imshow(next(x)[0][0])
#plt.imshow(np.squeeze(next(x)[1]))
#
#next(y).shape 


#X = np.zeros( (1150, 768, 1924,3)) 

#
#seed = 42
#
#data_gen_args = dict(horizontal_flip=True,
#                     vertical_flip=True,
#                     rotation_range=90.,
#                     width_shift_range=0.1,
#                     height_shift_range=0.1,
#                     zoom_range=0.1)
#image_datagen = ImageDataGenerator(**data_gen_args)
#mask_datagen = ImageDataGenerator(**data_gen_args)
##image_datagen.fit(xtr, seed=7)
##mask_datagen.fit(ytr, seed=7)
#image_datagen.fit(xtr)
#mask_datagen.fit(ytr)
#image_generator = image_datagen.flow(xtr, batch_size=1, seed=7)
#mask_generator = mask_datagen.flow(ytr, batch_size=1, seed=7)
#
#
#
#
#def gen(image_generator,mask_generator):
#    yield zip(image_generator, mask_generator)
#    
#    
#gen_object = gen(image_generator,mask_generator)  
#
#
#
#
#next(gen_object)
#
#
#for a,b in gen_object:
##  imshow(a[0])
##  plt.show()
##  imshow(b[0])
##  plt.show()
#  print(a.shape)
#  print(b.shape)
#   
##
#
#
#mask_preproc_fn = lambda t : (t[0], np.expand_dims(t[1][:,:,:,0], axis=3))
#
#mapped_train_generator = map(mask_preproc_fn,zip(image_generator, mask_generator))
#
#

#
#
#train_generator = provider.get_train_generator()
#
#

#print(next(train_generator))
#
#
#
#for a,b in train_generator:
##  imshow(a[0])
##  plt.show()
##  imshow(b[0])
##  plt.show()
#  print(a.shape)
#  print(b.shape)
#
#




#
#zipped_train_generator = zip(image_generator, mask_generator)
#
#
#print(*zipped_train_generator)
#
#inc_v3_enc_dec_model.fit_generator(train_generator, 
#                                   steps_per_epoch=len(xtr) ,
#                                   epochs=50)
#
#y = inc_v3_enc_dec_model.predict(xtr[:10])
#
#
#

#for x_batch, y_batch in image_datagen.flow(xtr, ytr, batch_size=1):
#  print(x_batch.shape)
#  imshow(x_batch[0])
#  plt.show()
#  imshow(np.squeeze(y_batch[0]))
#  plt.show()
#  
#  

#
#y[0].shape
#
#ix = random.randint(0, 6)
#imshow(X_train[5])
#plt.show()
#imshow(np.squeeze(y[1]))
#plt.show()
#
#
#
#
#
#for a,b in image_datagen:
##  imshow(a[0])
##  plt.show()
##  imshow(b[0])
##  plt.show()
#  print(a.shape)
#  print(b.shape)
#
#
#
#
#from tensorflow.python.util import nest
#
#
#
#li =nest.flatten(list(zipped_train_generator))[0]
#  

#
#mutants = ['charles xavier',
# 'bobby drake',
# 'kurt wagner',
# 'max eisenhardt',
# 'kitty pryde']
#
#
#aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
#
#
#
#powers = ['telepathy',
# 'thermokinesis',
# 'teleportation',
# 'magnetokinesis',
# 'intangibility']
#
#
#mutant_data = list(zip(mutants, aliases, powers))
#
#
#mutants_zip = zip(mutants, aliases, powers)
#
#
#
#image_generator[0].shape
#
#
#len(image_generator)


#np.array(nest.flatten(image_generator)).shape


#  layer.trainable = False
#  
#len(inc_v3_enc_dec_model.layers)
#

#inc_v3_enc_dec_model.summary()
#len(inc_v3_enc_dec_model.layers)
#

#inc_v3_enc_dec_model.layers[-2].trainable = True  
#inc_v3_enc_dec_model.layers[-1].trainable = True
#  
#  

#
#
#warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
#
##random.seed = seed
##np.random.seed = seed
#def generator(xtr, xval, ytr, yval, batch_size):
#    data_gen_args = dict(horizontal_flip=True,
#                         vertical_flip=True,
#                         rotation_range=90.,
#                         width_shift_range=0.1,
#                         height_shift_range=0.1,
#                         zoom_range=0.1)
#    image_datagen = ImageDataGenerator(**data_gen_args)
#    mask_datagen = ImageDataGenerator(**data_gen_args)
##    image_datagen.fit(xtr, seed=7)
##    mask_datagen.fit(ytr, seed=7)
#    image_datagen.fit(xtr)
#    mask_datagen.fit(ytr)
#    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
#    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
#    train_generator = zip(image_generator, mask_generator)
#
#    val_gen_args = dict()
#    image_datagen_val = ImageDataGenerator(**val_gen_args)
#    mask_datagen_val = ImageDataGenerator(**val_gen_args)
##    image_datagen_val.fit(xval, seed=7)
##    mask_datagen_val.fit(yval, seed=7)
#    image_datagen_val.fit(xval)
#    mask_datagen_val.fit(yval)
#    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
#    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
#    
#    
#    val_generator = zip(image_generator_val, mask_generator_val)
#
#    return train_generator, val_generator

#X_train, Y_train = Preprocess(TRAIN_PATH)
#X_test,  Y_test  = Preprocess(TEST_PATH)
#
#  
#plt.imshow(np.squeeze(Y_test[2]))
#
#xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)



#train_generator, val_generator = generator(xtr, xval, ytr, yval, BATCH_SIZE)


#inc_v3_enc_dec_model.summary()







#
#

#ix = random.randint(0, 6)
#imshow(X_train[10])
#plt.show()
#imshow(np.squeeze(Y_train[10]))
#plt.show()
#
#

#inc_v3_enc_dec_model.save('my_model.h5') 
#
#
#

#
#inc_v3_enc_dec_model.load_weights('model_weights.h5')
#
#model = load_model('my_model.h5')
#
#
#
#
#y = inc_v3_enc_dec_model.predict(X_test[1:7])
#
#
#
#h =xy_provider(train_ids)

#def xy_provider(image_id,infinite=True):
#        while True:
#            for image_id in train_ids:
#                image = imread(TRAIN_IMAGE_PATH+"/"+image_id)
#                target = imread(TRAIN_MASK_PATH+"/"+image_id)
#                
#                
#                print(image.shape)
#                print(target.shape)
#
#                # Some custom preprocesssing: resize
#                # ...
#                yield image, target
#            if not infinite:
#                return

#
#train_gen = ImageDataGenerator(pipeline=('random_transform', 'standardize'),
#                             featurewise_center=True,
#                             featurewise_std_normalization=True,
#                             rotation_range=90.,
#                             width_shift_range=0.15, height_shift_range=0.15,
#                             shear_range=3.14/6.0,
#                             zoom_range=0.25,
#                             channel_shift_range=0.1,
#                             horizontal_flip=True,
#                             vertical_flip=True)
#    
#
#
#train_gen.fit(X_train)
#
#
#image_generator = train_gen.flow_from_directory(
#    TRAIN_PATH,
#    save_to_dir="temp/",
#    save_format='png',
#    class_mode='categorical',
#    seed=seed)
#
#


#i = 0
#for batch in train_generator.flow_from_directory(TRAIN_IMAGE_PATH, 
#                                           target_size=(768, 1024), 
#                                           batch_size=1,
#                          save_to_dir='temp', save_prefix='', save_format='png'):
#    i += 1
#    if i > 20:
#        break  # otherwise the generator would loop indefinitely
#
##
##
#data_gen_args = dict(horizontal_flip=True,
#                     vertical_flip=True,
#                     rotation_range=90.,
#                     width_shift_range=0.1,
#                     height_shift_range=0.1,
#                     zoom_range=0.1)
#image_datagen = ImageDataGenerator(**data_gen_args)
#mask_datagen = ImageDataGenerator(**data_gen_args)
##    image_datagen.fit(xtr, seed=7)
##    mask_datagen.fit(ytr, seed=7)
#image_datagen.fit(xtr)
#mask_datagen.fit(ytr)
#
#
#
##i = 0
##for batch in image_datagen.flow(xtr,
##                           batch_size=1,
##                           seed=7, 
##                           save_to_dir='temp/image',
##                           save_prefix='', 
##                           save_format='png'):
##  i += 1
##  if i > 20:
##    break  # otherwise the generator would loop indefinitely
##
##
##i = 0
##for batch in mask_datagen.flow(xtr,
##                           batch_size=1,
##                           seed=7, 
##                           save_to_dir='temp/masks',
##                           save_prefix='', 
##                           save_format='png'):
##  i += 1
##  if i > 20:
##    break  # otherwise the generator would loop indefinitely
#
#image_generator = image_datagen.flow(xtr,
#                           batch_size=1,
#                           seed=7, 
#                           save_to_dir='temp/image',
#                           save_prefix='', 
#                           save_format='png')
#
#mask_generator = mask_datagen.flow(ytr,
#                           batch_size=1,
#                           seed=7, 
#                           save_to_dir='temp/mask',
#                           save_prefix='', 
#                           save_format='png')
#
#
#
#train_generator = zip(image_generator,mask_generator)
#
#type(train_generator)
#print(list(train_generator))
#
#for a,b in train_generator:
#  pass
##  imshow(a[0])
##  plt.show()
##  imshow(b[0])
##  plt.show()
##  print(a.shape)
##  print(b.shape)
#
#mask_generator = mask_datagen.flow(ytr, batch_size=1, seed=7)
#
##
##
##train_generator = train_datagen.flow_from_directory(
##        'data/train',  # this is the target directory
##        target_size=(150, 150),  # all images will be resized to 150x150
##        batch_size=batch_size,
##        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
##
### this is a similar generator, for validation data
##validation_generator = test_datagen.flow_from_directory(
##        'data/validation',
##        target_size=(150, 150),
##        batch_size=batch_size,
##        class_mode='binary')
##
##
##
#
#
