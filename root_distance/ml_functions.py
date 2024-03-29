### DL architecture
import tensorflow as tf
from root_distance.general_functions import check_dims

import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model

import os
import numpy as np
import cv2
import random
import itertools

import zipfile
from io import BytesIO
import requests
from urllib.parse import urlparse

def regNet_model_fixed():

    regnet = tf.keras.applications.regnet.RegNetX160(input_shape=(512,512,3),
                                          weights='imagenet',
                                          include_top=False)

    x = layers.Dropout(0.3)(regnet.output)
    x  = layers.Conv2DTranspose(512, (2, 2), strides=2, activation="relu", padding="same")(x)
    x  = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x  = layers.Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x  = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x  = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x  = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x  = layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x )
        
    return Model(regnet.input, x)


def vgg16_model_fixed(width = 512, heigth=512, channels = 3):

    vggmodel = tf.keras.applications.vgg16.VGG16(
                                          input_shape=(width,heigth,channels),
                                          weights='imagenet',
                                          include_top=False)


    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    x = global_average_layer(vggmodel.output)

    for i, layer in enumerate(vggmodel.layers):
        layer.trainable=False
        
    outputs = [layer.output for layer in vggmodel.layers[1:(18)]]


    # add a GAP layer
    model = Model(vggmodel.input, outputs)

    x = layers.Dropout(0.3)(model.output[-1])
    
    # Decoder
    x  = layers.Conv2DTranspose(512, (3, 3), strides=2, activation="relu", padding="same")(x)
    x  = keras.layers.BatchNormalization()(x)
    x  = layers.Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = keras.layers.BatchNormalization()(x)
    x  = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = keras.layers.BatchNormalization()(x)
    x  = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x )
    x  = keras.layers.BatchNormalization()(x)
    x  = layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x )

    model = Model(vggmodel.input, x)
    return model



def findepochnumber(weigthfiles, index, strepochref = 'epoch', stopepochstr ='-loss' ):
  flindex = [i[:-6] for i in weigthfiles if i.endswith(index)]  
  
  return np.array([int(i[(i.index(strepochref)+5):(i.index(stopepochstr))]) for i in flindex]), flindex


def find_best_epoch(folderpath, load_last = False):
    tfiles = os.listdir(folderpath)
    last_epoch = 0

    if load_last:
        flindex = [i[:-6] for i in tfiles if i.startswith("checkpoint") and i.endswith("index")]
    
        if len(flindex)>0:
            epochnumber = np.array([int(i[10:]) for i in flindex])

        else:
            epochnumber, flindex = findepochnumber(tfiles, index=".index")

    else:
        tfiles = [i for i in tfiles if np.logical_not(i.startswith("checkpoint"))]
        epochnumber, flindex = findepochnumber(tfiles, index="index")

    if len(epochnumber)>0:
        bestmodel = flindex[np.where(epochnumber == max(epochnumber))[0][0]]
        epochnumber = np.array(epochnumber)
        last_epoch = epochnumber[0]
        epochnumber = np.array(epochnumber)
        epochnumber[::-1].sort()
        last_epoch = epochnumber[0]

    else:
        bestmodel = None

    return os.path.join(folderpath, bestmodel), last_epoch


def readweigths_frompath(weigth_path, modelname = None):

    last_epoch = 0
    
    if weigth_path.startswith('http'):
        a = urlparse(weigth_path)

        if not os.path.exists(os.path.basename(a.path)):
            req = requests.get(weigth_path)

            with zipfile.ZipFile(BytesIO(req.content)) as zipobject:
                zipobject.extractall('models')
        
        else:
            with zipfile.ZipFile(os.path.basename(a.path)) as zipobject:
                zipobject.extractall('models')
        
        newpathtomodel = os.path.join('models',os.path.basename(a.path)[:-4])
        fileinfolder = [i for i in os.listdir(newpathtomodel) if i.endswith('index') and i.startswith(modelname)]
        
        if len(fileinfolder)>1:
            wp, last_epoch = find_best_epoch(newpathtomodel)
        elif len(fileinfolder)==1:
            wp = fileinfolder[0][:-6]
            wp = os.path.join(newpathtomodel, wp)
        else:
            raise ValueError("there is no weights files")

    else:
        wp, last_epoch = find_best_epoch(weigth_path)
    
    return wp, last_epoch
        
    
class root_detector(object):

    def restore_weights(self):

      self.bestmodel, self._last_epoch = readweigths_frompath(self.weigth, self.achitecturename)
        
      if self.bestmodel is not None:
        print("checkpoint load {}".format(self.bestmodel))

        self.model.load_weights(self.bestmodel)
        
          
      else:
        print("it was not possible to load weights **********")
        


    def detect_root(self,img, threshhold = 0.3):
        ## DxHxWxC
        if len(img.shape) == 4:
            imgorigshape = (img.shape[1],img.shape[2])
        else:
            ## HxWxC
            imgorigshape = (img.shape[0],img.shape[1])
        imgc = check_dims(img, referenceshape = self.inputshape)
        
        if len(imgc.shape) == 3:
            imgc = np.expand_dims(imgc, axis = 0)

        predicitionimg = self.model.predict(imgc/255.)
        predicitionimg[predicitionimg<threshhold] = 0
        predicitionimg[predicitionimg>=threshhold] = 1
        
        root_image = check_dims(predicitionimg, referenceshape = imgorigshape)
        
        #self.root_image = predicitionimg
        if len(root_image.shape) == 4:          
            root_image = np.squeeze(root_image, axis = 3)
        return root_image



    def _set_model(self):
        

        if self.achitecturename == "vgg16":
            self.model = vgg16_model_fixed()

        if self.achitecturename == "regnet":
            self.model = regNet_model_fixed()

        #else:
        #    raise ValueError("Only vgg16 is available so far")

        if self.weigth is not None:
            self.restore_weights()


    def __init__(self, weigths_path = None, architecture = "vgg16") -> None:
        tf.keras.backend.clear_session()

        self.inputshape = [512,512]
        self.achitecturename = architecture
        self._load_last = False
        self.weigth = weigths_path
        self._set_model()
        
            
    


        