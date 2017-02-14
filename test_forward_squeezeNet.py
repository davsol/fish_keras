from scipy import misc
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import theano
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
import sys
from scipy import misc
import time

sys.path.append(os.path.abspath('squeezeNet'))
from squeezenet import *


#import json

# test a forward
projectPath = "C:\Users\David\Desktop\GitFolder\project1"
dataPath = "C:\Users\David\Desktop\Fish"
scorePath = os.path.join(dataPath, "train", "binaryScoreMaps")
fishIndices = np.array([0,1,2,3,4,5,6,7,8,389,390,391,392,393,394,395,396])

# extracting the relevant paths for this function
trainingPath = os.path.join(dataPath, "train", "train")

# run through images and forward them in the squeezeNet
fishTypes = os.listdir(trainingPath)
model = get_squeezenet(1000,720,1080, dim_ordering='th')
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.load_weights(os.path.abspath('squeezeNet/model/squeezenet_weights_th_dim_ordering_th_kernels.h5'))
        
for fishIdx in np.arange(len(fishTypes)):
    fishPath = os.path.join(trainingPath, fishTypes[fishIdx])
    # filter all non jpg files\folders
    imageNames = [fn for fn in os.listdir(fishPath) if fn.endswith("jpg")]
    # run through images in a fishDir
    for imageIdx in np.arange(len(imageNames)):
        imagePath = os.path.join(fishPath, imageNames[imageIdx])
        
        origImg = cv2.imread(imagePath)            
        img = origImg.astype('float32')
        img = misc.imread(imagePath).astype(np.float32)
        img = misc.imresize(img, (720, 1080)).astype(np.float32)
        #plt.imshow(img)
        
        aux = copy.copy(img)
        img[:, :, 0] = aux[:, :, 2]
        img[:, :, 2] = aux[:, :, 0]
    
        # Remove train image mean
        img[:, :, 0] -= 104.006
        img[:, :, 1] -= 116.669
        img[:, :, 2] -= 122.679
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        t_start = time.time()
        res = model.predict(img)
        t_end = time.time()
        newRes = np.sum(res[:,fishIndices, :, :], axis = 1).squeeze()
        print newRes.shape        
        newRes = misc.imresize(newRes, (img.shape[2], img.shape[3])).astype(np.float32)
        
        plt.figure()
        plt.imshow(newRes)
        plt.figure()
        plt.imshow(origImg)
        plt.show()
        plt.waitforbuttonpress(timeout=2)
        #raw_input("Press Enter to continue...")
        
        
        
        
        
        
        
        
        
        
        
        
        