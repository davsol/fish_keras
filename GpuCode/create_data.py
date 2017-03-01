import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os 
import theano
import keras
import json

# this function is supposed to read the json files of the fish bounding boxes and to write
# the images into a new directory
def create_data():
    projectPath = "~\Desktop\GIT\FishProject1\keras_fish"
    dataPath = "~\Desktop\Fishdata"
    scorePath = os.path.join(dataPath, "binaryScoreMaps")
    
    # extracting the relevant paths for this function
    trainingDataPath = os.path.join(dataPath, "train")
    
    # creating path for the score maps that we will create
    if not os.path.exists(scorePath):
        os.mkdir(scorePath)
    
    #img = cv2.imread('messi4.jpg')
    #b,g,r = cv2.split(img)
    #img2 = cv2.merge([r,g,b])
    #plt.subplot(121);plt.imshow(img) # expects distorted color
    #plt.subplot(122);plt.imshow(img2) # expect true color
    #plt.show()
    
    # extracting list of txt files containing json box annotations
    txtFileNames = os.listdir(os.path.join(projectPath, "labeledData"))
    for txtFileName in np.arange(len(txtFileNames)):
        print "textFile : " +  txtFileNames[txtFileName]
        extractedTxt = open(os.path.join(projectPath, "labeledData",txtFileNames[txtFileName]))
        extractedTxt = extractedTxt.read()
        decoded = json.loads(extractedTxt)
        for imageIdx in np.arange(len(decoded)):
            print "    image number : " + imageIdx.astype(str)
            imageInfo = decoded[imageIdx]
            currentImagePath = imageInfo['filename']
            currentImagePath = currentImagePath.replace("/", "\\")
            imagePath = os.path.join(trainingDataPath, currentImagePath)
            img=cv2.imread(imagePath)
            #plt.imshow(img)
            scoreMap = np.zeros(img.shape, dtype=np.single)
            
            # extract annotations from image
            imageAnnotation = imageInfo['annotations']
            for boundingBoxIdx in np.arange(len(imageAnnotation)):
                currentBBox = imageAnnotation[boundingBoxIdx]
                x = np.int(currentBBox['x'])
                y = np.int(currentBBox['y'])
                h = np.int(currentBBox['height'])
                w = np.int(currentBBox['width'])
                bboxSize = h * w
                #scoreMap[y:y+h, x:x+w, :] += np.single((255.0*255.0)/ bboxSize)
                scoreMap[y:y+h, x:x+w, :] = 255
            
            if not os.path.exists(os.path.join(scorePath, currentImagePath.partition("\\")[0])):
                os.mkdir(os.path.join(scorePath, currentImagePath.partition("\\")[0]))
            cv2.imwrite(os.path.join(scorePath, currentImagePath), scoreMap)
            
                
            
            
            
    
    
    
