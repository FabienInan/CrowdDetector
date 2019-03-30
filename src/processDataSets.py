from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
import shutil
import random

def processUCFCC50():

    dirname = os.path.dirname(__file__)
    subDirDataSet = '../datasets/UCF_CC_50/'

    countUCFCC50DataSet = []
    
    #create cross validation set

    reorderSamples = random.sample(range(1,51), 50)

    for i in range(0, 5):

        destPath = subDirDataSet + 'set_' + str(i+1)
        if os.path.exists(destPath) and os.path.isdir(destPath):
            shutil.rmtree(destPath)
        os.mkdir(destPath)

        for j in range(1,11):
            imgId = str(reorderSamples[10*(i-1) + j])
            srcPath = subDirDataSet + imgId
            shutil.copy(srcPath + '.jpg', destPath)
            # count ground truth and generate density map
            im = Image.open(srcPath + '.jpg');  
            content = loadmat(os.path.join(dirname, srcPath + '_ann.mat'))
            annPoints = content['annPoints']
            generateDensityMap(im.size, annPoints, destPath + '/' + imgId)
            countUCFCC50DataSet.append(len(annPoints))

    np.savetxt(subDirDataSet + 'count.txt', countUCFCC50DataSet, '%d')


def generateDensityMap(imageSize, points, path):

    mapDensity = np.zeros(imageSize, dtype=int)

    np.savetxt(path + '.csv', mapDensity, '%d', delimiter=",")