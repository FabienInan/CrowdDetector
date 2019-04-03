from scipy.io import loadmat
from PIL import Image
import numpy as np
import math
import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def preProcessUCFCC50():

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
            generateDensityMap(im.size, annPoints, destPath + '/' + imgId, 4, 15)
            countUCFCC50DataSet.append(len(annPoints))

    np.savetxt(subDirDataSet + 'count.txt', countUCFCC50DataSet, '%d')

def preProcessShanghaiTechA():
    dirname = os.path.dirname(__file__)
    subDirDataSet = '../datasets/ShanghaiTech/part_A/'
    countShanghaiATrainDataSet = []

    #train data
    subDirTrainDataSet = subDirDataSet + 'train_data/'

    destPath = subDirTrainDataSet + 'densityMap'
    if os.path.exists(destPath) and os.path.isdir(destPath):
        shutil.rmtree(destPath)
    os.mkdir(destPath)

    for i in range(1,301):
        im = Image.open(subDirTrainDataSet + 'images/IMG_' + str(i) + '.jpg')
        content = loadmat(os.path.join(dirname, subDirTrainDataSet + 'ground-truth/GT_IMG_' + str(i) + '.mat'))
        annPoints = content['image_info'][0,0][0,0][0]
        generateDensityMap(im.size, annPoints, destPath + '/' + str(i), 4, 15)
        countShanghaiATrainDataSet.append(len(annPoints))

    np.savetxt(subDirTrainDataSet + 'count.txt', countShanghaiATrainDataSet, '%d')
    
    #test data
    subDirTrainDataSet = subDirDataSet + 'test_data/'

    destPath = subDirTrainDataSet + 'densityMap'
    if os.path.exists(destPath) and os.path.isdir(destPath):
        shutil.rmtree(destPath)
    os.mkdir(destPath)

    for i in range(1,183):
        im = Image.open(subDirTrainDataSet + 'images/IMG_' + str(i) + '.jpg')
        content = loadmat(os.path.join(dirname, subDirTrainDataSet + 'ground-truth/GT_IMG_' + str(i) + '.mat'))
        annPoints = content['image_info'][0,0][0,0][0]
        generateDensityMap(im.size, annPoints, destPath + '/' + str(i), 4, 15)
        countShanghaiATrainDataSet.append(len(annPoints))

    np.savetxt(subDirTrainDataSet + 'count.txt', countShanghaiATrainDataSet, '%d')

def preProcessShanghaiTechB():
    dirname = os.path.dirname(__file__)
    subDirDataSet = '../datasets/ShanghaiTech/part_B/'
    countShanghaiATrainDataSet = []

    #train data
    subDirTrainDataSet = subDirDataSet + 'train_data/'

    destPath = subDirTrainDataSet + 'densityMap'
    if os.path.exists(destPath) and os.path.isdir(destPath):
        shutil.rmtree(destPath)
    os.mkdir(destPath)

    for i in range(1,401):
        im = Image.open(subDirTrainDataSet + 'images/IMG_' + str(i) + '.jpg')
        content = loadmat(os.path.join(dirname, subDirTrainDataSet + 'ground-truth/GT_IMG_' + str(i) + '.mat'))
        annPoints = content['image_info'][0,0][0,0][0]
        generateDensityMap(im.size, annPoints, destPath + '/' + str(i), 4, 15)
        countShanghaiATrainDataSet.append(len(annPoints))

    np.savetxt(subDirTrainDataSet + 'count.txt', countShanghaiATrainDataSet, '%d')
    
    #test data
    subDirTrainDataSet = subDirDataSet + 'test_data/'

    destPath = subDirTrainDataSet + 'densityMap'
    if os.path.exists(destPath) and os.path.isdir(destPath):
        shutil.rmtree(destPath)
    os.mkdir(destPath)

    for i in range(1,317):
        im = Image.open(subDirTrainDataSet + 'images/IMG_' + str(i) + '.jpg')
        content = loadmat(os.path.join(dirname, subDirTrainDataSet + 'ground-truth/GT_IMG_' + str(i) + '.mat'))
        annPoints = content['image_info'][0,0][0,0][0]
        generateDensityMap(im.size, annPoints, destPath + '/' + str(i), 4, 15)
        countShanghaiATrainDataSet.append(len(annPoints))

    np.savetxt(subDirTrainDataSet + 'count.txt', countShanghaiATrainDataSet, '%d')

def generateDensityMap(imageSize, points, path, sigma, kernelSize):

    if len(points) == 0:
        return

    mapDensity = np.zeros([imageSize[1], imageSize[0]])

    for i in range(1, len(points + 1)):
        if points[i][0] < imageSize[0] and points[i][1] < imageSize[1] - 1 :
            x = int(abs(round(points[i][0])))
            y = int(abs(round(points[i][1])))
            x1 = x - int(math.floor(kernelSize/2))
            x2 = x + int(math.floor(kernelSize/2))
            y1 = y - int(math.floor(kernelSize/2))
            y2 = y + int(math.floor(kernelSize/2))
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > imageSize[0] - 1:
                x2 = imageSize[0] - 1
            if y2 > imageSize[1] - 1:
                y2 = imageSize[1] - 1
            gaussianFilter = gaussianKernel(y2-y1+1, x2-x1+1, sigma)
            mapDensity[y1:y2+1, x1:x2+1] = mapDensity[y1:y2+1, x1:x2+1]+ gaussianFilter

    np.savetxt(path + '.csv', mapDensity, '%f', delimiter=",")


def gaussianKernel(sizeX, sizeY, sigma):
    x, y = np.mgrid[-sizeX//2 + 1:sizeX//2 + 1, -sizeY//2 + 1:sizeY//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def displayMapDensity(mapDensity, title):
    plt.imshow(mapDensity, cmap=cm.jet)
    plt.title(title)
    plt.show()