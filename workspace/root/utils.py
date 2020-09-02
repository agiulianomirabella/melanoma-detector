import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
from multiprocessing import Process

import pydicom as dcm
import random
import cv2

from skimage.color import gray2rgb, rgb2gray
from skimage import io
from skimage.transform import resize as resize_image
from scipy.ndimage import label, find_objects
from skimage.filters import threshold_otsu, threshold_multiotsu

from pydicom.pixel_data_handlers.util import convert_color_space



#                                           EXPLORATION
#
def printImageInfo(image):
    #print image shape, min and max
    print('\nIMAGE INFO:')
    print('Image shape: {}; min: {}, max: {}; number of different gray levels: {}\n'.format(np.shape(image),
        np.min(image), np.max(image), len(np.unique(image))))

def displayImage(image):
    #display a 2D image
    plt.imshow(image, cmap='gray')
    plt.show()

def displaySeveralImages(images, ncols= -1, titles= [], save= None):

    if titles and len(titles) != len(images):
        print('There are more or less titles than images')
        titles = []

    if ncols < 0 or ncols > len(images):
        ncols = len(images)

    nrows= len(images)//ncols

    if len(images)%ncols != 0:
        nrows = nrows + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))

    if len(images) == 1:
        axs.imshow(images[0], cmap='gray')
        axs.axis('off')
        if titles:
            axs.title.set_text(titles[0])
    else:
        if nrows == 1:
            for i, image in enumerate(images):
                axs[i%ncols].imshow(image, cmap='gray')
                axs[i%ncols].axis('off')
                if titles:
                    axs[i%ncols].title.set_text(titles[i])

        else:
            for i, image in enumerate(images):
                axs[i//ncols, i%ncols].imshow(image, cmap='gray')
                axs[i//ncols, i%ncols].axis('off')
                if titles:
                    axs[i//ncols, i%ncols].title.set_text(titles[i])

    if isinstance(save, str):
        plt.savefig(save)

    plt.show()


def displayHistogram(image, axis= [], omitZeroValues= True):
    f = rgb2gray(image).flatten()
    if omitZeroValues:
        f = f[f!=0]
    plt.hist(f,  bins= 'auto')
    if axis:
        plt.axis(axis)
    plt.ylabel('Frequency')
    plt.xlabel('Gray Values')
    plt.xlim(left = 0)
    if omitZeroValues:
        plt.title('Image Histogram (zero values omitted).')
    else:
        plt.title('Image Histogram.')
    plt.show()

def exploreImage(image, omitZeroValuesInHistogram= True):
    #print the image info, then display the image and its histogram
    procs = []

    proc1 = Process(target=printImageInfo, args=(image,))
    procs.append(proc1)
    proc2 = Process(target=displayImage, args=(image,))
    procs.append(proc2)
    proc3 = Process(target=displayHistogram, args=(image, omitZeroValuesInHistogram))
    procs.append(proc3)
    proc1.start()
    proc2.start()
    proc3.start()

def exploreRegionOfInterest(image, regionPixels):
    print("Region size: {};".format(len(regionPixels)))
    aux = copy.copy(image)

    for c in regionPixels:
        aux[tuple(c)] = [0, 0, 1]

    exploreImage(aux)





#                                           EXTRACTION & WRITING
#
def readImage(path):
    #given an image path, return the image as an numpy array
    return normalize(np.array(io.imread(path)))

def readRandomImage(path):
    img_name = random.sample(os.listdir(path), 1)[0]
    return img_name, readImage(path + img_name)

def readDicomImage(path):
    #given a dicom file path, return its image as a numpy array
    return convert_color_space(dcm.dcmread(path).pixel_array, 'YBR_FULL_422', 'RGB')

def getImages(images_names):
    out = []
    path = '../data/input/train/'
    for i in images_names:
        img = readDicomImage(path + i + '.dcm')
        img = resize(img)
        img = normalize(img)
        out.append(img)
    return out

def getJPGImages(images_names):
    out = []
    path = '../data/input/jpeg/train/'
    for i in images_names:
        img = readImage(path + i + '.jpg')
        img = resize(img)
        img = normalize(img)
        out.append(img)
    return out

def getFirstNImages(images_numbers):

    if isinstance(images_numbers, int):
        images_numbers = [i for i in range(images_numbers)]

    images_names = []
    path = '../data/input/train/'
    names = os.listdir(path)

    for i in images_numbers:
        images_names.append(names[i][:-4])

    return getImages(images_names)


def getDICOMHeader(n=5):
    #n: number of rows required
    folder = '../data/input/train/'
    images = list(os.listdir(folder))
    df = pd.DataFrame()
    for image in images[0:n]:
        image_name = image.split(".")[0]
        dicom_file_path = os.path.join(folder, image)
        dicom_file_dataset = dcm.read_file(dicom_file_path)
        study_date = dicom_file_dataset.StudyDate
        modality = dicom_file_dataset.Modality
        age = dicom_file_dataset.PatientAge
        sex = dicom_file_dataset.PatientSex
        body_part_examined = dicom_file_dataset.BodyPartExamined
        patient_orientation = dicom_file_dataset.PatientOrientation
        photometric_interpretation = dicom_file_dataset.PhotometricInterpretation
        rows = dicom_file_dataset.Rows
        columns = dicom_file_dataset.Columns

        df = df.append(pd.DataFrame({'image_name': image_name, 
                        'dcm_modality': modality,'dcm_study_date':study_date, 'dcm_age': age, 'dcm_sex': sex,
                        'dcm_body_part_examined': body_part_examined,'dcm_patient_orientation': patient_orientation,
                        'dcm_photometric_interpretation': photometric_interpretation,
                        'dcm_rows': rows, 'dcm_columns': columns}, index=[0]))
    return df



 
#                                           PROCESSING
#
def normalize(image):
    #return the imagen ormalized
    image = image - np.min(image)
    if np.max(image) == 0:
        return image
    return image/np.max(image)

def binarize(image, grayLevel):
    #set grayLevel voxels to 1 and different to 0
    out = np.zeros(image.shape)
    out[image == grayLevel] = 1
    return out

def otsuBinarization(image, grayLevel = None):
    #set grayLevel or less voxels to 0 and more-than-grayLevel voxels to 1
    if grayLevel is None:
        grayLevel = threshold_otsu(image)
    out = copy.copy(image)
    out[image <= grayLevel] = 0
    out[image > grayLevel] = 1
    return out

def multiOtsuBinarization(image, n):
    #set grayLevel or less voxels to 0 and more-than-grayLevel voxels to 1
    return normalize(np.digitize(image, bins = threshold_multiotsu))


def setGrayThresholds(image, lowerThreshold = 0, upperThreshold = 1):
    #set lower than lowerThreshold grayValues to lowerThreshold and higher than upperThreshold to upperThreshold

    if lowerThreshold > upperThreshold:
        raise ValueError('Upper threshold must be grater or equal than lower')

    if lowerThreshold < 0:
        lowerThreshold = 0
    if upperThreshold > 1:
        upperThreshold = 1

    out = copy.copy(image)
    out[image <= lowerThreshold] = lowerThreshold
    out[image > upperThreshold] = upperThreshold
    return normalize(out)

def setPercentileGrayThresholds(image, lowerThresholdPercentile = 0, upperThresholdPercentile = 100):
    #Set lowerThresholdPercentile grayLeves to lowerThresholdPercentile and upperThresholdPercentile to upperThresholdPercentile

    if lowerThresholdPercentile < 0:
        lowerThresholdPercentile = 0
    if upperThresholdPercentile > 100:
        upperThresholdPercentile = 100

    lowerThreshold = np.percentile(image, lowerThresholdPercentile)
    upperThreshold = np.percentile(image, upperThresholdPercentile)

    #print()
    #print('The {} (lower) percentile gray level is: {}'.format(lowerThresholdPercentile, round(lowerThreshold, 2)))
    #print('The {} (upper) percentile gray level is: {}'.format(upperThresholdPercentile, round(upperThreshold, 2)))

    return setGrayThresholds(image, lowerThreshold, upperThreshold)

def digitizeToEqualFrequencies(image, binsNumber):
    #equal frequency binning of the image
    out = np.zeros(image.shape)
    labels = [i for i in range(binsNumber)]
    elements = np.sort(image.flatten())
    elementsPerBin = image.size//binsNumber
    bin_edges = [elements[i*elementsPerBin] for i in labels]
    for i, e in enumerate(bin_edges):
        out[image > e] = labels[i]
    return normalize(out)
    
def digitizeToEqualWidth(image, GrayLevelsNumber):
    #equal width binning of the image
    if GrayLevelsNumber > len(np.unique(image)):
        GrayLevelsNumber = len(np.unique(image))
    return normalize(np.digitize(image, bins = np.linspace(0, np.max(image), GrayLevelsNumber)) - 1)

def resize(image, shape= None):
    if shape is None:
        return normalize(resize_image(image, (128, 128), anti_aliasing=True))
    else:
        return normalize(resize_image(image, shape, anti_aliasing=True))

def centerImage(image):
    #eliminate empty borders of the image
    aux = copy.copy(image)
    aux[image!=0] = 1
    aux = aux.astype(int)
    margins = find_objects(aux)[0]
    return image[margins]

'''
def otsu(image):
    img = normalize(rgb2gray(image))
    img = otsuBinarization(img)
    img = binarize(img, 0)
    return img
'''
def toGray(image):
    return normalize(rgb2gray(image))





#                                           PIXELS
#
def makeUnique(voxelsList):
    #delete duplicates in a list of voxels coordinates
    return [np.array(a) for a in set([tuple(e) for e in voxelsList])]

def getGrayValueCoordinates(image, grayValue):
    #return a list of all coordinates where image ncells takes the value grayValue
    out = []
    info = np.where(image == grayValue)
    for i in range(len(info[0])):
        a = []
        for dim in info:
            a.append(dim[i])
        out.append(np.array(a))
    return out


'''
def addZeroMargins(image):
    #add zero margins to ndarray
    padWidth = tuple()
    constantValues = tuple()
    for _ in range(image.ndim):
        padWidth = padWidth  + ((1, 1), )
        constantValues = constantValues  + ((0, 0), )
    return np.pad(image, padWidth, 'constant', constant_values=constantValues)
'''
