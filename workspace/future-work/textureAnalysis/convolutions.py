import numpy as np
import time
from root.topology.textureAnalysis.convolutionUtils import auxSplitMonochromeWindowsCoordinates, extractWindow, isZerosWindow

'''
This module is intended to define functions capable of convolving
another funtion in an image. The parameter 'function' is a function that will be applied 
to a small sized squared window centered in each ncell of the image.
'''

def applyFunctionConvolution(image, function, windowWidth):
    #returns a dict (center) -> value of the convolution applied to the window located in center
    if windowWidth%2 == 0:
        print('WARNING: windowWidth must be an odd number, 3 is being used by default')
        windowWidth = 3
    out = dict()
    for center in np.ndindex(image.shape):
        window = extractWindow(image, center, windowWidth)
        if window.size!=0 and not isZerosWindow(window): #border-elements-centered windows size will equal zero
            out[center] = function(window)
    return out

def applyNonMonochromeFunctionConvolution(image, monochromeFunction, nonMonochromeFunction, windowWidth):
    out = dict()
    monochrome, nonMonochrome = auxSplitMonochromeWindowsCoordinates(image, windowWidth)
    print('monochrome percentage: {}%'.format(round((len(monochrome)/len(nonMonochrome))*100, 1)))

    for center in monochrome:
        window = extractWindow(image, center, windowWidth)
        if window.size != 0: #border values will equal 0
            out[center] = monochromeFunction(window)

    for center in nonMonochrome:
        window = extractWindow(image, center, windowWidth)
        if window.size != 0: #border values will equal 0
            out[center] = nonMonochromeFunction(window)

    return out


