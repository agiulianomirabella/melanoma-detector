import numpy as np

def extractWindow(image, center, windowWidth):
    n = windowWidth//2
    slices = tuple()
    for d in range(len(center)):
        slices = slices + (slice(center[d]-n, center[d]+n+1), )
    return image[slices]

def isWindowShape(window, expectedShape):
    if any([type(i) == list for i in np.ndarray.flatten(window)]):
        return False
    return window.shape == expectedShape

def isZerosWindow(window):
    unique = np.unique(window)
    return len(unique) == 1 and unique[0] == 0

def auxSplitMonochromeWindowsCoordinates(image, windowWidth):
    #returns a list of not monochrome windows centers coordinates
    if windowWidth%2 == 0:
        print('WARNING: windowWidth must be an odd number, 3 is being used by default')
        windowWidth = 3
    outMonochrome = []
    outNotMonochrome = []
    for indices in np.ndindex(image.shape):
        window = extractWindow(image, indices, windowWidth)
        if window.size!=0 and not isZerosWindow(window):
            if len(np.unique(window)) == 1:
                outMonochrome.append(indices)
            else:
                outNotMonochrome.append(indices)
    return (outMonochrome, outNotMonochrome)


'''
def extractForeWindow(image, center):
    slices = tuple()
    for d in range(len(center)):
        slices = slices + (slice(center[d], center[d]+2), )
    return image[slices]
'''
