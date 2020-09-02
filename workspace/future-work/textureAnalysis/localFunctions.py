import numpy as np

'''
This module will define functions to convolve on an image in order to acquire later a texture analysis.
Window parameter is therefore supposed to be a small sized squared ndarray
'''

def diversity(window):
    return len(np.unique(window))

