from root.topology.textureAnalysis.localFunctions import diversity
from root.topology.textureAnalysis.convolutions import applyFunctionConvolution, applyNonMonochromeFunctionConvolution

'''
This module is intended to define functions 
to compute the texture analysis on a complete image
'''

def diversityTextureAnalysis(image, windowWidth):
    return applyFunctionConvolution(image, diversity, windowWidth)


'''
def completeLocalTopologicalHistogram(image, windowWidth):
    return applyNonMonochromeFunctionConvolution(image, auxiliarLocalCompleteTopologyForMonochromeWindow, 
        topologicalHistogram, windowWidth)
'''
