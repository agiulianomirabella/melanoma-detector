from root.utils import * # pylint: disable= unused-wildcard-import
from root.utils import makeUnique, getGrayValueCoordinates
import numpy as np
from copy import copy
from scipy.ndimage.morphology import generate_binary_structure


'''
A helpful auxiliary permutation list for subcells computation
'''

#Maximum value for spaceDimension:
maximumSpaceDimension = 2
permutations1 = [[], [[-0.5], [0.5]], [[-0.5, -0.5], [-0.5, 0], [-0.5, 0.5], [0, -0.5], [0, 0.5], [0.5, -0.5], [0.5, 0], [0.5, 0.5]]]



'''
This module will compute cell's features, such as dimension, or subcells.
    - A cell is a  numpy array

'''

def dim(cell):
    return len([i for i in range(len(cell)) if cell[i]%1 == 0])

def getRationalIndices(cell):
    return [i for i in range(len(cell)) if cell[i]%1 != 0]

def getSubCells(cell): #return a list of subCells
    out = []
    x = permutations1[len(cell)]
    for l in x:
        if all(l[i]==0 for i in getRationalIndices(cell)):
            a = cell + np.array(l)
            out.append(a)
    return makeUnique(out)






'''
This module will define functions to extract CCs eulerchar feature.
    - A CC is a list of arrays (coordinates of cells belonging to the CC)
'''

def getAllCells(cc):
    out = copy(cc)
    for cell in cc:
        out = out + getSubCells(cell)
    return makeUnique(out)

def euler(cc):
    if len(cc)==0:
        return 0
    out = 0
    allCells = getAllCells(cc)
    for d in range(len(cc[0])+1):
        out = out + ((-1)**d) * len([c for c in allCells if dim(c) == d])
    return out

