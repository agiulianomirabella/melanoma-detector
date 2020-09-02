import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.io import imsave

from root.utils import readImage, centerImage, displaySeveralImages, resize

seg_input_path      = '../data/halfway/seg/128/'
original_input_path = '../data/input/jpeg/train/'

def stats(image_name):
    segmented = readImage(seg_input_path + image_name + '_seg.jpg')

    if len(np.unique(segmented)) == 1:
        print('blank image')
        return np.zeros((11,))

    original  = resize(readImage(original_input_path + image_name + '.jpg'), (128, 128))

    try:
        out = np.concatenate((shape_properties(segmented, display=False), color_properties(segmented, original)))
    except:
        print('blank image')
        out = np.zeros((11,))
    
    return out


def shape_properties(segmented, display= False):

    label_img = label(segmented)
    props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'major_axis_length',
                                                 'minor_axis_length',
                                                 'area',
                                                 'bbox_area',
                                                 'eccentricity'))

    props = pd.DataFrame(props)

    areas_ratio = sum([row['area']/row['bbox_area'] for _, row in props.iterrows()])

    K_major = sum(props['major_axis_length'].values)
    K_minor = sum(props['minor_axis_length'].values)

    major_axis_weighted_mean = sum([x**2 for x in props['major_axis_length'].values]) / K_major
    minor_axis_weighted_mean = sum([x**2 for x in props['minor_axis_length'].values]) / K_minor

    eccentricity_mean = np.mean(props['eccentricity'].values)
    eccentricity_std  = np.std(props['eccentricity'].values)

    out = np.array([areas_ratio, major_axis_weighted_mean, minor_axis_weighted_mean, 
        eccentricity_mean, eccentricity_std])

    if display:
        fig, ax = plt.subplots(figsize= (10, 10))
        ax.imshow(segmented, cmap='gray')
        regions = regionprops(label_img)
        
        for props_aux in regions:
            y0, x0 = props_aux.centroid
            orientation = props_aux.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props_aux.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props_aux.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props_aux.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props_aux.major_axis_length

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax.plot(x0, y0, '.g', markersize=15)

            minr, minc, maxr, maxc = props_aux.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-b', linewidth=2.5)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        properties_image = resize(data, (128, 128))

        for name in ['major_axis_length','minor_axis_length','area','bbox_area','eccentricity']:

            print()
            print(name + ':')
            print(props[name].values)

        print()
        print('OUT: [ areas_ratio, major_axis_weighted_mean, minor_axis_weighted_mean, eccentricity_mean, eccentricity_std]')
        print('OUT: {}'.format(out))
        print()

        displaySeveralImages([segmented, properties_image])
        
    return out

def color_properties(segmented, original):

    img = original.copy()

    R_mean = np.mean(img[segmented==1][0])
    G_mean = np.mean(img[segmented==1][1])
    B_mean = np.mean(img[segmented==1][2])

    R_std = np.std(img[segmented==1][0])
    G_std = np.std(img[segmented==1][1])
    B_std = np.std(img[segmented==1][2])

    out = np.array([R_mean, G_mean, B_mean, R_std, G_std, B_std])

    return out
