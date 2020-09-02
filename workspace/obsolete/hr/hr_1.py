import numpy as np
from hair_removal import bank_of_structuring_elements, remove_and_inpaint

def hairs_remove1(image):
    tophats_se = bank_of_structuring_elements(side_enclosing_square_in_px=9, num_orientations=3)
    inpainting_se = np.ones((41, 41), dtype='float32')
    hairless_image, steps = remove_and_inpaint(image, tophats_se=tophats_se, inpainting_se=inpainting_se)
    return hairless_image
