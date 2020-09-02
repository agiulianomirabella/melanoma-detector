from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from multiprocessing import freeze_support
from root.segmentation.hr import have_hairs, best_ones

if __name__ == '__main__':
    freeze_support()

    image = readImage('../data/input/jpeg/train/ISIC_6595186.jpg')
    exploreImage(image)

    '''
    train_images = getFirstNImages(best_ones, 'train')
    displaySeveralImages(train_images, ncols = 5, titles=best_ones)
    '''

