#from root.utils import * # pylint: disable= unused-wildcard-import
from root.utils import resize, toGray
from root.readData import readData, readCSV
import time
import os

#CHOOSE THE MODEL:
#from root.cnn.cnn_v1 import createModel, testModel, input_shape
#from root.cnn.cnn_tutorial import createModel, testModel, input_shape
#from root.cnn.cnn_dense import createModel, testModel, input_shape
from root.cnn.cnn_generator import createModel, testModel, input_shape


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

t1 = time.time()
#images loading:
train_df, train_rgb_images, train_labels, test_df, test_rgb_images, test_labels = readData(100, valid_size=0.1)

t2 = time.time()
#images processing:
'''
train_images = [resize(toGray(image), input_shape[:2]) for image in train_rgb_images]
'''
test_images  = [resize(toGray(image), input_shape[:2]) for image in test_rgb_images]

train_df = readCSV(270)
test_df  = readCSV(30)

t3 = time.time()
history, model = createModel(train_df)
#score = testModel(model, test_df)

t4 = time.time()

t_total = round((t4-t1)/60, 2)
t_load = round((t2-t1)/60, 2)
t_process = round((t3-t2)/60, 2)
t_cnn = round((t4-t3)/60, 2)

print()
print()
print('TOTAL TIME            : {}m.      '.format(t_total))
print('IMAGES LOAD TIME      : {}m. ({}%)'.format(t_load,    round(t_load*100/t_total, 2)))
print('IMAGES PROCESSING TIME: {}m. ({}%)'.format(t_process, round(t_process*100/t_total, 2)))
print('CNN TIME              : {}m. ({}%)'.format(t_cnn,     round(t_cnn*100/t_total, 2)))
print()
print()

