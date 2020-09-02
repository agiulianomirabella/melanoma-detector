from root.transfer.denseNet201 import create_model, fit_model, plot_train_hist
from root.readData import readCSV
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

'''
Model:
    - input_shape: (128, 128, 3)
    - batch_size: 32
    - epochs: 15
'''


#aug parameters:
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest"

aug_parameters = (rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode)

def denseNet201_server0():

    print()
    print()
    print('###   model 0: ###')
    print()
    print()

    #images loading:
    data = readCSV(1000, valid_size=0.2)

    #model:
    model = create_model()
    history = fit_model(model, data, epochs=20, batch_size=32, aug_parameters=aug_parameters)
    model_name = 'model0'
    save_path = '../data/output/transfer/denseNet201/' + model_name + '/'
    model.save(save_path + model_name + '.h5')

    plot_train_hist(history, save_path, model_name)
