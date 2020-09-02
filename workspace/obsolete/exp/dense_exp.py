from root.transfer.denseNet201 import create_model, fit_model, plot_train_hist
from root.readData import readCSV
import time
import os

'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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

t1 = time.time()

#images loading:
data = readCSV(10, valid_size=0.2)

#model:
model = create_model()
history = fit_model(model, data, epochs=1, batch_size=32, aug_parameters=aug_parameters)
model_name = 'prueba0'
save_path = '../data/output/transfer/pruebas/'
model.save(save_path + model_name + '.h5')


t2 = time.time()

t_total   = round((t2-t1)/60, 2)

print()
print()
print('TOTAL TIME: {}m.      '.format(t_total))
print()
print()

plot_train_hist(history, save_path, model_name)