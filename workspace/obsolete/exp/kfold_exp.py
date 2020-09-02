from root.transfer.denseNet201Kfold import create_and_train_model, plot_score
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

#model:
model_name = 'prueba0'
save_path = '../data/output/transfer/pruebas/'
score = create_and_train_model(model_name, save_path, n_sample= 50, k= 5, epochs= 5, batch_size= 32, aug_parameters= aug_parameters)
plot_score(score, save_path, model_name)

t2 = time.time()

t_total   = round((t2-t1)/60, 2)

print()
print()
print('TOTAL TIME: {}m.      '.format(t_total))
print()
print()

