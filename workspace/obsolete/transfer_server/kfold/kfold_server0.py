from root.transfer.denseNet201Kfold import create_and_train_model, plot_score, print_model_parameters
import os

'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''


EPOCHS= 15
BATCH_SIZE= 64

#aug parameters:
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest"

aug_parameters = (rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode)

def kfold_server0(model_name, n_sample, k=5, batch_size=BATCH_SIZE, epochs=EPOCHS):

    print()
    print()
    print('##       MODEL:      ##')
    print('##   ' + model_name)
    print()
    print()
    
    save_path = '../data/output/transfer/kfold/'
    if not os.path.exists(save_path + model_name):
        os.mkdir(save_path + model_name)
    save_path = save_path + model_name + '/'

    print_model_parameters(model_name, save_path, n_sample, batch_size, epochs, k, aug_parameters)

    score = create_and_train_model(model_name, save_path, n_sample=n_sample, batch_size= batch_size, epochs= epochs, k= k, aug_parameters= aug_parameters)
    plot_score(score, save_path, model_name)

