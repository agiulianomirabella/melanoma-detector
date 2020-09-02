import matplotlib.pyplot as plt


def plot_train_hist(hist, path, model_name):
    # Plot training & validation accuracy values
    plt.plot(hist.history['AUC'])
    plt.plot(hist.history['val_AUC'])
    plt.title('Model AUC')
    plt.ylabel('Auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_auc.jpg')
    plt.clf()

    # Plot training & validation accuracy values
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_accuracy.jpg')
    plt.clf()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_loss.jpg')
    plt.clf()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_score(auc_per_epochs, path, model_name):
    points = smooth_curve(auc_per_epochs[10:])
    plt.plot(range(1, len(points) + 1), points)
    plt.xlabel('Epochs')
    plt.ylabel('AUC score')
    plt.savefig(path + model_name + '_auc_vs_epochs.jpg')
    plt.clf()

def print_model_parameters(model_name, save_path, n_sample, batch_size, epochs, k, aug_parameters):
    f = open(save_path + model_name + '_parameters.txt', 'w')
    f.write(model_name + '\n')
    f.write('batch_size: {}\n'.format(batch_size))
    f.write('epochs: {}\n'.format(epochs))
    f.write('k: {}\n'.format(k))
    f.write('n_sample: {}\n'.format(n_sample))

    rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode = aug_parameters
    f.write('\ndata augmentation parameters:\n')
    f.write('         rotation_range: {}\n'.format(rotation_range))
    f.write('         zoom_range: {}\n'.format(zoom_range))
    f.write('         width_shift_range: {}\n'.format(width_shift_range))
    f.write('         height_shift_range: {}\n'.format(height_shift_range))
    f.write('         shear_range: {}\n'.format(shear_range))
    f.write('         horizontal_flip: {}\n'.format(horizontal_flip))
    f.write('         fill_mode: {}\n'.format(fill_mode))

    f.close()



def write_results(auc_per_epochs, save_path):
    f = open(save_path + 'results.txt', 'w')
    for i, auc in enumerate(auc_per_epochs):
        f.write('Epoch: {}. AUC: {}.'.format(i, auc))
    f.close()


