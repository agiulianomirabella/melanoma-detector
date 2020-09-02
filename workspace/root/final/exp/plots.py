import matplotlib.pyplot as plt
import numpy as np
import os

def plot_train_hist(hist, path, model_name):
    # Plot training & validation accuracy values
    plt.plot(hist.history['auc'])
    plt.plot(hist.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(path + model_name + '_auc.jpg')
    plt.clf()

    # Plot training & validation accuracy values
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(path + model_name + '_accuracy.jpg')
    plt.clf()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(path + model_name + '_loss.jpg')
    plt.clf()

def plot_experiment_means(auc, acc, loss, val_auc, val_acc, val_loss, path, save_name, batch_size, keyword, model_name):
    points = []
    points.append(smooth_curve(auc))
    points.append(smooth_curve(acc))
    points.append(smooth_curve(loss))
    points.append(smooth_curve(val_auc))
    points.append(smooth_curve(val_acc))
    points.append(smooth_curve(val_loss))

    max_val_auc = np.max(points[3])
    max_val_acc = np.max(points[4])
    min_val_loss = np.min(points[5])

    for p in points:
        plt.plot(p)

    plt.xlabel('Epochs')
    plt.ylabel('mean values per epoch')
    plt.legend(['AUC', 'accuracy', 'loss', 'val_AUC', 'val_accuracy', 'val_loss'], loc='upper left')
    plt.xticks(np.arange(0, len(points[0])+1, len(points[0])/10))
    plt.tick_params(labelright=True, right=True)
    plt.axis([0, len(points[0]), 0, 1])
    plt.text(0.05*len(points[0]), 0.23, '                             val_:       epoch: ')
    plt.text(0.05*len(points[0]), 0.15, 'max AUC:             {},             {}'.format(round(max_val_auc, 2), np.where(points[3] == max_val_auc)[0][0]))
    plt.text(0.05*len(points[0]), 0.1, 'max accuracy:     {},             {}'.format(round(max_val_acc, 2), np.where(points[4] == max_val_acc)[0][0]))
    plt.text(0.05*len(points[0]), 0.05, 'min loss:              {},             {}'.format(round(min_val_loss, 2), np.where(points[5] == min_val_loss)[0][0]))
    plt.text(0.75*len(points[0]), 0.05, '')
    plt.title('Model name: {}; inputs: {}; batch size: {}'.format(model_name, keyword, batch_size))
    
    if os.path.exists(path + save_name + '_means_vs_epochs.jpg'):
        os.remove(path + save_name + '_means_vs_epochs.jpg')
    plt.savefig(path + save_name + '_means_vs_epochs.jpg', bbox_inches='tight')
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

def write_experiment_parameters(model_name, save_path, n_sample, batch_size, epochs, k):
    f = open(save_path + model_name + '_parameters.txt', 'w')
    f.write(model_name + '\n')
    f.write('batch_size: {}\n'.format(batch_size))
    f.write('epochs: {}\n'.format(epochs))
    f.write('k: {}\n'.format(k))
    f.write('n_sample: {}\n'.format(n_sample))

    f.close()

def write_results(auc_per_epochs, save_path):
    f = open(save_path + 'results.txt', 'w')
    for i, auc in enumerate(auc_per_epochs):
        f.write('Epoch: {}. AUC: {}.'.format(i, auc))
    f.close()

