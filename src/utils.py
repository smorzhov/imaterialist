"""
Some useful utilities
"""
try:
    import cPickle as pickle
except ImportError:
    import pickle
from itertools import product
from os import path, makedirs
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import exposure
import matplotlib
# generates images without having a window appear
matplotlib.use('Agg')
import matplotlib.pylab as plt
"""
Absolute utils.py file path. It is considered as the project root path.
"""
CWD = path.dirname(path.realpath(__file__))
"""
It must contain files with raw data
"""
DATA_PATH = path.join(CWD, 'data')
TEST_DATA_PATH = path.join(DATA_PATH, 'test')
TRAIN_DATA_PATH = path.join(DATA_PATH, 'train')
VALIDATION_DATA_PATH = path.join(DATA_PATH, 'validation')

LOG_PATH = path.join(CWD, 'log')
"""
Trained models must be stored here
"""
MODELS_PATH = path.join(CWD, 'models')
"""
Pickled objects must be stored here
"""
PICKLES_PATH = path.join(CWD, 'pickles')
CLASSES = list(map(str, range(1, 128 + 1)))


def def ahe(img):
    """
    Performs adaptive histogram equalization

    Returns processed image
    """
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


def try_makedirs(name):
    """
    Makes path if it doesn't exist
    """
    try:
        if not path.exists(name):
            # Strange, but it may raise winerror 123
            makedirs(name)
    except OSError:
        return


def plot_loss_acc(history, model_path):
    """
    Saves into files accuracy and loss plots
    """
    plt.gcf().clear()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path.join(model_path, 'accuracy.png'))
    plt.gcf().clear()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path.join(model_path, 'loss.png'))
    plt.gcf().clear()


def plot_confusion_matrix(cm, classes, model_path, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.gcf().clear()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='none')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=2)
    plt.yticks(tick_marks, classes, fontsize=2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.savefig(path.join(model_path, 'confusion_matrix.pdf'), format='pdf')

    plt.gcf().clear()
