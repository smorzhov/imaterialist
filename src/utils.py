"""
Some useful utilities
"""
try:
    import cPickle as pickle
except ImportError:
    import pickle
from os import path, makedirs
from sklearn.metrics import confusion_matrix
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
RAW_DATA_PATH = path.join(CWD, 'data', 'raw')
"""
Processed test, train and other files used in training (testing) process
must be saved here.
By default, this directory is being ignored by GIT. It is not recommended
to exclude this directory from .gitignore unless there is no extreme necessity.
"""
PROCESSED_DATA_PATH = path.join(CWD, 'data', 'processed')
LOG_PATH = path.join(CWD, 'log')
"""
Trained models must be stored here
"""
MODELS_PATH = path.join(CWD, 'models')
"""
Pickled objects must be stored here
"""
PICKLES_PATH = path.join(CWD, 'pickles')


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


def plot_loss_acc(history, aucs, model_path=None):
    """
    Saves into files accuracy and loss plots
    """
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
    plt.plot(aucs)
    plt.title('model loss, ROC AUC')
    plt.ylabel('loss, ROC AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'ROC AUC'], loc='upper left')
    plt.savefig(path.join(model_path, 'loss.png'))