import keras.backend.tensorflow_backend as K
from keras.applications.vgg16 import VGG16
from keras.utils import multi_gpu_model

CLASSES = 128
"""
TODO
2. VGG19
3. ResNet50
4. Inception V3
5. Xception
"""


def get_gpus(gpus):
    """
    Returns a list of integers (numbers of gpus)
    """
    return list(map(int, gpus.split(',')))


def get_model(model, gpus=1, **kwargs):
    """
    Returns compiled keras parallel model ready for training
    and base model that must be used for saving weights

    Params:
    - model - model type
    - gpus - a list with numbers of GPUs
    """
    if model == 'vgg16':
        return vgg16(gpus)
    raise ValueError('Wrong model value!')


def vgg16(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    gpus = get_gpus(gpus)
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            model = VGG16(weights=None, classes=CLASSES)
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            # creates a model that includes
            model = VGG16(weights=None, classes=CLASSES)
        parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return parallel_model, model
