import keras.backend.tensorflow_backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from utils import CLASSES
"""
TODO
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
    if model == 'vgg16' or model == 'vgg19':
        return vgg(gpus, model)
    raise ValueError('Wrong model value!')


def vgg(gpus, model):
    """
    Returns compiled keras vgg16 model ready for training
    """
    if model == 'vgg16':
        base_model = VGG16(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        frozen = 14
    elif model == 'vgg19':
        base_model = VGG19(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        frozen = 16
    else:
        raise ValueError('Wrong VGG model type!')

    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(len(CLASSES), activation='softmax')(x)

    gpus = get_gpus(gpus)
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            model = Model(base_model.input, output)
            for layer in model.layers[:frozen]:
                layer.trainable = False
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            model = Model(base_model.input, output)
            for layer in model.layers[:frozen]:
                layer.trainable = False
        parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=0.00001),
        metrics=['accuracy'])
    return parallel_model, model
