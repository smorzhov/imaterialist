from os import environ
import keras.backend.tensorflow_backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from utils import CLASSES


def get_gpus(gpus):
    """
    Returns a list of integers (numbers of gpus)
    """
    return list(map(int, gpus.split(',')))


def get_model(model, **kwargs):
    """
    Returns compiled keras parallel model ready for training
    and base model that must be used for saving weights

    Params:
    - model - model type
    """
    if model == 'vgg16' or model == 'vgg19':
        return vgg(model)
    if model == 'incresnet':
        return inception_res_net_v2()
    if model == 'incv3':
        return inception_v3()
    if model == 'xcept':
        return xception()
    if model == 'resnet50':
        return resnet50()
    if model == 'densenet':
        return dense_net()
    if model == 'nasnet':
        return nasnet()
    raise ValueError('Wrong model value!')


def vgg(model):
    """
    Returns compiled keras model ready for training
    """
    if model == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg')
        frozen = 14
    elif model == 'vgg19':
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg')
        frozen = 16
    else:
        raise ValueError('Wrong VGG model type!')

    x = Dense(4096, activation='relu', name='fc1')(base_model.output)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output = Dense(len(CLASSES), activation='softmax')(x)

    return _compile(base_model.input, output, frozen)


def inception_v3():
    """
    Returns compiled keras model ready for training
    """
    frozen = 249
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3),
        pooling='avg')

    x = Dense(1024, activation='relu')(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)
    return _compile(base_model.input, output, frozen)


def inception_res_net_v2():
    """
    Returns compiled keras model ready for training
    """
    frozen = 0  # TODO
    base_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3),
        pooling='avg')

    output = Dense(
        len(CLASSES), activation='softmax',
        name='predictions')(base_model.output)

    return _compile(base_model.input, output, frozen)


def xception():
    """
    Returns compiled keras model ready for training
    """
    frozen = 125
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3),
        pooling='avg')

    x = Dense(1024, activation='relu')(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

    return _compile(base_model.input, output, frozen)


def resnet50():
    """
    Returns compiled keras model ready for training
    """
    frozen = 0
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg')

    output = Dense(
        len(CLASSES), activation='softmax',
        name='predictions')(base_model.output)

    return _compile(base_model.input, output, frozen)


def dense_net():
    """
    Returns compiled keras model ready for training
    """
    frozen = 0
    base_model = DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg')

    output = Dense(
        len(CLASSES), activation='softmax',
        name='predictions')(base_model.output)

    return _compile(base_model.input, output, frozen)


def nasnet():
    """
    Returns compiled keras model ready for training
    """
    frozen = -44
    base_model = NASNetLarge(
        weights='imagenet',
        include_top=False,
        input_shape=(331, 331, 3),
        pooling='avg')

    output = Dense(
        len(CLASSES), activation='softmax',
        name='predictions')(base_model.output)

    return _compile(base_model.input, output, frozen)


def _compile(input, output, frozen):
    gpus = get_gpus(environ['CUDA_VISIBLE_DEVICES'])
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            model = Model(input, output)
            for layer in model.layers[:frozen]:
                layer.trainable = False
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            model = Model(input, output)
            for layer in model.layers[:frozen]:
                layer.trainable = False
        # gpus=len(gpus) - simple workaround for reassigned GPUs ids.
        # While passing CUDA_VISIBLE_DEVICES as environment variable
        # it is assumed that all specified GPUs will be used during training
        parallel_model = multi_gpu_model(model, gpus=len(gpus))
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=0.001),
        # optimizer=Adam(lr=0.001, decay=0.0001),
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    return parallel_model, model
