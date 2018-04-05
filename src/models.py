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
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from utils import CLASSES


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
    if model == 'incresnet':
        return inception_res_net_v2(gpus)
    if model == 'incv3':
        return inception_v3(gpus)
    if model == 'xcept':
        return xception(gpus)
    if model == 'resnet50':
        return resnet50(gpus)
    if model == 'densenet':
        return dense_net(gpus)
    if model == 'nasnet':
        return nasnet(gpus)
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

    x = Flatten(name='flatten')(base_model.output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output = Dense(len(CLASSES), activation='softmax')(x)

    return _compile(gpus, base_model.input, output, frozen)


def inception_v3(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    frozen = 249
    base_model = InceptionV3(
        weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)
    return _compile(gpus, base_model.input, output, frozen)


def inception_res_net_v2(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    frozen = 0  # TODO
    base_model = InceptionResNetV2(
        weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

    return _compile(gpus, base_model.input, output, frozen)


def xception(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    frozen = 125
    base_model = Xception(
        weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

    return _compile(gpus, base_model.input, output, frozen)


def resnet50(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    frozen = 0
    base_model = ResNet50(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = Flatten()(base_model.output)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

    return _compile(gpus, base_model.input, output, frozen)


def dense_net(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    frozen = 0
    base_model = DenseNet201(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

    return _compile(gpus, base_model.input, output, frozen)


def nasnet(gpus):
    """
    Returns compiled keras vgg16 model ready for training
    """
    frozen = 0
    base_model = NASNetLarge(
        weights='imagenet', include_top=False, input_shape=(331, 331, 3))

    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

    return _compile(gpus, base_model.input, output, frozen)


def _compile(gpus, input, output, frozen):
    gpus = get_gpus(gpus)
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
        parallel_model = multi_gpu_model(model, gpus=gpus)
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=0.0001),
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    return parallel_model, model
