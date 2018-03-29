"""
Trains model

Usage: python train.py [-h]
"""
from argparse import ArgumentParser
from multiprocessing import cpu_count
from os import path, environ
import pandas as pd
import numpy as np
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from utils import (TEST_DATA_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH,
                   MODELS_PATH, try_makedirs, plot_loss_acc,
                   plot_confusion_matrix)
from sklearn.metrics import confusion_matrix
from models import get_model
from config import config


def init_argparse():
    """
    Initializes argparse

    Returns parser
    """
    parser = ArgumentParser(description='Trains toxic comment classifier')
    parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help='model architecture (vgg16, )',
        default='vgg16',
        type=str)
    parser.add_argument(
        '--gpus',
        nargs='?',
        help="A list of GPU device numbers ('1', '1,2,5')",
        default=0,
        type=str)
    return parser


def train_and_predict(model_type, gpus):
    """
    Trains model and makes predictions file
    """
    # creating data generators
    train_datagen = ImageDataGenerator(
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_PATH,
        class_mode='categorical',
        seed=171717,
        **config[model_type]['flow_generator'])
    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_PATH,
        class_mode='categorical',
        shuffle=False,
        **config[model_type]['flow_generator'])
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        class_mode='categorical',
        shuffle=False,
        **config[model_type]['flow_generator'])

    # loading the model
    parallel_model, model = get_model(model=model_type, gpus=gpus)
    print('Training model')
    print(model.summary())
    history = parallel_model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        callbacks=[
            EarlyStopping(monitor='val_loss', min_delta=0, patience=7),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
        ],
        use_multiprocessing=True,
        workers=cpu_count(),
        **config[model_type]['fit_generator'])
    # history of training
    # print(history.history.keys())
    # Saving architecture + weights + optimizer state
    model_path = path.join(MODELS_PATH, '{}_{:.4f}_{:.4f}'.format(
        model_type, history.history['val_loss'][-1]
        if 'val_loss' in history.history else history.history['loss'][-1],
        history.history['val_acc'][-1]
        if 'val_acc' in history.history else history.history['acc'][-1]))
    try_makedirs(model_path)
    plot_model(model, path.join(model_path, 'model.png'), show_shapes=True)
    plot_loss_acc(history, model_path)

    print('Saving model')
    model.save(path.join(model_path, 'model.h5'))
    # Building confusion matrices for every class for validation data
    print("Building confusion matrices")
    val_preds = model.predict_generator(validation_generator)
    plot_confusion_matrix(
        confusion_matrix(validation_generator.classes, np.argmax(val_preds)),
        map(str, range(1, 128 + 1)),
        model_path,
        normalize=True)

    print('Generating predictions')
    predictions = model.predict_generator(test_generator)
    pd.DataFrame({
        'id': test_generator.filenames,
        'predicted': np.argmax(predictions)
    }).sort_values(by='id').to_csv(
        path.join(model_path, 'predictions.csv'), index=False)


def main():
    """
    Main function
    """
    args = init_argparse().parse_args()

    environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    train_and_predict(args.model, args.gpus)


if __name__ == '__main__':
    main()
