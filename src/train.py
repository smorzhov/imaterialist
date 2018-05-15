"""
Trains model

Usage: python train.py [-h]
"""
from argparse import ArgumentParser
from multiprocessing import cpu_count
from os import path
import pandas as pd
import numpy as np
from keras.utils import plot_model
from keras.callbacks import (EarlyStopping, ReduceLROnPlateau, TerminateOnNaN,
                             ModelCheckpoint)
from keras.preprocessing.image import ImageDataGenerator
from utils import (TEST_DATA_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH,
                   MODELS_PATH, CLASSES, try_makedirs, plot_loss_acc,
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
        help='model architecture (vgg16, vgg19, incresnet, incv3, xcept, resnet50, densnet, nasnet)',
        default='incresnet',
        type=str)
    parser.add_argument(
        '--test',
        nargs='?',
        help='test data path',
        default=TEST_DATA_PATH,
        type=str)
    parser.add_argument(
        '--train',
        nargs='?',
        help='train data path',
        default=TRAIN_DATA_PATH,
        type=str)
    parser.add_argument(
        '--validation',
        nargs='?',
        help='test data path',
        default=VALIDATION_DATA_PATH,
        type=str)
    return parser


def train_and_predict(model_type,
                      train_path=TRAIN_DATA_PATH,
                      test_path=TEST_DATA_PATH,
                      validation_path=VALIDATION_DATA_PATH):
    """
    Trains model and makes predictions file
    """
    # creating data generators
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rescale=1. / 255,
        shear_range=0.4,
        zoom_range=0.3,
        channel_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        classes=CLASSES,
        class_mode='categorical',
        seed=42,
        **config[model_type]['flow_generator'])
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False,
        **config[model_type]['flow_generator'])
    test_generator = test_datagen.flow_from_directory(
        test_path,
        class_mode=None,
        shuffle=False,
        **config[model_type]['flow_generator'])

    # loading the model
    parallel_model, model = get_model(model=model_type)
    print('Training model')
    print(model.summary())
    history = parallel_model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        callbacks=[
            # EarlyStopping(monitor='val_loss', min_delta=0, patience=5),
            ModelCheckpoint(
                path.join(MODELS_PATH, model_type +
                          '.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5'),
                save_weights_only=True,
                save_best_only=True),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=0.0000001),
            TerminateOnNaN()
        ],
        max_queue_size=100,
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
    val_preds = parallel_model.predict_generator(
        validation_generator,
        max_queue_size=100,
        use_multiprocessing=True,
        workers=cpu_count())
    plot_confusion_matrix(
        confusion_matrix(
            list(validation_generator.classes), np.argmax(val_preds, axis=1)),
        CLASSES, model_path)

    print('Generating predictions')
    predictions = parallel_model.predict_generator(
        test_generator,
        max_queue_size=100,
        use_multiprocessing=True,
        workers=cpu_count())
    pred_classes = np.argmax(predictions, axis=1)
    # Dealing with missing data
    ids = list(map(lambda id: id[5:-4], test_generator.filenames))
    proba = predictions[np.arange(len(predictions)), pred_classes]
    # Generating predictions.csv for Kaggle
    pd.DataFrame({
        'id': ids,
        'predicted': pred_classes,
    }).sort_values(by='id').to_csv(
        path.join(model_path, 'predictions.csv'), index=False)
    # Generating predictions.csv with some additional data for post-processing
    pd.DataFrame({
        'id': ids,
        'predicted': pred_classes,
        'proba': proba
    }).sort_values(by='id').to_csv(
        path.join(model_path, 'predictions_extd.csv'), index=False)


def main():
    """
    Main function
    """
    args = init_argparse().parse_args()
    train_and_predict(args.model, args.train, args.test, args.validation)


if __name__ == '__main__':
    main()
