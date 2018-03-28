config = {
    'vgg16': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'steps_per_epoch': 10000,
            'epochs': 50,
            'validation_steps': 3000
        }
    }
}