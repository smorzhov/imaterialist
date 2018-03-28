config = {
    'vgg16': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 50,
            'validation_steps': 15000
        }
    }
}