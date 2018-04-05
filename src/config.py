config = {
    'vgg16': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'vgg19': {
        'flow_generator': {
            'target_size': (224, 224),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 15
        }
    },
    'incresnet': {
        'flow_generator': {
            'target_size': (299, 299),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 20
        }
    },
    'incv3': {
        'flow_generator': {
            'target_size': (299, 299),
            'batch_size': 256
        },
        'fit_generator': {
            'epochs': 20
        }
    }
}