import json
from collections import namedtuple


def LoadDataProvisionParameters():
    with open('config.json') as datafile:
        data = json.load(datafile)
    data_provider_params = namedtuple("DataParameters",
                                      ["window_size", "window_height", "step_size", "channels", "num_channels",
                                       "labels", "num_labels", "shuffle", "normalize"],
                                      verbose=False, rename=False)
    data_provider_params.window_size = data['data_provider']['window_size']
    data_provider_params.window_height = data['data_provider']['window_height']
    data_provider_params.step_size = data['data_provider']['step_size']
    data_provider_params.signals = data['data_provider']['signals']
    data_provider_params.channels = data['data_provider']['channels']
    data_provider_params.num_channels = len(data_provider_params.channels)
    data_provider_params.labels = data['data_provider']['labels']
    data_provider_params.num_labels = len(data_provider_params.labels)
    data_provider_params.shuffle = data['data_provider']['shuffle']
    data_provider_params.normalize = data['data_provider']['normalize']
    return data_provider_params


def LoadModelParameters():
    with open('config.json') as datafile:
        data = json.load(datafile)
    model_params = namedtuple("ModelParameters", ["type", "depth", "kernel", "hidden", "learning_rate"],
                              verbose=False, rename=False)
    model_params.type = data['model']['type']
    model_params.depth = data['model']['depth']
    model_params.kernel = data['model']['kernel']
    model_params.hidden = data['model']['hidden']
    model_params.learning_rate = data['model']['learning_rate']
    return model_params


def LoadTrainingParameters():
    with open('config.json') as datafile:
        data = json.load(datafile)
    training_params = namedtuple("TrainingParameters",
                                 ["batch_size", "epochs"], verbose=False, rename=False)
    training_params.batch_size = data['training']['batch_size']
    training_params.epochs = data['training']['epochs']
    return training_params