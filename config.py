import json
from collections import namedtuple


def LoadDataProvisionParameters():
    with open('config.json') as datafile:
        data = json.load(datafile)
    data_provider_params = namedtuple("DataParameters",
                                      ["window_size", "window_height", "step_size", "channels", "num_channels",
                                       "labels", "window_tag_method", "num_labels", "shuffle", "normalize"],
                                      verbose=False, rename=False)
    data_provider_params.window_size = data['data_provider']['window_size']
    data_provider_params.window_height = data['data_provider']['window_height']
    data_provider_params.step_size = data['data_provider']['step_size']
    data_provider_params.signals = data['data_provider']['signals']
    data_provider_params.channels = data['data_provider']['channels']
    data_provider_params.num_channels = len(data_provider_params.channels)
    data_provider_params.labels = data['data_provider']['labels']
    data_provider_params.window_tag_method = data['data_provider']['window_tag_method']
    data_provider_params.num_labels = len(data_provider_params.labels)
    data_provider_params.shuffle = data['data_provider']['shuffle']
    data_provider_params.normalize = data['data_provider']['normalize']
    return data_provider_params


def LoadModelParameters():
    with open('config.json') as datafile:
        data = json.load(datafile)
    model_params = namedtuple("ModelParameters", ["type", "convolution_kernel", "convolution_stride",
                                                  "convolution_channel_multiplier", "pooling_kernel", "pooling_stride",
                                                  "hidden", "model_layout", "learning_rate"],
                              verbose=False, rename=False)
    model_params.type = data['model']['type']
    model_params.convolution_kernel = data['model']['convolution_kernel']
    model_params.convolution_stride = data['model']['convolution_stride']
    model_params.convolution_channel_multiplier = data['model']['convolution_channel_multiplier']
    model_params.pooling_kernel = data['model']['pooling_kernel']
    model_params.pooling_stride = data['model']['pooling_stride']
    model_params.hidden = data['model']['hidden']
    model_params.model_layout = data['model']['model_layout']
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
