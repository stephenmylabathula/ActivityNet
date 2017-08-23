import os
import scipy.io
import numpy as np
import scipy.signal
import config


# Load Dataset
def LoadDataset(file_name, parameters):
    # Read Data File and Extract Desired Signals
    data_file = scipy.io.loadmat('./data/training/' + file_name)
    data = data_file['dataset']['data'][0][0]
    tags = data_file['dataset']['tags'][0][0].T[0]  # shape into 1d array
    selected_signals_data = np.vstack([data[parameters.signals[i]][0][0][:, j]
                                       for i in range(len(parameters.signals))
                                       for j in parameters.channels[i]])
    # Generate Sliding Windows
    signal_windows = np.expand_dims(GenerateWindowSlides(selected_signals_data[0],
                                                         parameters.window_size, parameters.step_size), axis=0)
    for i in range(1, len(selected_signals_data)):
        signal_windows = np.append(signal_windows,
                                   np.expand_dims(GenerateWindowSlides(selected_signals_data[i],
                                                                       parameters.window_size,
                                                                       parameters.step_size), axis=0), axis=0)
    action_tag_windows = GenerateWindowSlides(tags, parameters.window_size, parameters.step_size)

    # Generate Input/Output Dataset
    input_signal_data = signal_windows.swapaxes(0, 1)
    output_class_data = action_tag_windows[:, -1]

    return input_signal_data, output_class_data


# Sliding Window
def GenerateWindowSlides(data, window, step):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    slided_windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return slided_windows[::step]


def GenerateInputOutputData():
    # Retrieve Parameters
    parameters = config.LoadDataProvisionParameters()

    # Retrieve Training Data Files
    data_files = os.listdir('./data/training')
    X, Y = LoadDataset(data_files[0], parameters)

    # Generate I/O Data
    for i in range(1, len(data_files)):
        x, y = LoadDataset(data_files[i], parameters)
        X = np.append(x, X, axis=0)
        Y = np.append(y, Y, axis=0)

    return X, Y
