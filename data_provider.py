import os
import scipy.io
import numpy as np
import scipy.signal
import config
import pandas as pd
import scipy.stats
import plotting


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
    if parameters.window_tag_method == "MODE":
        output_class_data = scipy.stats.mode(action_tag_windows, axis=1)[0].flatten()
    elif parameters.window_tag_method == "LAST":
        output_class_data = action_tag_windows[:, -1]
    else:
        raise ValueError("Window Tag Method " + parameters.window_tag_method + " does not match either MODE or LAST.")

    return input_signal_data, output_class_data


# Sliding Window Function
def GenerateWindowSlides(data, window, step):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    slided_windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return slided_windows[::step]


# Generate X/Y Data
def GenerateInputOutputData(one_hot=True, sanity_plot=False):
    # Retrieve Parameters
    parameters = config.LoadDataProvisionParameters()

    # Retrieve Training Data Files
    data_files = os.listdir('./data/training')
    X, Y = LoadDataset(data_files[0], parameters)

    # Generate I/O Data
    for i in range(1, len(data_files)):
        x, y = LoadDataset(data_files[i], parameters)
        X = np.append(X, x, axis=0)
        Y = np.append(Y, y, axis=0)
    X = np.expand_dims(X, axis=0).swapaxes(0, 1).swapaxes(2, 3)     # conform to NHWC shape
    Y[Y == 0] = 2   # equalize move and idle

    if sanity_plot:
        plotting.PlotSequentialInputOutputWindowedData(X, Y, parameters.window_size,
                                                       parameters.window_size // parameters.step_size)

    if one_hot:
        return X, pd.get_dummies(Y).as_matrix()
    return X, Y


# Generate X/Y Data Split for Learning and Evaluation
def GenerateTrainTestData(holdout=0.3, sanity_plot=False):
    # Get Input/Output Data
    X, Y = GenerateInputOutputData(sanity_plot)

    # Split Data
    train_test_split = np.random.rand(len(Y)) > holdout     # true: training, false: testing
    train_x = X[train_test_split]
    train_y = Y[train_test_split]
    test_x = X[~train_test_split]
    test_y = Y[~train_test_split]

    return train_x, train_y, test_x, test_y
