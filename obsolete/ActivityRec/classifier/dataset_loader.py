import scipy
import scipy.io
import scipy.stats
import numpy as np
import pandas as pd


num_channels = 10

def Normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def SlidingWindow(data, window, step):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1], )
    slided_windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return slided_windows[::step]

def GenerateDataset(window_size, step_size, holdout=0.3, dataset_list=['20170708_1.mat', '20170708_2.mat', '20170708_3.mat', '20170708_4.mat', '20170708_5.mat', '20170711_1.mat']):
    input_windows = np.empty([0, 1, window_size, num_channels])
    output_labels = np.empty([0])
    path_prefix = '/usr/local/google/home/smylabathula/Projects/ActivityRec/datasets/'

    for dataset_name in dataset_list:
        # Load Dataset
        raw_dataset = scipy.io.loadmat(path_prefix + dataset_name)

        # Extract Selected Channels
        accel_global = Normalize(raw_dataset['dataset']['data'][0][0]['accel_global'][0][0])
        velocity = Normalize(raw_dataset['dataset']['data'][0][0]['velocity'][0][0])
        position = Normalize(raw_dataset['dataset']['data'][0][0]['position'][0][0])
        omega_global = Normalize(raw_dataset['dataset']['data'][0][0]['omega_global'][0][0])

        # Generate Windowed Data and Reshape
        accel_windows = SlidingWindow(accel_global[:, 2], window_size, step_size)
        accel_windows = accel_windows.reshape(accel_windows.shape[0], 1, accel_windows.shape[1])

        velo_windows = SlidingWindow(velocity[:, 2], window_size, step_size)
        velo_windows = velo_windows.reshape(velo_windows.shape[0], 1, velo_windows.shape[1])

        pos_windows = SlidingWindow(position[:, 2], window_size, step_size)
        pos_windows = pos_windows.reshape(pos_windows.shape[0], 1, pos_windows.shape[1])

        accel_x_windows = SlidingWindow(accel_global[:, 0], window_size, step_size)
        accel_x_windows = accel_x_windows.reshape(accel_x_windows.shape[0], 1, accel_x_windows.shape[1])

        accel_y_windows = SlidingWindow(accel_global[:, 1], window_size, step_size)
        accel_y_windows = accel_y_windows.reshape(accel_y_windows.shape[0], 1, accel_y_windows.shape[1])

        gyro_x_windows = SlidingWindow(omega_global[:, 0], window_size, step_size)
        gyro_x_windows = gyro_x_windows.reshape(gyro_x_windows.shape[0], 1, gyro_x_windows.shape[1])

        gyro_y_windows = SlidingWindow(omega_global[:, 1], window_size, step_size)
        gyro_y_windows = gyro_y_windows.reshape(gyro_y_windows.shape[0], 1, gyro_y_windows.shape[1])

        gyro_z_windows = SlidingWindow(omega_global[:, 2], window_size, step_size)
        gyro_z_windows = gyro_z_windows.reshape(gyro_z_windows.shape[0], 1, gyro_z_windows.shape[1])

        pos_x_windows = SlidingWindow(position[:, 0], window_size, step_size)
        pos_x_windows = pos_x_windows.reshape(pos_x_windows.shape[0], 1, pos_x_windows.shape[1])

        pos_y_windows = SlidingWindow(position[:, 1], window_size, step_size)
        pos_y_windows = pos_y_windows.reshape(pos_y_windows.shape[0], 1, pos_y_windows.shape[1])

        # Concatenate Windows Channelwise
        action_windows = np.append(accel_windows, velo_windows, axis=1)
        action_windows = np.append(action_windows, pos_windows, axis=1)
        action_windows = np.append(action_windows, accel_x_windows, axis=1)
        action_windows = np.append(action_windows, accel_y_windows, axis=1)
        action_windows = np.append(action_windows, gyro_x_windows, axis=1)
        action_windows = np.append(action_windows, gyro_y_windows, axis=1)
        action_windows = np.append(action_windows, gyro_z_windows, axis=1)
        action_windows = np.append(action_windows, pos_x_windows, axis=1)
        action_windows = np.append(action_windows, pos_y_windows, axis=1)

        # Extract Action Tags
        action_tags = raw_dataset['dataset']['tags'][0][0].T[0]
        tag_windows = SlidingWindow(action_tags, window_size, step_size)

        # Generate Final I/O Data
        output_data = scipy.stats.mode(tag_windows, axis=1)[0].T[0]
        output_labels = np.append(output_labels, output_data, axis=0)
        input_data = action_windows.reshape(len(action_windows), 1, window_size, num_channels)
        input_windows = np.append(input_windows, input_data, axis=0)

        print "Generated", len(input_data), "windows for dataset", dataset_name

    # Convert to One-Hot
    output_labels = np.asarray(pd.get_dummies(output_labels), dtype = np.int8)

    # Perform Train/Test Split
    train_test_split = np.random.rand(len(input_windows)) > holdout
    train_x = input_windows[train_test_split]
    train_y = output_labels[train_test_split]
    test_x = input_windows[~train_test_split]
    test_y = output_labels[~train_test_split]

    return (train_x, train_y, test_x, test_y)