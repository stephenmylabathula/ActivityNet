import scipy.io
import numpy as np
import scipy.signal
import config
import matplotlib.pyplot as plt


def GenerateDataActionWindows():
    # Get Parameters
    parameters = config.LoadDataProvisionParameters()
    LoadDataset('20170708_1.mat', parameters)


# Load Dataset
def LoadDataset(file_name, parameters):
    data_file = scipy.io.loadmat('./data/training/' + file_name)
    jump_tags = data_file['dataset']['jump_tags'][0][0]
    move_tags = data_file['dataset']['move_tags'][0][0]
    sit_tags = data_file['dataset']['sit_tags'][0][0]
    stand_tags = data_file['dataset']['stand_tags'][0][0]
    walk_tags = data_file['dataset']['walk_tags'][0][0]
    jump_action_windows = GenerateJumpActions(data_file, jump_tags, parameters)
    # move_action_windows = GenerateMoveActions(data_file, move_tags, parameters)
    sit_acton_windows = GenerateSitActions(data_file, sit_tags, parameters)
    stand_action_windows = GenerateStandActions(data_file, stand_tags, parameters)
    walk_action_windows = GenerateWalkActions(data_file, walk_tags, parameters)
    for stand in stand_action_windows:
        plt.plot(stand[0])
        plt.show()


# Create Jump Sequence Array
def GenerateJumpActions(data_file, jump_tags, parameters):
    jump_actions = []
    for jump_tag in jump_tags:
        start_index = jump_tag[0]
        end_index = jump_tag[-1]
        jump_action = np.zeros((1, end_index - start_index), dtype=np.float32)
        for signal_index in range(len(parameters.signals)):
            for channel in parameters.channels[signal_index]:
                jump_action = np.vstack((jump_action,
                                         data_file['dataset']['data'][0][0][parameters.signals[signal_index]][0][0][
                                         start_index:end_index, channel]))
        jump_actions.append(jump_action[1:])
    return jump_actions


# Create Move Sequence Array
def GenerateMoveActions(data_file, move_tags, parameters):
    move_actions = []
    for move_tag in move_tags:
        start_index = move_tag[0]
        end_index = move_tag[-1]
        move_action = np.zeros((1, end_index - start_index), dtype=np.float32)
        for signal_index in range(len(parameters.signals)):
            for channel in parameters.channels[signal_index]:
                move_action = np.vstack((move_action,
                                         data_file['dataset']['data'][0][0][parameters.signals[signal_index]][0][0][
                                         start_index:end_index, channel]))
        move_actions.append(move_action[1:])
    return move_actions


# Create Sit Sequence Array
def GenerateSitActions(data_file, sit_tags, parameters):
    sit_actions = []
    for sit_tag in sit_tags:
        start_index = sit_tag[0]
        end_index = sit_tag[-1]
        sit_action = np.zeros((1, end_index - start_index), dtype=np.float32)
        for signal_index in range(len(parameters.signals)):
            for channel in parameters.channels[signal_index]:
                sit_action = np.vstack((sit_action,
                                         data_file['dataset']['data'][0][0][parameters.signals[signal_index]][0][0][
                                         start_index:end_index, channel]))
        sit_actions.append(sit_action[1:])
    return sit_actions


# Create Stand Sequence Array
def GenerateStandActions(data_file, stand_tags, parameters):
    stand_actions = []
    for stand_tag in stand_tags:
        start_index = stand_tag[0]
        end_index = stand_tag[-1]
        stand_action = np.zeros((1, end_index - start_index), dtype=np.float32)
        for signal_index in range(len(parameters.signals)):
            for channel in parameters.channels[signal_index]:
                stand_action = np.vstack((stand_action,
                                         data_file['dataset']['data'][0][0][parameters.signals[signal_index]][0][0][
                                         start_index:end_index, channel]))
        stand_actions.append(stand_action[1:])
    return stand_actions


# Create Walk Sequence Array
def GenerateWalkActions(data_file, walk_tags, parameters):
    walk_actions = []
    for walk_tag in walk_tags:
        start_index = walk_tag[0]
        end_index = walk_tag[-1]
        walk_action = np.zeros((1, end_index - start_index), dtype=np.float32)
        for signal_index in range(len(parameters.signals)):
            for channel in parameters.channels[signal_index]:
                walk_action = np.vstack((walk_action,
                                         data_file['dataset']['data'][0][0][parameters.signals[signal_index]][0][0][
                                         start_index:end_index, channel]))
        walk_actions.append(walk_action[1:])
    return walk_actions
