import os
import plotting
import numpy as np
import tensorflow as tf


# Setup Tensorboard and Generate Summary Writer
def GetTensorboardSummaryWriter(session):
    summary_writer = tf.summary.FileWriter('tensorboard', session.graph)
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    return summary_writer


# Add Scalars to Tensorboard
def AddScalars(nodes, names):
    assert len(nodes) == len(names)
    for i in range(len(nodes)):
        with tf.name_scope(names[i]):
            tf.summary.scalar(names[i], nodes[i])


# Returns Merged Summary of all Summary Nodes
def GetTensorboardMergedSummary():
    merged_summary = tf.summary.merge_all()
    return merged_summary


# Delete Tensorboard Summaries
def DeleteSummaries():
    for event in os.listdir('tensorboard'):
        os.remove('tensorboard/' + event)


# Generate Confusion Matrix
def AddConfusionMatrixImage(test_prediction, test_y, action_titles):
    confusion_matrix = np.zeros((len(action_titles), len(action_titles)))
    for i in range(len(test_y)):
        confusion_matrix[np.argmax(test_y[i])][np.argmax(test_prediction[i])] += 1
    confusion_matrix_img = plotting.PlotConfusionMatrix(np.array(confusion_matrix), action_titles)
    confusion_matrix_img = tf.image.decode_png(confusion_matrix_img.getvalue(), channels=4)
    confusion_matrix_img = tf.expand_dims(confusion_matrix_img, 0)
    image_summary_op = tf.summary.image("Test Confusion Matrix", confusion_matrix_img)
    return image_summary_op
