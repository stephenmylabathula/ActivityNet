import os
import model
import plotting
import numpy as np
import dataset_loader
import tensorflow as tf
import matplotlib.pyplot as plt


# Initialize Classifier Variables
#   Data Parameters
num_labels = 5
input_height = 1
input_width = 200
input_num_channels = 3
#   Model Parameters
depth = 60
kernel_size = 60
num_hidden = 1000
#   Training Parameters
batch_size = 100
training_epochs = 8
learning_rate = 0.0001


# Load features.
# Read the array from disk
feature = np.genfromtxt('v1_train_features0.csv', delimiter=',')
input_height = 1
input_width = 200
num_channels = input_num_channels
num_samples =  feature.shape[0] // input_height
feature = feature.reshape((num_samples, input_height, num_channels, input_width))
train_x = np.swapaxes(feature,3,2)
# Load labels.
train_y = np.genfromtxt('v1_train_labels.csv', delimiter=',')


feature = np.genfromtxt('v1_test_features0.csv', delimiter=',')
input_height = 1
input_width = 200
num_channels = input_num_channels
num_samples =  feature.shape[0] // input_height
feature = feature.reshape((num_samples, input_height, num_channels, input_width))
test_x = np.swapaxes(feature,3,2)
# Load labels.
test_y = np.genfromtxt('v1_test_labels.csv', delimiter=',')



# Load Data
#(train_x, train_y, test_x, test_y) = dataset_loader.GenerateDataset(window_size=input_width, step_size=10, holdout=0.1, dataset_list=['20170708_1.mat', '20170708_2.mat', '20170708_3.mat', '20170708_4.mat', '20170708_5.mat', '20170711_1.mat'])

# Build Model
(y_, accuracy, loss, optimizer, X, Y) = model.build_model(input_height, input_width, input_num_channels, num_labels, kernel_size, depth, num_hidden, learning_rate)




# Start Tensorflow Session
session = tf.Session()

# Generate Tensorboard Summary
summary_writer = tf.summary.FileWriter('tensorboard', session.graph)
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
with tf.name_scope('Loss'):
    tf.summary.scalar('Loss', loss)
with tf.name_scope('Accuracy'):
    tf.summary.scalar('Accuracy', accuracy)
merged_summary = tf.summary.merge_all()

# Start Training
tf.global_variables_initializer().run(session=session)
total_batches = train_x.shape[0] // batch_size
print "Training ..."
for epoch in range(training_epochs):
    for b in range(total_batches):
        print("Batch:", b)
        offset = (b * batch_size) % (train_y.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :, :]
        batch_y = train_y[offset:(offset + batch_size), :]
        _, acc, cost, summary = session.run([optimizer, accuracy, loss, merged_summary], feed_dict={X: batch_x, Y: batch_y})
    print "Epoch: ", epoch, " Training Loss: ", cost, " Training Accuracy: ", acc
    # Write Summary to Tensorboard
    summary_writer.add_summary(summary, epoch)
test_acc, test_prediction = session.run([accuracy, y_], feed_dict={X: test_x, Y: test_y})
print "Testing Accuracy:", test_acc


# Generate Confusion Matrix for Test set
confusion_matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
for i in range(len(test_y)):
    confusion_matrix[np.argmax(test_y[i])][np.argmax(test_prediction[i])] += 1
confusion_matrix_img = plotting.PlotConfusionMatrix(np.array(confusion_matrix), ['Walk', 'Stand', 'Sit', 'Move', 'Jump'])
confusion_matrix_img = tf.image.decode_png(confusion_matrix_img.getvalue(), channels=4)
confusion_matrix_img = tf.expand_dims(confusion_matrix_img, 0)
image_summary_op = tf.summary.image("Test Confusion Matrix", confusion_matrix_img)
summary_writer.add_summary(session.run(image_summary_op), i)


# Generate Validation Set Confusion Matrix
(train_x, train_y, _, _) = dataset_loader.GenerateDataset(window_size=input_width, step_size=10,
                                                                    holdout=0.0, dataset_list=['20170708_6.mat'])
action_pred = session.run(y_, feed_dict={X: train_x})

validation_confusion_matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
for i in range(len(train_y)):
    validation_confusion_matrix[np.argmax(train_y[i])][np.argmax(action_pred[i])] += 1
validation_confusion_matrix_img = plotting.PlotConfusionMatrix(np.array(validation_confusion_matrix), ['Walk', 'Stand', 'Sit', 'Move', 'Jump'])
validation_confusion_matrix_img = tf.image.decode_png(validation_confusion_matrix_img.getvalue(), channels=4)
validation_confusion_matrix_img = tf.expand_dims(validation_confusion_matrix_img, 0)
image_summary_op = tf.summary.image("Validation Confusion Matrix", validation_confusion_matrix_img)
summary_writer.add_summary(session.run(image_summary_op), i)

action_sequence_img = plotting.PlotActionSequence(np.argmax(train_y, axis=1), np.argmax(action_pred, axis=1))
action_sequence_img = tf.image.decode_png(action_sequence_img.getvalue(), channels=4)
action_sequence_img = tf.expand_dims(action_sequence_img, 0)
image_summary_op = tf.summary.image("Action Sequence", action_sequence_img)
summary_writer.add_summary(session.run(image_summary_op), i)

#plt.plot(train_y)
#plt.show()
#plt.plot(action_pred)
#plt.show()

summary_writer.close()
session.close()