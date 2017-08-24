import os
import model
import config
import plotting
import data_provider
import numpy as np
import tensorflow as tf


# Load Parameters
parameters = config.LoadTrainingParameters()

# Load Data
train_x, train_y, test_x, test_y = data_provider.GenerateTrainTestData(sanity_plot=True)

# Build Model
y_, accuracy, loss, optimizer, X, Y = model.build_model(1, train_x.shape[2], train_x.shape[3], train_y.shape[1])

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
total_batches = train_x.shape[0] // parameters.batch_size
print "Training ..."
for epoch in range(parameters.epochs):
    for b in range(total_batches):
        print("Batch:", b)
        offset = (b * parameters.batch_size) % (train_y.shape[0] - parameters.batch_size)
        batch_x = train_x[offset:(offset + parameters.batch_size), :, :]
        batch_y = train_y[offset:(offset + parameters.batch_size), :]
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
confusion_matrix_img = plotting.PlotConfusionMatrix(np.array(confusion_matrix), ['Jump', 'Move', 'Sit', 'Stand', 'Walk'])
confusion_matrix_img = tf.image.decode_png(confusion_matrix_img.getvalue(), channels=4)
confusion_matrix_img = tf.expand_dims(confusion_matrix_img, 0)
image_summary_op = tf.summary.image("Test Confusion Matrix", confusion_matrix_img)
summary_writer.add_summary(session.run(image_summary_op), i)

summary_writer.close()
session.close()
