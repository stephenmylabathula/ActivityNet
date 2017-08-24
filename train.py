import config
import model
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
        _, acc, cost = session.run([optimizer, accuracy, loss], feed_dict={X: batch_x, Y: batch_y})
    print "Epoch: ", epoch, " Training Loss: ", cost, " Training Accuracy: ", acc
test_acc, test_prediction = session.run([accuracy, y_], feed_dict={X: test_x, Y: test_y})
print "Testing Accuracy:", test_acc

# Generate Confusion Matrix for Test set
confusion_matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
for i in range(len(test_y)):
    confusion_matrix[np.argmax(test_y[i])][np.argmax(test_prediction[i])] += 1
print confusion_matrix

session.close()
