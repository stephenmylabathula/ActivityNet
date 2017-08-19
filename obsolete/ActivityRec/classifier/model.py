import tensorflow as tf


# Weights Matrix
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Bias Vector
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# Apply Depthwise Convolution
def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(tf.nn.depthwise_conv2d(x, weights, [1, 1, 1, 1], padding='VALID'), biases))


# Apply Max Pooling
def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')


# Create Classifier Model
def build_model(input_height, input_width, num_channels, num_labels, kernel_size, depth, num_hidden, learning_rate):
    # Input/Ouput
    X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
    Y = tf.placeholder(tf.float32, shape=[None,num_labels])

    # Convolution Layer 1
    c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
    p = apply_max_pool(c,20,2)

    # Convolution Layer 2
    c = apply_depthwise_conv(p,6,depth*num_channels,depth//10)

    # Fully Connected
    #   Hidden Layer 1
    shape = c.get_shape().as_list()
    c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])
    f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
    f_biases_l1 = bias_variable([num_hidden])
    #   Hidden Layer 2
    f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))
    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    # Output
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
    # Loss
    loss = -tf.reduce_sum(Y * tf.log(y_))
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(Y,1)), tf.float32))
    return (y_, accuracy, loss, optimizer, X, Y)
