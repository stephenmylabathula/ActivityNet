import tensorflow as tf
import config


# Weights Matrix
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Bias Vector
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# Depthwise Convolution
def depthwise_convolution_layer(x, kernel_size, stride_size, num_input_channels, channel_multiplier, activation="ReLU"):
    convolution_filter = weight_variable([1, kernel_size, num_input_channels, channel_multiplier])
    biases = bias_variable([num_input_channels * channel_multiplier])
    if activation == "ReLU":
        return tf.nn.relu(tf.add(tf.nn.depthwise_conv2d(x, convolution_filter, [1, 1, stride_size, 1],
                                                        padding='VALID'), biases))
    else:
        return tf.nn.sigmoid(tf.add(tf.nn.depthwise_conv2d(x, convolution_filter, [1, 1, stride_size, 1],
                                                           padding='VALID'), biases))


# Max Pooling
def max_pool_layer(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


# Create Classifier Model
def build_model(input_height, input_width, num_channels, num_labels):
    # Load Parameters
    parameters = config.LoadModelParameters()

    # Input/Output
    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    # Convolution Layer 1
    convolution_layer_1 = depthwise_convolution_layer(X, parameters.kernel, parameters.stride,
                                                      num_channels, parameters.channel_multiplier)
    pooling_layer_1 = max_pool_layer(convolution_layer_1, 20, 2)

    # Convolution Layer 2
    convolution_layer_2 = depthwise_convolution_layer(pooling_layer_1, 6, 1, parameters.channel_multiplier * num_channels,
                                                      parameters.channel_multiplier // 10)

    # Fully Connected
    #   Hidden Layer 1
    shape = convolution_layer_2.get_shape().as_list()
    c_flat = tf.reshape(convolution_layer_2, [-1, shape[1] * shape[2] * shape[3]])
    f_weights_l1 = weight_variable([shape[1] * shape[2] * parameters.channel_multiplier * num_channels *
                                    (parameters.channel_multiplier//10), parameters.hidden])
    f_biases_l1 = bias_variable([parameters.hidden])
    #   Hidden Layer 2
    f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))
    out_weights = weight_variable([parameters.hidden, num_labels])
    out_biases = bias_variable([num_labels])
    # Output
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
    # Loss
    loss = -tf.reduce_sum(Y * tf.log(y_))
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate).minimize(loss)
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1)), tf.float32))
    return y_, accuracy, loss, optimizer, X, Y
