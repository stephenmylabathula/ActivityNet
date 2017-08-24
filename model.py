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
    elif activation == "Sigmoid":
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

    # Assert Valid Model
    assert len(parameters.convolution_kernel) == len(parameters.convolution_stride) == len(parameters.convolution_channel_multiplier)
    assert len(parameters.pooling_kernel) == len(parameters.pooling_stride)
    assert len(parameters.model_layout) == len(parameters.convolution_kernel) + len(parameters.pooling_kernel) + len(parameters.hidden) + 1

    # Build Model
    current_input = X
    current_channel_count = num_channels
    need_flatten = True     # the first hidden layer requires input to be flattened
    convolution_index = 0
    pooling_index = 0
    hidden_index = 0
    for layer_index in range(len(parameters.model_layout)):
        if parameters.model_layout[layer_index] == "C":     # if convolution layer
            conv_layer = depthwise_convolution_layer(current_input, parameters.convolution_kernel[convolution_index],
                                                     parameters.convolution_stride[convolution_index],
                                                     current_channel_count,
                                                     parameters.convolution_channel_multiplier[convolution_index])
            current_input = conv_layer      # update input for next layer
            current_channel_count *= parameters.convolution_channel_multiplier[convolution_index]   # update # channels
            convolution_index += 1      # increment index for next convolution layer
        elif parameters.model_layout[layer_index] == "P":   # if pooling layer
            pool_layer = max_pool_layer(current_input, parameters.pooling_kernel[pooling_index],
                                        parameters.pooling_stride[pooling_index])
            current_input = pool_layer      # update input for next layer
            pooling_index += 1      # increment index for next pooling layer
        elif parameters.model_layout[layer_index] == "H":
            if need_flatten:    # if first hidden layer, flatten input
                need_flatten = False
                current_input_layer_shape = current_input.get_shape().as_list()
                flattened_input = tf.reshape(current_input, [-1, current_input_layer_shape[1] *
                                                             current_input_layer_shape[2] *
                                                             current_input_layer_shape[3]])
                hidden_layer_weights = weight_variable((current_input_layer_shape[1] * current_input_layer_shape[2] *
                                                        current_input_layer_shape[3], parameters.hidden[hidden_index]))
                hidden_layer_bias = bias_variable([parameters.hidden[hidden_index]])
                # TODO: put activation in config
                fully_connected_layer = tf.nn.tanh(tf.add(tf.matmul(flattened_input, hidden_layer_weights),
                                                          hidden_layer_bias))
            else:
                hidden_layer_weights = weight_variable((parameters.hidden[hidden_index-1],
                                                        parameters.hidden[hidden_index]))
                hidden_layer_bias = bias_variable([parameters.hidden[hidden_index]])
                # TODO: put activation in config
                fully_connected_layer = tf.nn.tanh(tf.add(tf.matmul(current_input, hidden_layer_weights),
                                                          hidden_layer_bias))
            current_input = fully_connected_layer   # update input for next layer
            hidden_index += 1   # increment index for next hidden layer
        elif parameters.model_layout[layer_index] == "S":   # if softmax layer
            output_weights = weight_variable([parameters.hidden[-1], num_labels])
            output_bias = bias_variable([num_labels])
            current_input = tf.nn.softmax(tf.matmul(current_input, output_weights) + output_bias)
    # Prediction
    y_ = current_input
    # Loss
    cross_entropy = -tf.reduce_sum(Y * tf.log(y_))
    loss = cross_entropy
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate).minimize(loss)
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1)), tf.float32))
    return y_, accuracy, loss, optimizer, X, Y
