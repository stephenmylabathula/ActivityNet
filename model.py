import tensorflow as tf



















def BuildCNN(data_params, model_params):
    # Define Input/Output Placeholders
    X = tf.placeholder(tf.float32,
                       shape=[None, data_params.window_height, data_params.window_size, data_params.num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, data_params.num_labels])

    #