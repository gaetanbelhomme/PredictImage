import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

# Import data :

pickle_file = 'Data2D.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset_start = save['train_dataset_start']
    train_dataset_end = save['train_dataset_end']
    valid_dataset_start = save['valid_dataset_start']
    valid_dataset_end = save['valid_dataset_end']
    test_dataset_start = save['test_dataset_start']
    test_dataset_end = save['test_dataset_end']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset_start.shape, train_dataset_end.shape)
    print('Validation set', valid_dataset_start.shape, valid_dataset_end.shape)
    print('Test set', test_dataset_start.shape, test_dataset_end.shape)

image_size = 64
num_channels = 1

print train_dataset_start


def reformat(dataset_start, dataset_end):
    dataset_start = dataset_start.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    dataset_end = dataset_end.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset_start, dataset_end


train_dataset_start, train_dataset_end = reformat(train_dataset_start, train_dataset_end)
valid_dataset_start, valid_dataset_end = reformat(valid_dataset_start, valid_dataset_end)
test_dataset_start, test_dataset_end = reformat(test_dataset_start, test_dataset_end)
print('Training set', train_dataset_start.shape, train_dataset_end.shape)
print('Validation set', valid_dataset_start.shape, valid_dataset_end.shape)
print('Test set', test_dataset_start.shape, test_dataset_end.shape)

print train_dataset_start


def accuracy(predict, truth):
    # test :
    print "NON ZERO ", np.count_nonzero(predict)
    testval1 = predict[7][30][30]
    testtruth = truth[7][30][30]
    print "TEST :", testval1, testtruth

    value = np.isclose(predict, truth, 0.01)
    nbTrue = np.sum(value)
    nbElements = np.size(predict)*1.0

    return 100.0*(nbTrue / nbElements)


batch_size = 8
patch_size = 4

depth1 = 32
depth2 = 64

num_layers = 2
last_image_size = image_size / (num_layers*2)

num_hidden = 2048
num_hidden2 = 10

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset_start = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_dataset_end = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_valid_dataset_start = tf.constant(valid_dataset_start)
    tf_test_dataset_start = tf.constant(test_dataset_start)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    def deconv2d(x, W, S):
        return tf.nn.conv2d_transpose(x, W, S, strides=[1, 2, 2, 1], padding='SAME')


    W_conv1 = weight_variable([patch_size, patch_size, num_channels, depth1])
    b_conv1 = bias_variable([depth1])

    W_conv2 = weight_variable([patch_size, patch_size, depth1, depth2])
    b_conv2 = bias_variable([depth2])

    W_fc = weight_variable([16 * 16 * depth2, num_hidden])
    b_fc = bias_variable([num_hidden])

    W_Ufc = weight_variable([num_hidden, 16 * 16 * depth2])
    b_Ufc = bias_variable([16 * 16 * depth2])

    W_deconv1 = weight_variable([patch_size, patch_size, depth1, depth2])
    b_deconv1 = bias_variable([depth1])

    W_deconv2 = weight_variable([patch_size, patch_size, 1, depth1])
    b_deconv2 = bias_variable([1])


    def model(data):
        with tf.name_scope('Conv1'):  # input : 64*64*1 ; output : 32*32*32


            h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            print '\ninput Conv1 : ', data.get_shape()
            print 'output Conv1 : ', h_pool1.get_shape()

        with tf.name_scope('Conv2'):  # input : 32*32*32 ; output : 16*16*64

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            print '\ninput Conv2 : ', h_pool1.get_shape()
            print 'output Conv2 : ', h_pool2.get_shape()

        with tf.name_scope('FullyConnected'):  # input : 16*16*64 ; output : 1*2048

            shape = h_pool2.get_shape().as_list()

            h_pool2_flat = tf.reshape(h_pool2, [-1, shape[1]*shape[2]*shape[3]])

            h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

            print '\ninput FullyConnected : ', h_pool2.get_shape()
            print 'middle flat FullyConnected : ', h_pool2_flat.get_shape()
            print 'output FullyConnected : ', h_fc.get_shape()

            # keep_prob = tf.placeholder(tf.float32)
            # h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        with tf.name_scope('UnFullyConnected'):  # input : 1*2048 ; output : 16*16*64


            # h_fc_drop_mat = tf.reshape(h_fc, [-1, last_image_size, last_image_size, depth2])
            # print 'middle mat UnFullyConnected : ', h_fc_drop_mat.get_shape()
            h_Ufc1 = tf.nn.relu(tf.matmul(h_fc, W_Ufc) + b_Ufc)

            h_Ufc2 = tf.reshape(h_Ufc1, [-1, shape[1], shape[2], shape[3]])

            print '\ninput UnFullyConnected : ', h_fc.get_shape()
            print 'middle UnFullyConnected : ', h_Ufc1.get_shape()
            print 'output UnFullyConnected : ', h_Ufc2.get_shape()

        with tf.name_scope('DeConv1'):  # input : 16*16*64 ; output : 32*32*32


            h_deconv1 = tf.nn.relu(deconv2d(h_Ufc2, W_deconv1, [shape[0], 32, 32, depth1]) + b_deconv1)

            print '\ninput Deconv1 : ', h_Ufc2.get_shape()
            print 'output Deconv1 : ', h_deconv1.get_shape()

        with tf.name_scope('DeConv2'):  # input : 32*32*32 ; output : 64*64*1

            h_deconv2 = tf.nn.relu(deconv2d(h_deconv1, W_deconv2, [shape[0], 64, 64, 1]) + b_deconv2)

            print '\ninput Deconv2 : ', h_deconv1.get_shape()
            print 'output Deconv2 : ', h_deconv2.get_shape()

        return h_deconv2


    # Training computation.
    train_output = model(tf_train_dataset_start)

    print "\ntrain output : ", train_output.get_shape()
    print "data : ", tf_train_dataset_end.get_shape()
    # loss = tf.abs(train_output - tf_train_dataset_end)

    # truth = tf.cast(tf_train_dataset_end, tf.int32)
    # loss = tf.nn.embedding_lookup(train_output, truth , partition_strategy='div')

    # loss = tf.cross(train_output,tf_train_dataset_end)

    # loss = tf.abs(tf.sub(tf.reduce_sum(tf_train_dataset_end), tf.reduce_sum(train_output)))

    # loss = tf.nn.l2_loss(tf_train_dataset_end - train_output)

    loss = tf.reduce_sum(tf.squared_difference(tf_train_dataset_end, train_output))*1.0

    # print "loss : ", loss

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # optimizer = tf.train.AdadeltaOptimizer(0.05).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(0.05). minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = train_output
    valid_prediction = model(tf_valid_dataset_start)
    test_prediction = model(tf_test_dataset_start)

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_dataset_start.shape[0] - batch_size)
        batch_data_start = train_dataset_start[offset:(offset + batch_size), :, :, :]
        batch_data_end = train_dataset_end[offset:(offset + batch_size), :, :, :]
        feed_dict = {tf_train_dataset_start: batch_data_start, tf_train_dataset_end: batch_data_end}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 50 == 0:
            print('\nMinibatch loss at step %d: %f' % (step, l))

            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_data_end))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_dataset_end))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_dataset_end))
