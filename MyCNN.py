import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from matplotlib import pyplot

def removeChannel(vect, size_of_image):
	new = vect.reshape(-1, size_of_image,size_of_image).astype(np.float)
	return new

def display(train_array):
	print "train array shape :", train_array.shape
	pyplot.figure(dpi=300)
	pyplot.set_cmap(pyplot.gray())
	pyplot.pcolormesh(np.flipud(train_array[:,:]))
	pyplot.show()

#### IMPORT DATA : ####
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
    
print ""
print("Training set", train_dataset_start.shape, train_dataset_end.shape)
print('Validation set', valid_dataset_start.shape, valid_dataset_end.shape)
print('Test set', test_dataset_start.shape, test_dataset_end.shape)


#### TEST : #####
#print train_dataset_start
#print valid_dataset_start

print ("")

##### IMAGE SETTINGS : #####
image_size = 64
num_channels = 1


#### RESIZE IMAGE : #### 
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


#### TEST : #####
#print train_dataset_start
#print valid_dataset_start


#### GET THE ACCURACY OF THE CNN : #####
def accuracy(predict, truth):
    ## BOOLARRAY : boolean array where predict & truth are element-wise equal within a tolerance ##
    boolArray = np.isclose(predict, truth, 5)
    ## NBTRUE : nb of the same pixels
    nbTrue = np.sum(boolArray)
    ## NBELEMENTS : nb pixels belong to one batch of Images
    nbElements = np.size(predict)*1.0

    ## TEST : Display one value of tensors ###
    print "NON ZERO ", np.count_nonzero(predict)
    # print "array : ", np.size(np.nonzero(predict)[0])
    testval1 = predict[7][30][30]
    testtruth = truth[7][30][30]
    # print "nbTrue", nbTrue
    # print "TEST :", testval1, testtruth
    # print "nbElements : ", nbElements

    return 100.0*(nbTrue / nbElements)


#### SETTINGS CNN : #####
batch_size = 8

## CONV: ###
patch_size = 3
depth1 = 32
depth2 = 64

num_layers = 2
last_image_size = image_size / (num_layers*2)

num_hidden = 2048
num_hidden2 = 10

graph = tf.Graph()


#### VARIABLES INIT : #####
with graph.as_default():
    ## INPUT DATA : ##
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


    glob_step = tf.Variable(0,trainable=False)


    W_conv1 = weight_variable([patch_size, patch_size, num_channels, depth1])
    b_conv1 = bias_variable([depth1])

    W_conv2 = weight_variable([patch_size, patch_size, depth1, depth2])
    b_conv2 = bias_variable([depth2])

    W_fc = weight_variable([16 * 16 * depth2, num_hidden])
    b_fc = bias_variable([num_hidden])
    # keep_prob = tf.placeholder(tf.float32)

    W_Ufc = weight_variable([num_hidden, 16 * 16 * depth2])
    b_Ufc = bias_variable([16 * 16 * depth2])

    W_deconv1 = weight_variable([patch_size, patch_size, depth1, depth2])
    b_deconv1 = bias_variable([depth1])

    W_deconv2 = weight_variable([patch_size, patch_size, 1, depth1])
    b_deconv2 = bias_variable([1])


    ## CNN CONFIGURATION : ##
    def model(data):
	    	## CONV 1 : ##
	    	## INPUT : 64*64*1  OUTPUT : 32*32*32 ##
	        with tf.name_scope('Conv1'):  
		            h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
		            h_pool1 = max_pool_2x2(h_conv1)

		            ## TEST : ##
		            print '\ninput  Conv1 : ', data.get_shape()
		            print 'output Conv1 : ', h_pool1.get_shape()

		    ## CONV 2 : ##
	    	## INPUT : 32*32*32  OUTPUT : 16*16*64 ##
	        with tf.name_scope('Conv2'):  
		            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		            h_pool2 = max_pool_2x2(h_conv2)

		            ## TEST : ##
		            print '\ninput  Conv2 : ', h_pool1.get_shape()
		            print 'output Conv2 : ', h_pool2.get_shape()

		    ## FULLYCONNECTED : ##
	    	## INPUT : 16*16*64  OUTPUT : 1*2048 ##     
	        with tf.name_scope('FullyConnected'):  
		            shape = h_pool2.get_shape().as_list()
		            h_pool2_flat = tf.reshape(h_pool2, [-1, shape[1]*shape[2]*shape[3]])
		            h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

		            # h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

		            ## TEST : ##
		            print '\ninput  FullyConnected : ', h_pool2.get_shape()
		            print 'middle flat FullyConnected : ', h_pool2_flat.get_shape()
		            print 'output FullyConnected : ', h_fc.get_shape()

		    ## UNFULLYCONNECTED : ##
	    	## INPUT : 1*2048  OUTPUT : 16*16*64 ##
	        with tf.name_scope('UnFullyConnected'):
		            h_Ufc1 = tf.nn.relu(tf.matmul(h_fc, W_Ufc) + b_Ufc)
		            h_Ufc2 = tf.reshape(h_Ufc1, [-1, shape[1], shape[2], shape[3]])

		            ## TEST : ##
		            print '\ninput  UnFullyConnected : ', h_fc.get_shape()
		            print 'middle UnFullyConnected : ', h_Ufc1.get_shape()
		            print 'output UnFullyConnected : ', h_Ufc2.get_shape()

		    ## UNCONV 1 : ##
	    	## INPUT : 16*16*64  OUTPUT : 32*32*32 ##
	        with tf.name_scope('DeConv1'):
		            h_deconv1 = tf.nn.relu(deconv2d(h_Ufc2, W_deconv1, [shape[0], 32, 32, depth1]) + b_deconv1)

		            ## TEST : ##
		            print '\ninput  Deconv1 : ', h_Ufc2.get_shape()
		            print 'output Deconv1 : ', h_deconv1.get_shape()

		    ## UNCONV 2 : ##
	    	## INPUT : 32*32*32  OUTPUT : 64*64*1 ##
	        with tf.name_scope('DeConv2'):
		            h_deconv2 = deconv2d(h_deconv1, W_deconv2, [shape[0], 64, 64, 1]) + b_deconv2

		            ## TEST : ##
		            print '\ninput  Deconv2 : ', h_deconv1.get_shape()
		            print 'output Deconv2 : ', h_deconv2.get_shape()

	        return h_deconv2


    ## TRAINING COMPUTATION : ##
    train_output = model(tf_train_dataset_start)

    ## SEVERAL LOSS FUNCTIONS : ##
    # loss = tf.reduce_sum(tf.abs(tf.sub(train_output,tf_train_dataset_end)))
    # loss = tf.reduce_sum(tf.cross(train_output,tf_train_dataset_end))
    # loss = tf.abs(tf.sub(tf.reduce_sum(tf_train_dataset_end), tf.reduce_sum(train_output)))
    # loss = tf.reduce_sum(tf.squared_difference(tf_train_dataset_end, train_output))
    loss = tf.reduce_sum(tf.nn.l2_loss(tf.abs(tf.sub(tf_train_dataset_end,train_output))))
    # loss = tf.reduce_sum(tf.nn.log_poisson_loss(train_output, tf_train_dataset_end))

    ## LEARNING RATE : ##
    learning_rate = tf.train.exponential_decay(0.1, glob_step, 10, 0.9)

    ## SEVERAL OPTIMIZERS :
    # optimizer = tf.train.AdadeltaOptimizer(0.05).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(0.05). minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=glob_step)

    ## PREDICTIONS FOR THE TRAINING, VALIDATION, AND TEST DATA : ##
    train_prediction = train_output
    valid_prediction = model(tf_valid_dataset_start)
    test_prediction = model(tf_test_dataset_start)


#### TRAINING : ####
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

        if step % 15 == 0:
            print('\nMinibatch loss at step %d: \x1b[4;0;47m %f \x1b[0m' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(batch_data_end, batch_data_end))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_dataset_end))
            display(removeChannel(predictions, image_size)[0])

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_dataset_end))

