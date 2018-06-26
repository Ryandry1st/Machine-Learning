# Convolutional Neural Network
# similar to tensor flow Deep MNIST for experts
# Made of hidden layers which are pools of convolutions of input data
# then a fully connected layer and an output layer
# the fully connected and output layer are like normal perceptron layers

# pooling is kind of simplifying the grouping
# should be better than RNN with bigger datasets
# gives 98.5% accuracy with 10 epochs

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128
hm_epochs = 10
nodes = 1024

x = tf.placeholder('float',[None, 784]) # will throw an error if the shape is not correct
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') #strides is the movement, so 1 pixel
                                                    # padding is if there is a bigger pool than the length of data remaining
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 2*2 pooling, moving 2*2


def convolutional_neural_network(x):
    # tensor of weights with random values
    weights = {'W_conv1':tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)), # 5x5 conv,1 input, 32 features
               'W_conv2':tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
               'W_fc':tf.Variable(tf.truncated_normal([7*7*64, nodes], stddev=0.1)), # smaller feature map from 28*28
               'out':tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1))}
    
    biases = {'b_conv1':tf.Variable(tf.truncated_normal([32], stddev=0.1)), # 5x5 conv,1 input, 32 features
               'b_conv2':tf.Variable(tf.truncated_normal([64], stddev=0.1)),
               'b_fc':tf.Variable(tf.truncated_normal([1024], stddev=0.1)), # smaller feature map from 28*28
               'out':tf.Variable(tf.truncated_normal([n_classes], stddev=0.1))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = conv2d(conv1, weights['W_conv2']) + biases['b_conv2']
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    fc = tf.nn.dropout(fc, keep_rate) # may lower accuracy, but good if weights get too crazy

    output = tf.matmul(fc, weights['out'])+biases['out']


    return output

def train_neural_net(x):
    prediction = convolutional_neural_network(x)
    saver = tf.train.Saver()
    tf_log = 'CNN.log'
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
            print('Starting:', epoch)
        except:
            epoch = 1
            
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "./cnnmodel.ckpt")
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                
            saver.save(sess, "./cnnmodel.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            with open(tf_log, 'a') as f:
                f.write(str(epoch)+'\n')
            epoch+=1

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_net(x)
    

    
    
    
