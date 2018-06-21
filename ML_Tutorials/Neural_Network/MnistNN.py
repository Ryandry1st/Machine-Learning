# number of inputs are not big but the number of samples are what matters
# Some have data pools of numbers like 500 million samples
# imagenet is a source for image data
# for text data you can go to wikipedia data dump or reddit dumps
# commoncrawl is a huge repository for places like aws massive servers, full of petabytes of data

'''
feed forward NN
input > weighted > hidden layer 1 ( activation function) > weights > HL2 (activation) ....
 > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimzer... SGD, AdaGrad)

backpropagation
feed forward + backprop = epoch
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# good for multiclasses, we have 10 which are handwritten 0-9 values

n_nodes_hl1 = 400 #these can be different and whatever you like
n_nodes_hl2 = 200
n_nodes_hl3 = 200
n_nodes_hl4 = 200

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784]) # will throw an error if the shape is not correct
y = tf.placeholder('float')

def neural_network(data):
    # tensor of weights with random values
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl1], stddev=0.1)),
                      'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl3]))}

    hidden_4_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_nodes_hl4], stddev=0.1)),
                'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl4]))}
    
    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl4, n_classes], stddev=0.1)),
                'biases':tf.Variable(tf.constant(0.1,shape=[n_classes]))}

    # input_data*weights + biases
    layer1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer1 = tf.nn.relu(layer1) # applies an activation function for rectified linear

    layer2 = tf.add(tf.matmul(layer1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer3 = tf.nn.relu(layer3)

    layer4 = tf.add(tf.matmul(layer3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    layer4 = tf.nn.relu(layer4)

    output = tf.matmul(layer4, output_layer['weights'])+ output_layer['biases']

    return output

def train_neural_net(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 13
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_net(x)

    
    
    
