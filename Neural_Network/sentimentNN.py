import tensorflow as tf
import pickle
import numpy as np
import io
from NN_Preprocessing import create_feature_labels

train_x, train_y, test_x, test_y = create_feature_labels('pos.txt', 'neg.txt')


n_nodes_hl1 = 400 #these can be different and whatever you like
n_nodes_hl2 = 400
n_nodes_hl3 = 300
n_nodes_hl4 = 300

n_classes = 2
batch_size = 100

x = tf.placeholder('float',[None,len(train_x[0])]) # will throw an error if the shape is not correct
y = tf.placeholder('float')

def neural_network(data):
    # tensor of weights with random values
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1)),
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 13
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+= batch_size
                
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_net(x)
