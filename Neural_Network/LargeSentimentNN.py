import tensorflow as tf
import nltk
import io
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # useful for finding roots of words
import numpy as np
import pickle



n_nodes_hl1 = 400 #these can be different and whatever you like
n_nodes_hl2 = 400

lemmatizer = WordNetLemmatizer()
n_classes = 2
batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 10

x = tf.placeholder('float') # will throw an error if the shape is not correct
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1, 'weights':tf.Variable(tf.truncated_normal([2569, n_nodes_hl1], stddev=0.1)),
                  'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2, 'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
            'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl2]))}

output_layer = {'f_fum': None, 'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_classes], stddev=0.1)),
            'biases':tf.Variable(tf.constant(0.1,shape=[n_classes]))}

def neural_network_model(data):
    # input_data*weights + biases
    layer1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer1 = tf.nn.relu(layer1) # applies an activation function for rectified linear

    layer2 = tf.add(tf.matmul(layer1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer2 = tf.nn.relu(layer2)

    output = tf.matmul(layer2, output_layer['weights'])+ output_layer['biases']

    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_net(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
            print('Starting:', epoch)
        except:
            epoch = 1
            
        while epoch <= hm_epochs:

            if epoch != 1:
                saver.restore(sess, "./model.ckpt")
            epoch_loss = 1

            with open('lexicon-2500-2569.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering = 200000, encoding='latin-1') as f:
                counter = 0
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    counter += 1
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            features[index_value] += 1

                    batch_x.append(list(features))
                    batch_y.append(eval(label))
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        if (batches_run / 1000).is_integer():
                            print('Batch run:', batches_run, '/', total_batches,'| Epoch:', epoch, '| Batch Loss:', c,)

            saver.save(sess, "./model.ckpt")              
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            with open(tf_log, 'a') as f:
                f.write(str(epoch)+'\n')
            epoch +=1

# Do once!
# train_neural_net(x)

def test_NN():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess, './model.ckpt')
            except Exception as e:
                print(str(e))
            epoch_loss = 0

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        featuresets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=200000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    featuresets.append(features)
                    labels.append(label)
                    counter +=1
                except:
                    pass
        print("Tested", counter, "samples.")
        test_x = np.array(featuresets)
        test_y= np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

test_NN()

