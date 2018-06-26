import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, y, test_X, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_X = test_X.reshape([-1, 28, 28, 1])

convnet = input_data(shape=[None,28,28,1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convent = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax') # output layer
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

# this can be commented out after it has been done once before
model.fit({'input':X},{'targets':y}, n_epoch=3,
          validation_set=({'input':test_X},{'targets':test_y}),
          snapshot_step=5000, show_metric=True, run_id='mnist')

model.save('tflearncnn.model') # only saves weights so you are not pickling the model

model.load('tflearncnn.model')
print(model.predict([test_X[7]]))
