import numpy as np
import pickle
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import keras
from keras import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.models import load_model

# purely CNN with a dense layer results in 74.5% before overfitting
# adding an LSTM layer to it results in 75.5% without tuning
# The speed of compiling the CNN was interesting but the accuracy is not impressive
# I will play around with adding some different layers to see if I can make
# a viable use of CNN 

lemmatizer = WordNetLemmatizer()
batch_size = 256
hm_epochs = 5
embedding_size = 400
maxseqlen = 40

with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)
        
maxvocab = len(lexicon)

with open('XY.pickle', 'rb') as f:
    features, Y = pickle.load(f)
    
split_index = int(0.90*len(features))
train_x, val_x = features[:split_index], features[split_index:]
train_y, val_y = Y[:split_index], Y[split_index:]

with open('testpick.pickle', 'rb') as f:
    test_x, test_y = pickle.load(f)


def batcher(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    while True:
        ran = np.random.randint(len(x)//batch_size)
        for i in range(batch_size*ran, len(x), batch_size):
            yield np.array(x[i:i+batch_size]), np.array(y[i:i+batch_size])


train_batcher = batcher(train_x, train_y, batch_size)
val_batcher = batcher(val_x, val_y, batch_size)
val_steps = len(val_x)//batch_size
test_batcher = batcher(test_x, test_y, batch_size)
test_steps = len(test_x)//batch_size

def buildCNN():
    cnn = Sequential()
    cnn.add(Embedding(maxvocab+1, embedding_size, input_length=maxseqlen))
    cnn.add(Dropout(0.2))
    cnn.add(Conv1D(64, 3, activation='relu'))
    cnn.add(MaxPooling1D(pool_size=3))
    cnn.add(LSTM(64, recurrent_dropout=0.4, dropout=0.4, return_sequences=True))
    cnn.add(Dense(32, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Flatten())
    cnn.add(Dense(1, activation='sigmoid'))
    print(cnn.summary())

    return cnn

# comment out either the load or the first build when using
cnn = buildCNN()
# cnn = load_model('cnnmodel.h5')

cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = cnn.fit_generator(train_batcher, steps_per_epoch=500, epochs=hm_epochs,
                              validation_data=val_batcher, validation_steps=val_steps,
                              verbose = 2)
cnn.save('cnnmodel.h5')

def testaccuracy():
    model = load_model('cnnmodel.h5')
    scores = model.evaluate_generator(test_batcher, steps = test_steps)
    print('test accuracy: ', scores[1])

testaccuracy()


def makeprediction(sentence):
    model = load_model('cnnmodel.h5')
    words = word_tokenize(sentence.lower())
    words = [i for i in words if i not in punctuation]
    words = [i for i in words if i != '...']
    int_words = []
    for word in words:
        word = lemmatizer.lemmatize(word)
        if word in lexicon:
            int_words.append(lexicon.index(word)+1)
        else:
            print(word, "not in my vocabulary")
    print(int_words)
    
    features = np.zeros((1,maxseqlen), dtype=int)
    features[0, -len(int_words):] = np.array(int_words)[:maxseqlen]
            
    pred = model.predict_classes(features)
    if pred == [[1]]:
        print("Positive! :)")
    else:
        print("Negative :(")
