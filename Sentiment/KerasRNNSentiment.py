import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import keras
from keras import Sequential
from keras.preprocessing import sequence
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from keras.models import load_model

# working with embedding layers for increasing vocabulary size
lemmatizer = WordNetLemmatizer()
batch_size = 64
hm_epochs = 40
embedding_size = 400
maxseqlen = 40

with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)
        
maxvocab = len(lexicon)
        
def RNN():

    def makedata():
        with open('train_set_shuffled.csv', 'r', buffering = 100000, encoding="latin-1") as f:
            X = []
            Y = []
            xset = []
            for line in f:
                score = line.split(':::')[0]
                Y.append((score[4]))
                tweet = line.split(':::')[1]
                tweet = [lemmatizer.lemmatize(i) for i in word_tokenize(tweet.lower())]
                tweet = [c for c in tweet if c not in punctuation]
                tweet = [c for c in tweet if c != '...']
                X.append(tweet)

        for response in X:
            content = []
            for word in response:
                if word in lexicon:
                    index_val = lexicon.index(word)
                    content.append(index_val+1)
            xset.append(content)

        newx = []
        newy = []

        for i, row in enumerate(xset):
            if len(row) > 0:
                newx.append(row[0:maxseqlen])
                newy.append(Y[i])

        features = np.zeros((len(newx), maxseqlen), dtype=int)
        for i, row in enumerate(newx):
            features[i, -len(row):] = np.array(row)[:maxseqlen]

        with open('XY.pickle', 'wb') as f:
            pickle.dump((features, newy), f)

    # do one time!
    # makedata()

    with open('XY.pickle', 'rb') as f:
        features, Y = pickle.load(f)
        
    split_index = int(0.80*len(features))
    train_x, val_x = features[:split_index], features[split_index:]
    train_y, val_y = Y[:split_index], Y[split_index:]

    split_index = int(0.6*len(val_x))
    val_x, test_x = val_x[:split_index], val_x[split_index:]
    val_y, test_y = val_y[:split_index], val_y[split_index:]

    def batcher(x, y, batch_size):
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        while True:
            for i in range(0, len(x), batch_size):
                yield np.array(x[i:i+batch_size]), np.array(y[i:i+batch_size])


    train_batcher = batcher(train_x, train_y, batch_size)
    val_batcher = batcher(val_x, val_y, batch_size)
    val_steps = len(val_x)//batch_size
    test_batcher = batcher(test_x, test_y, batch_size)
    test_steps = len(test_x)//batch_size
    
    def makeRNN():
        def firstbuild():
            model = Sequential()
            model.add(Embedding(maxvocab+1, embedding_size, input_length=maxseqlen))
            model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
            model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
            model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
            model.add(Dense(1, activation='sigmoid'))

            print(model.summary())
            return model

        model = firstbuild()
        
        def training():
            # model = load_model('rnnmodel.h5')

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit_generator(train_batcher, steps_per_epoch=500, epochs=hm_epochs,
                                          validation_data=val_batcher, validation_steps=val_steps,
                                          verbose = 2)
            model.save('rnnmodel.h5')
        training()
        

    makeRNN()
    
    
    def testaccuracy():
        model = load_model('rnnmodel.h5')
        scores = model.evaluate_generator(test_batcher, steps=test_steps)
        print('test accuracy: ', scores[1])
    
    testaccuracy()

RNN()


def makeprediction(sentence):
    model = load_model('rnnmodel.h5')
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
            
    print(model.predict_classes(features))
    
