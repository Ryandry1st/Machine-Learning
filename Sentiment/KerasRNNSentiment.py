import numpy as np
import pickle
import tensorflow as tf
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
# 80.8% accuracy with around 50 epochs of 64 batch size and 3 lstm layers
# adding a dense layer before the output layer increased accuracy above 81.1%

lemmatizer = WordNetLemmatizer()
batch_size = 128
hm_epochs = 5
embedding_size = 400
maxseqlen = 40

with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)
        
maxvocab = len(lexicon)
        
def RNN():

    with open('XY.pickle', 'rb') as f:
        features, Y = pickle.load(f)
        
    split_index = int(0.9*len(features))
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
    
    def makeRNN():
        def firstbuild():
            model = Sequential()
            model.add(Embedding(maxvocab+1, embedding_size, input_length=maxseqlen))
            model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
            model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
            model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
            model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.4))
            model.add(Dense(1, activation='sigmoid'))

            print(model.summary())
            return model

        # either comment out the first build or the loading for training when running
        model = firstbuild()
        
        def training():
            # model = load_model('rnnmodel.h5')

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit_generator(train_batcher, steps_per_epoch=300, epochs=hm_epochs,
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
            
    pred = model.predict_classes(features)
    if pred == [[1]]:
        print("Positive! :)")
    else:
        print("Negative :(")
        
    
