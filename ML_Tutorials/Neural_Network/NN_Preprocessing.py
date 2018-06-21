# uses about 5000 positive and 5000 negative sentiments to decide sentiment of a sentence
# data must be made into a lexicon of words
# make a form of hot setting where each sentence will make vectors based on which words are present

import nltk
import io
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # useful for finding roots of words
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000
max_occurences = 1000
min_occurences = 50
# if you see MemoryError, you ran out of RAM
# can be fixed with less layers or less data

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with io.open(fi, 'r', encoding='cp437') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    lex2 = []
    for w in w_counts:
        if max_occurences > w_counts[w] > min_occurences:
            lex2.append(w)
    print(len(lex2))
    return lex2

def sample_handling(sample, lexicon, classification):
    featureset = []

    with io.open(sample, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_val = lexicon.index(word.lower())
                    features[index_val] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset

def create_feature_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0]) # positive is 1, 0
    features += sample_handling('neg.txt', lexicon, [0, 1]) # negative is 0, 1

    random.shuffle(features) # needs to happen to help the net as it tries weights
    # general question is, does tf.argmax([output]) == tf.argmax([expectation])?

    features = np.array(features)
    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size]) # this means all of the 0th elements
                                  #takes all of the values that are from the features in sample handling
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)




