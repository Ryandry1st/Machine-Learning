import numpy as np
import pickle
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation


lemmatizer = WordNetLemmatizer()
batch_size = 128
hm_epochs = 3
embedding_size = 400
maxseqlen = 40

# converts the data into a more manageable form. Used in different methods
# so the creation and deconstruction later of lists for labels is necessary

def init_process(fin, fout):
    outfile = open(fout, 'a')
    error_counter = 0
    with open(fin, buffering=200000, encoding='latin-1') as f:
        for line in f:
            try:
                line = line.replace('"', '')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                    tweet = line.split(',')[-1]
                    outline = str(initial_polarity)+':::'+tweet
                    outfile.write(outline)
                    
                elif initial_polarity == '4':
                    initial_polarity = [0,1]
                    tweet = line.split(',')[-1]
                    outline = str(initial_polarity)+':::'+tweet
                    outfile.write(outline)
                    
                
            except Exception as e:
                error_counter += 1
    print("Total errors:", error_counter)
    outfile.close()


# Only run to create the vocabulary list for the word embeddings
def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter / 250.00).is_integer():
                    tweet = line.split(':::')[1]
                    content += tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
        
        except Exception as e:
            print(str(e))

    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)


# Only run once to create a shuffled training set
def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False, encoding='latin-1')
    df = df.iloc[np.random.permutation(len(df))]
    df.to_csv('train_set_shuffled.csv', index=False)


# convert strings to integers for word embeddings
with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)
        
maxvocab = len(lexicon)
        
def makedata(fin, fout):
    with open(fin, 'r', buffering = 100000, encoding="latin-1") as f:
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

    with open(fout, 'wb') as f:
        pickle.dump((features, newy), f)

def run_once():
    # init_process('training.1600000.processed.noemoticon.csv', 'train_set.csv')
    init_process('testdata.manual.2009.06.14.csv', 'test_set.csv')
    # create_lexicon('train_set.csv')
    # convert_to_vec('test_set.csv', 'processed-test-set.csv', 'lexicon.pickle')
    # shuffle_data('train_set.csv')
    # makedata('train_set_shuffled.csv', 'XY.pickle')
    makedata('test_set.csv', 'testpick.pickle')

run_once()
