import pandas as pd
import math, quandl, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

def retrainclassifier():
    df = quandl.get('WIKI/GOOGL')
    #gets data

    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100
    df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100

    df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

    forecast_col = 'Adj. Close'
    df.fillna('-99999', inplace=True)

    forecast_out = int(math.ceil(0.01*len(df))) #predicts 1% days out, this is 30 days
    print forecast_out


    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'],1))


    #converts data to be -1<=X<=1
    #if new data is brought in it must be scaled the same way
    #you would skip for any high frequency inputs like trading
    X = preprocessing.scale(X)
    X = X[:-forecast_out]
    X_lately = X[-forecast_out:]

    df.dropna(inplace=True)
    y = np.array(df['label'])

    #save data for other workings
    with open('dataframe.pickle', 'wb') as f:
        pickle.dump(df, f)


    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

    #creates classifiers with 80% of the data
    #MAKE sure to check the documentation for which can be threaded
    #This is generally n_jobs
    clf = LinearRegression(n_jobs=-1) #n_jobs makes training faster
    #clf2 = svm.SVR() #Produces much worse results

    clf.fit(X_train, y_train)

    #save the classifier, good if you have lots of data
    with open('linearreg.pickle', 'wb') as f:
        pickle.dump(clf, f)

    with open('xtest.pickle', 'wb') as g:
        pickle.dump(X_test, g)

    with open('ytest.pickle', 'wb') as h:
        pickle.dump(y_test, h)

    with open('xlate.pickle', 'wb') as j:
        pickle.dump(X_lately, j)


#comment this top line out to just load the old classifier
#retrainclassifier()
clfin = open('linearreg.pickle', 'rb')
clf = pickle.load(clfin)

dfin = open('dataframe.pickle', 'rb')
df = pickle.load(dfin)

xtestin = open('xtest.pickle', 'rb')
X_test = pickle.load(xtestin)

ytestin = open('ytest.pickle', 'rb')
y_test = pickle.load(ytestin)

xlately = open('xlate.pickle', 'rb')
X_lately = pickle.load(xlately)

#tests the classifier against 20% of the data
#this is the squared error
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print forecast_set, accuracy

df['forecast'] = np.nan

#Hard coded dates to get time
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    #iterates through the forecast set to shift the dates forward
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    #fill data with NaN so that the rest of the dataframe works
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


