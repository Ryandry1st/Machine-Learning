# Machine-Learning
First, tutorials on Machine Learning and the basis for how to use data science tools with Python are attempted in the Machine learning folder.

The primary source for tutorials is by going through the videos hosted by Sentdex at https://pythonprogramming.net/machine-learning-tutorial-python-introduction/

Tools like pickle are used to reduce the need for internet connections (due to very restrictive local internet providers preventing many common apps like whatsapp, Facebook, and, in this case, Quandl for data, as well as to reduce the need to recalculate things like neural networks.

There is generally an example of a machine learning method, like SVM or K-Nearest_Neighbors, along with another set of code which derives how those functions work within the machine learning folder.


The seperate folder for neural network code is also done through tutorials by Sentdex, and will be added with tutorials through tensorflow and keras as well.

Doing the larger neural network code, especially training a large NN or RNN, takes hours using the cpu version with the current code. I will be putting checkpoint saving in so that the network will not need to be retrained unless desired.


The third folder specifically focuses on my own project, which is creating or improving a neural network for sentiment analysis. I initially started with the NN from the tutorial, and have changed some preprocessing to allow for word embedding, as well as a RNN and CNN neural network architectures. The goal is for greater than 85% accuracy, with the RNN at about 81.1% currently.
