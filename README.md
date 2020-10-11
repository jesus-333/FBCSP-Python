# FBCSP Python
Python implemementation of the FBCSP algorithm. Based on my previous work on the CSP algorithm available at this [link](https://github.com/jesus-333/CSP-Python).

This repository contain an extension of the CSP (Common Spatial Filter) known as FBCSP (Filter-Bank CSP). This algorithm is used to extract features and classify EEG trials. The algorithm is used for binary classification.

## How the class work
The class must receive in input with the initialization a training set inside a dictionary. The keys of the dictionary must be the label of the two class and each element must be a numpy matrix of dimension "n. trials x n. channels x n.samples". The class must also receive the frequency sampling of the signal.

Once initialized the class automaticaly found the various spatial filter and the best features to train the classifier. The classifier used is an [LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) classifier implemented through sklearn. You could also provided another sklean classifier during the initialization of the class if you prefer. The class split the data in a train and test set and then trains and evaluates the classifiers.

An evalaution method to classify new trials is implemented with the name *evaluateTrial()*. The input of method must have the dimensions "n. trials x n. channels x n.samples". The method return a vector where each element is the label of the respective trial. The label are *1* for class 1 and *-1* for class 2.
