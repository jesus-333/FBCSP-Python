# FBCSP Python
Python implemementation of the FBCSP algorithm. Based on my previous work on the CSP algorithm available at this [link](https://github.com/jesus-333/CSP-Python).

This repository contain an extension of the CSP (Common Spatial Pattern) known as FBCSP (Filter-Bank CSP). This algorithm is used to extract features and classify EEG trials. The algorithm is used for binary classification.

Version 4 (V4) is based on the work of [Kai Keng Ang et al.](https://ieeexplore.ieee.org/document/4634130) easily available following the link. Also the multiclass variant is based on a strategy described in that article. Version 1 (V1) and version 2 (V2) are the first implementation and contain error. They are not supposed to be used and lack the method for complete training and evaluation. Version 3 (V3) work's without big problem but I improved some function in the 4.

## How the class work
The class must receive in input with the initialization a training set inside a dictionary. The keys of the dictionary must be the label of the two class and each element must be a numpy matrix of dimension "n. trials x n. channels x n.samples". The class must also receive the frequency sampling of the signal.

Once initialized the class automaticaly found the various spatial filter and the best features to train the classifier. The classifier used is an [LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) classifier implemented through sklearn. You could also provided another sklean classifier during the initialization of the class if you prefer. The class split the data in a train and test set and then trains and evaluates the classifiers.

An evalaution method to classify new trials is implemented with the name *evaluateTrial()*. The input of method must have the dimensions "n. trials x n. channels x n.samples". The method return a vector where each element is the label of the respective trial. The label are *"1"* for class 1 and *"2"* for class 2.

## Multiclass Classification
In case you have data with multiple label I also create a second class for multiclassification. The class is inside the file "FBCSP_Multiclass.py". The input of the class must be also in this case a dictionary with key the label of the varios trial and element the trials matrix with dimensions "n. trials x n. channels x n.samples". Results for the [dataset 2a](http://www.bbci.de/competition/iv/desc_2a.pdf) of the BCI competition dataset is reported in the table below:

| Subject |  [Kai Keng Ang et al.](https://ieeexplore.ieee.org/document/4634130) | My Results |
|:-------:|:-----:|:----------:|
|    1    | 0.676 |    *0.698*   |
|    2    | 0.417 |    **0.555**   |
|    3    | 0.745 |    **0.805**   |
|    4    | 0.481 |   **0.5381**   |
|    5    | 0.398 |    **0.444**   |
|    6    | 0.273 |   **0.4027**   |
|    7    | 0.773 |   *0.7743*   |
|    8    | 0.755 |    **0.684**   |
|    9    | 0.606 |    **0.725**   |
|   AVG   | 0.569 |   **0.6251**   |
