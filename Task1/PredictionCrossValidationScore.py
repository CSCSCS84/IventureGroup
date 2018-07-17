from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = "%s/Input/test_dataset_sample10000.csv" % (
    ROOT_DIR)

train = pd.read_csv(filename,
                    index_col='username')
train=train[1:10000]

clf = KNeighborsClassifier(n_neighbors = 5)
columns = ["publisherFake0.0", "publisherFake0.1", "publisherFake0.2", "publisherFake0.3", "publisherFake0.4",
           "publisherFake0.5", "publisherFake0.6", "publisherFake0.7", "publisherFake0.8", "publisherFake0.9",
           "partner", "domainNumber"]

X = train[columns].as_matrix()
y = train["is_fake"].as_matrix()
cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))