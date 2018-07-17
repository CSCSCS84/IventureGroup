from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = "%s/Input/test_dataset1.csv" % (
    ROOT_DIR)

train = pd.read_csv(filename,
                    index_col='username')
train = train[1:10000]

columns = ["publisherFake0.0", "publisherFake0.1", "publisherFake0.2", "publisherFake0.3", "publisherFake0.4",
           "publisherFake0.5", "publisherFake0.6", "publisherFake0.7", "publisherFake0.8", "publisherFake0.9",
           "partner", "is_fake", "domainNumber"]

y_train = train['is_fake']
X_train = train[columns]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(X_train, y_train)))
