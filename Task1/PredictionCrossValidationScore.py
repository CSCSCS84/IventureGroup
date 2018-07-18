# script for a first view how the classifiers perform on prepared dataset
from sklearn.model_selection import cross_val_score
from Task1 import InputReader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

filename = "test_dataset_sample10000Prepared"
train = InputReader.createInstance(filename)

clf = KNeighborsClassifier(n_neighbors=5)
features = ["publisherFake0.0", "publisherFake0.1", "publisherFake0.2", "publisherFake0.3", "publisherFake0.4",
           "publisherFake0.5", "publisherFake0.6", "publisherFake0.7", "publisherFake0.8", "publisherFake0.9",
           "partner", "domainNumber"]

X_train = train[features].as_matrix()
y_train = train["is_fake"].as_matrix()
cv_scores = cross_val_score(clf, X_train, y_train)

print("Cross-validation scores (3-fold):", cv_scores)
print("Mean cross-validation score (3-fold): {:.3f}"
      .format(np.mean(cv_scores)))
