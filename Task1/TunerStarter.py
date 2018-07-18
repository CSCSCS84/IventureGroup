from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import os
from Task1 import ClassifierTuner


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = "%s/Input/test_dataset_sample10000Prepared.csv" % (
    ROOT_DIR)

features = ["publisherFake0.0", "publisherFake0.1", "publisherFake0.2", "publisherFake0.3", "publisherFake0.4",
           "publisherFake0.5", "publisherFake0.6", "publisherFake0.7", "publisherFake0.8", "publisherFake0.9",
           "partner", "domainNumber"]

train = pd.read_csv(filename,
                        index_col='username')
targetValue=train["is_fake"]
searchGrid = ClassifierTuner.getLogisticRegressionGrid()


classifier = LogisticRegression()

tunedClassifier = ClassifierTuner.tuneClassifier(train, classifier, searchGrid, features,targetValue)
print(tunedClassifier)
