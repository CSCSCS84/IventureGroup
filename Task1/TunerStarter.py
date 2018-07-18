#Tunes the classifiers. Tuner runs on a sample of the training set due to running times.
# The training set has to be prepared and is not the original dataset. Sample training sets can be created using RandomSampleInput
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from Task1 import ClassifierTuner
from Task1 import InputReader

filename = "test_dataset_sample10000Prepared"
train = InputReader.createInstance(filename)

features = ["publisherFake0.0", "publisherFake0.1", "publisherFake0.2", "publisherFake0.3", "publisherFake0.4",
            "publisherFake0.5", "publisherFake0.6", "publisherFake0.7", "publisherFake0.8", "publisherFake0.9",
            "partner", "domainNumber"]

y_train = train["is_fake"]
searchGrid = ClassifierTuner.getLogisticRegressionGrid()
classifier = LogisticRegression()

tunedClassifier = ClassifierTuner.tuneClassifier(train, classifier, searchGrid, features, y_train)
print(tunedClassifier)
