from Task1 import PrepareInput
from Task1 import InputReader
from Task1 import TunedClassifiers
from Task1 import OutputWriter


def createInstances(filenameTrain, filenameTest):
    train = InputReader.createInstance(filenameTrain)
    test = InputReader.createInstance(filenameTest)

    features = ["publisherFakeRate", "publisherFake0.0", 'publisherFake0.1', 'publisherFake0.2', 'publisherFake0.3',
                'publisherFake0.4','publisherFake0.5', 'publisherFake0.6', 'publisherFake0.7', "publisherFake0.8",
                'publisherFake0.9', "partner", "domainNumber"]

    X_train = PrepareInput.prepare(train)
    y_train = train["is_fake"]
    X_train = X_train[features]

    test = PrepareInput.prepare(test)
    X_test = test[features]
    return [X_train, y_train, X_test]


filenameTrain = "test_dataset_sample10000"
filenameTest = "test_dataset_sample10000"
filenamePrediction = "Prediction"

X_train, y_train, X_test = createInstances(filenameTrain, filenameTest)
classifier = TunedClassifiers.getTunedDecisionTree()
classifier.fit(X_train, y_train)
result = classifier.predict(X_test)

OutputWriter.writeResultToFile(result, X_test, filenamePrediction)
