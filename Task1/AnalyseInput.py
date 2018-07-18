#Analyse data that is already prepared
from Task1 import InputReader
import seaborn as sns
import matplotlib.pyplot as plt


def plotFraudProbality(train):
    features = ["partner", "domainName_com", "domainName_de", "domainName_eu", "domainName_ch", "domainNumber"]
    for f in features:
        plot = sns.factorplot(x=f, y='is_fake', data=train, kind='bar')
        plot = plot.set_ylabels("fraud probality")
        plt.show()


def analysePublisher(train):
    train = train[["publisher", "is_fake"]]
    print(train.groupby(by="publisher").sum() / train.groupby(by="publisher").count())


def calcCorrelation(train):
    correlation = train.corr().round(2)
    ax = sns.heatmap(correlation, annot=True)
    plt.show()


filenameTrain = "test_datasetPrepared"
train = InputReader.createInstance(filenameTrain)

print(train.isnull().sum())
print(train.info())
calcCorrelation(train)
analysePublisher(train)
plotFraudProbality(train)
