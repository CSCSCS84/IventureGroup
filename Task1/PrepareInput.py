# Prepare the dataset for prediction. In example, for publisher "pub_12345" only 12345 is used in new dataset.
# Enter a filename on the end of this script and the new dataset is written to a csv with the ending "Prepared"
import numpy as np
import pandas as pd
from Task1 import InputReader
from Task1 import definitions


def prepare(dataset):
    dataset['domainName'] = dataset['domain'].apply(replaceDomainName)
    dataset['domainNumber'] = dataset['domain'].apply(replaceDomainNumber)
    dataset = replaceSubstrings(dataset)
    dataset = convertIndicatorValuesDomainName(dataset)
    dataset = calculatePublisherFakeRate(dataset)
    return dataset


def replacePar(par):
    return par.replace("par_", "")


def replacePub(pub):
    return pub.replace("pub_", "")


def replaceDomainName(domain):
    return domain.split('_')[-1]


def replaceDomainNumber(domain):
    return domain.split('_')[-2]


def replaceSubstrings(dataset):
    dataset['partner'] = dataset['partner'].apply(replacePar)
    dataset['publisher'] = dataset['publisher'].apply(replacePub)
    return dataset


def convertIndicatorValuesDomainName(dataset):
    dataset = pd.get_dummies(dataset, columns=['domainName'], prefix=['domainName'])
    return dataset


def calculatePublisherFakeRate(dataset):
    rate = dataset.groupby(by="publisher").sum() / dataset.groupby(by="publisher").count()
    rate = rate["is_fake"]
    publisherFakeRate = np.zeros(dataset.shape[0])

    i = 0
    for index, row in dataset.iterrows():
        publisher = row['publisher']
        publisherFakeRate[i] = rate.loc[publisher]
        i = i + 1

    dataset['publisherFakeRate'] = publisherFakeRate
    dataset['publisherFake0.9'] = dataset['publisherFakeRate'].map(lambda s: 1 if s >= 0.9 else 0)
    dataset['publisherFake0.8'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.8 and s < 0.9) else 0)
    dataset['publisherFake0.7'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.7 and s < 0.8) else 0)
    dataset['publisherFake0.6'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.6 and s < 0.7) else 0)
    dataset['publisherFake0.5'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.5 and s < 0.6) else 0)
    dataset['publisherFake0.4'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.4 and s < 0.5) else 0)
    dataset['publisherFake0.3'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.3 and s < 0.4) else 0)
    dataset['publisherFake0.2'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.2 and s < 0.3) else 0)
    dataset['publisherFake0.1'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.1 and s < 0.2) else 0)
    dataset['publisherFake0.0'] = dataset['publisherFakeRate'].map(lambda s: 1 if (s >= 0.0 and s < 0.1) else 0)
    return dataset


def prepareAndSaveAsCSV(filename):
    train = InputReader.createInstance(filename)
    train = prepare(train)
    train.to_csv("%s/Input/%sPrepared" % (
        definitions.ROOT_DIR, filename), header=train.columns, sep=',')


#filename = "test_dataset_sample10000"
#prepareAndSaveAsCSV(filename)
