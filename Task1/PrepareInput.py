import numpy as np
import pandas as pd


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


def replaceSubstrings(train):
    train['partner'] = train['partner'].apply(replacePar)
    train['publisher'] = train['publisher'].apply(replacePub)
    return train


def convertIndicatorValuesDomainName(dataset):
    dataset = pd.get_dummies(dataset, columns=['domainName'], prefix=['domainName'])
    return dataset


def calculatePublisherFakeRate(train):
    rate = train.groupby(by="publisher").sum() / train.groupby(by="publisher").count()
    rate = rate["is_fake"]
    publisherFakeRate = np.zeros(train.shape[0])
    i = 0
    for index, row in train.iterrows():
        publisher = row['publisher']
        publisherFakeRate[i] = rate.loc[publisher]
        i = i + 1
    train['publisherFakeRate'] = publisherFakeRate
    train['publisherFake0.9'] = train['publisherFakeRate'].map(lambda s: 1 if s >= 0.9 else 0)
    train['publisherFake0.8'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.8 and s < 0.9) else 0)
    train['publisherFake0.7'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.7 and s < 0.8) else 0)
    train['publisherFake0.6'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.6 and s < 0.7) else 0)
    train['publisherFake0.5'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.5 and s < 0.6) else 0)
    train['publisherFake0.4'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.4 and s < 0.5) else 0)
    train['publisherFake0.3'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.3 and s < 0.4) else 0)
    train['publisherFake0.2'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.2 and s < 0.3) else 0)
    train['publisherFake0.1'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.1 and s < 0.2) else 0)
    train['publisherFake0.0'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.0 and s < 0.1) else 0)
    return train
