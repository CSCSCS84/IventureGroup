import numpy as np
import pandas as pd
# from Task1 import definitions
import os
from time import time

def replacePar(par):
    return par.replace("par_", "")

def replacePub(pub):
    return pub.replace("pub_", "")

def replaceDomainName(domain):
    return domain.split('_')[-1]

def replaceDomainNumber(domain):
    return domain.split('_')[-2]

def prepare(dataset):
    dataset['domainName'] = dataset['domain'].apply(replaceDomainName)
    dataset['domainNumber'] = dataset['domain'].apply(replaceDomainNumber)

    dataset = cleanData(dataset)
    dataset = convertIndicatorValuesGroupId(dataset)

    dataset = convertPublisher(dataset)
    return dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = "%s/Input/test_dataset_sample10000.csv" % (
    ROOT_DIR)

train = pd.read_csv(filename,
                    index_col='username')


# print(train[train["is_fake"]==0 ])
# print(train['partner'].unique())

# print(train['partner'].apply(checkPar))
# print(train['publisher'].apply(checkPublisher))

def cleanData(train):
    train['partner'] = train['partner'].apply(replacePar)
    train['publisher'] = train['publisher'].apply(replacePub)
    return train


def convertIndicatorValuesGroupId(dataset):
    dataset = pd.get_dummies(dataset, columns=['domainName'], prefix=['domainName'])
    return dataset


def convertPublisher(train):
    rate = train.groupby(by="publisher").sum() / train.groupby(by="publisher").count()

    rate = rate["is_fake"]
    pubRate = np.zeros(train.shape[0])
    i = 0
    t1 = time()
    for index, row in train.iterrows():
        publisher = row['publisher']
        pubRate[i] = rate.loc[publisher]
        i = i + 1
    t2 = time()
    print("Zeit in ms %s" % (t2 - t1))
    train['publisherFakeRate'] = pubRate
    train['publisherFake0.9'] = train['publisherFakeRate'].map(lambda s: 1 if s >= 0.9 else 0)
    train['publisherFake0.8'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.8 and s < 0.9) else 0)
    train['publisherFake0.7'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.7 and s < 0.8)  else 0)
    train['publisherFake0.6'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.6 and s < 0.7)  else 0)
    train['publisherFake0.5'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.5 and s < 0.6) else 0)
    train['publisherFake0.4'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.4 and s < 0.5)  else 0)
    train['publisherFake0.3'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.3 and s < 0.4)  else 0)
    train['publisherFake0.2'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.2 and s < 0.3)  else 0)
    train['publisherFake0.1'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.1 and s < 0.2)  else 0)
    train['publisherFake0.0'] = train['publisherFakeRate'].map(lambda s: 1 if (s >= 0.0 and s < 0.1) else 0)
    return train


columns = ["publisherFakeRate","publisherFake0.0", 'publisherFake0.1', 'publisherFake0.2', 'publisherFake0.3', 'publisherFake0.4',
           'publisherFake0.5', 'publisherFake0.6', 'publisherFake0.7', "publisherFake0.8", 'publisherFake0.9',
           "publisher", "domain", "partner", "is_fake",  "domainNumber"]

#columns=list(set(columns).intersection(train.columns))
print(train)
train = train[columns]
train.to_csv("%s/Input/test_dataset_sample10000%s.csv" % (
    ROOT_DIR,"Prepared"), header=columns, sep=',')



# print(train.describe())
