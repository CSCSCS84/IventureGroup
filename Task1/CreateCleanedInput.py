import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#from Task1 import definitions
import os

def checkPar(par):
    if par[0:4]!="par_":
        print("Not all String begin with 'par_'")

def checkPublisher(publisher):
    if publisher[0:4]!="pub_":
        print("Not all String begin with 'pub_'")

def checkPublisher(publisher):
    if publisher[0:4]!="pub_":
        print("Not all String begin with 'pub_'")

def replacePar(par):
    return par.replace("par_","")

def replacePub(pub):
    return pub.replace("pub_", "")

def replaceDomainName(domain):
    return domain.split('_')[-1]


def replaceDomainNumber(domain):

    return domain.split('_')[-2]



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = "%s/Input/test_dataset.csv" % (
    ROOT_DIR)


train = pd.read_csv(filename,
                        index_col='username')


#print(train[train["is_fake"]==0 ])
#print(train['partner'].unique())

#print(train['partner'].apply(checkPar))
#print(train['publisher'].apply(checkPublisher))

def cleanData(train):
    train['partner']=train['partner'].apply(replacePar)
    train['publisher']=train['publisher'].apply(replacePub)
    return train


def convertIndicatorValuesGroupId(dataset):
    dataset = pd.get_dummies(dataset, columns=['domainName'], prefix=['domainName'])

    return dataset

train['domainName']=train['domain'].apply(replaceDomainName)
train['domainNumber']=train['domain'].apply(replaceDomainNumber)
print(train['domainName'].unique())

train=cleanData(train)
train=convertIndicatorValuesGroupId(train)

#print(train)
print(train.sort_values(by='domainNumber'))
columns=["publisher","domain","partner","is_fake","domainName_com","domainName_de","domainName_eu","domainName_ch","domainNumber"]
print(train)
train=train[columns]
train.to_csv("%s/Input/test_dataset1.csv" % (
    ROOT_DIR), header=columns, sep=',')



#print(train.describe())