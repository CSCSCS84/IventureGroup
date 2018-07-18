import pandas as pd
from Task1 import definitions


def createInstance(filename):
    ROOT_DIR = definitions.ROOT_DIR
    file = "%s/Input/%s.csv" % (ROOT_DIR, filename)
    train = pd.read_csv(file, index_col="username")
    return train
