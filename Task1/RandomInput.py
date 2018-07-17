#return a file with a random subset of the input file
import pandas as pd
import os

def randomSample(df, n):
    return df.sample(n)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filename = "%s/Input/test_dataset.csv" % (
    ROOT_DIR)


train = pd.read_csv(filename,
                    index_col='username')
sampleSize=500000
sample=randomSample(train,sampleSize)
sample.to_csv("%s/Input/test_dataset_sample%s.csv" % (
    ROOT_DIR,sampleSize), header=sample.columns, sep=',')




