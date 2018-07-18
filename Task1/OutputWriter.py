from Task1 import definitions
import pandas as pd


def writeResultToFile(y_test, X_test, filename):
    ROOT_DIR = definitions.ROOT_DIR
    file = "%s/Output/%s.csv" % (ROOT_DIR, filename)
    prediction = pd.DataFrame(index=X_test.index)
    prediction["is_fake"] = y_test
    prediction.to_csv(file, header='username\t is_fake', sep=',')
