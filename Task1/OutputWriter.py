from Task1 import definitions
import pandas as pd


def writeResultToFile(result,test, filename):
    ROOT_DIR = definitions.ROOT_DIR
    file = "%s/Output/%s.csv" % (
        ROOT_DIR, filename)
    result2 = pd.DataFrame(index=test.index)

    #result2['Survived'] = result
    result2["is_fake"] = result
    #print(result)

    result2.to_csv(file, header='username\tis_fake', sep=',')
