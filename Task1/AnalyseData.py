from Task1 import InputReader
#checks if all partner and publisher data start with "par_" or "pub_"

def checkPartner(partner):
    if partner[0:4] != "par_":
        print("Not all Strings in column Partner start with 'par_'")


def checkPublisher(publisher):
    if publisher[0:4] != "pub_":
        print("Not all Strings in column 'Publisher' start with 'pub_'")

def checkRows():
    train=InputReader.createInstance("test_dataset")
    train['partner'].apply(checkPartner)
    train['publisher'].apply(checkPublisher)

checkRows()