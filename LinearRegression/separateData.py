from sklearn.model_selection import train_test_split
from extractData import ExtractData

class SeparateData():
    def __init__(self):
        self.extratedData = ExtractData()

    def separateTrainTest(self):
        trainX, testX, trainY, testY = train_test_split(self.extratedData.extractValueX(), self.extratedData.extractValueY(), test_size=0.3, random_state=42)
        trainTest = [trainX, testX, trainY, testY]
        return trainTest
    