import numpy as np

class NormalizeScoreZ():

    def get_mean_value(self, rawValue):
        sumValue = 0
        for i in range(len(rawValue)):
            sumValue += rawValue[i]

        return sumValue/len(rawValue)

    def get_standard_deviation(self, rawValue):
        mean = self.get_mean_value(rawValue)
        sumValue = 0
        for i in range(len(rawValue)):
            sumValue += (rawValue[i] - mean) ** 2
        
        return (sumValue / len(rawValue)) ** 0.5

    def get_score_Z_normalize(self, rawValue):
        meanValue = self.get_mean_value(rawValue)
        stdValue = self.get_standard_deviation(rawValue)
        listValue = []

        for i in range(len(self, rawValue)):
            value = (rawValue[i] - meanValue) / stdValue
            listValue.append(value)
        
        return np.vstack(listValue)


