import numpy as np
from loadData import LoadData
from sklearn.model_selection import train_test_split

class ExtractData():
    def __init__(self):
        self.data = LoadData()
        self.dataset = self.data.getData()
    
    def extractValueX(self):
        return self.dataset[['age', 'education', 'experience', 'seniority']].values.astype(np.float32)
    
    def extractValueY(self):
        return self.dataset[['salary']].values.astype(np.float32)
    
        