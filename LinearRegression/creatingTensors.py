import torch
from separateData import SeparateData

class CreatingTensors():
    def __init__(self):
        self.separatedData = SeparateData()

    def tensorTrainX(self):
        return torch.tensor(self.separatedData.separateTrainTest()[0])
    
    def tensorTestX(self):
        return torch.tensor(self.separatedData.separateTrainTest()[1])
    
    def tensorTrainY(self):
        return torch.tensor(self.separatedData.separateTrainTest()[2])
    
    def tensorTestY(self):
        return torch.tensor(self.separatedData.separateTrainTest()[3])
    