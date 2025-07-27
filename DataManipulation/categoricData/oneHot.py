import numpy as np

class OneHot():
    def __init__(self):
        self.list = []
        self.index = []
        self.dict = {}

    def get_clean_data(self, dataset):
        for i in range(len(dataset)):
            if dataset[i] not in self.list:
                self.list.append(dataset[i])
        return self.list

    def get_index_from_data(self, list):
        for i in range(len(list)):
            self.index.append(i)
            self.dict[list[i]] = i
        index_dict = [self.index, self.dict]
        return index_dict

    def get_identity_matriz(self, list):
        return np.eye(len(list))
    
