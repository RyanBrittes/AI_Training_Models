import numpy as np

class NormalizeLog():

    def get_log_normalize(self, rawValue):
        listValue = []
        const_not_zero = 1e-8

        for i in range(len(rawValue)):
            listValue.append(np.log(rawValue[i] + const_not_zero))
        
        return listValue

