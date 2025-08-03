import numpy as np

class Sigmoid():
    def sigmoid_calc(self, z_value):
        return 1/(1 + np.exp(-z_value))