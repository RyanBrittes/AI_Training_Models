import numpy as np

class LogLoss():
    def log_loss_value(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1-y_pred + 1e-9))
    
    def simple_loss(self, y_pred, y_true):
        return y_pred - y_true