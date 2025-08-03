from sigmoid import Sigmoid
from logLoss import LogLoss
from loadData import LoadData
import numpy as np

class LogisticRegression():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.loss = LogLoss()
        self.data = LoadData()
        self.x = self.data.get_x_value()
        self.y = self.data.get_y_value()
        self.weights = np.zeros(self.x.shape[1])
        self.bias = 0
        self.lr = 0.00001
        self.epochs = 1000
        self.losses = []
        self.n_sample = len(self.x)
        self.range_prob = 0.7
    
    def training_model(self):

        for i in range(self.epochs):
            z_value = np.array(self.x @ self.weights + self.bias).reshape(-1, 1)
            y_pred = self.sigmoid.sigmoid_calc(z_value)
            simple_loss = self.loss.simple_loss(y_pred, self.y)

            dw = np.array((self.x.T @ simple_loss) / self.n_sample).flatten()
            db = np.sum(simple_loss) / self.n_sample

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self.loss.log_loss_value(y_pred, self.y)
            self.losses.append(loss)

            if(i % 100) == 0:
                print(f"Epoch: {i}\nLoss: {loss:.4f}")
    
        return [self.weights, self.bias, self.losses]
            
    def show_results(self):
        training_results = self.training_model()

        z_value = self.x @ training_results[0] + training_results[1]
        prediction = self.sigmoid.sigmoid_calc(z_value)

        calc = (prediction >= self.range_prob).astype(int)

        accuracy = np.mean(calc == self.y)

        print(f"Final Acurracy: {accuracy:.2f}")

    def values(self):
        print((self.n_sample))

A = LogisticRegression()

A.show_results()


