import pandas as pd
import numpy as np

class LoadData():
    def __init__(self):
        self.__data = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/files/diabetes.csv')
        self.__y_true = self.__data[['Outcome']].values
        self.__x_true = self.__data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values

    def get_x_value(self):
        return self.__x_true
    
    def get_y_value(self):
        return self.__y_true
    

