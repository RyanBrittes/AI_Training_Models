import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/files/diabetes.csv')

valueX = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values
valueY = dataset[['Age']].values

for i in range(len(valueX[0])):
    plt.plot(valueX[:,i], valueY, '.')
    plt.title(f'{dataset.columns[i]} vs Age')
    plt.xlabel(f'{dataset.columns[i]}')
    plt.ylabel('Age')
    plt.show()

for i in range(len(valueX[0])):
    plt.hist(valueX[:,i], bins=10, color='blue', edgecolor='black')
    plt.title(dataset.columns[i])
    plt.xlabel('Value')
    plt.ylabel('Quantity')
    plt.show()