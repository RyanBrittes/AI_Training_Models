import pandas as pd

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/LinearRegression/Pandas/Idades.csv')

datasetClean = dataset.dropna()

print(f"Dados sem alteração:\n{dataset}")
print(f"Dados otimizados:\n{datasetClean}")