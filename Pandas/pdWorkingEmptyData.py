import pandas as pd

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/LinearRegression/Pandas/Idades.csv')

dataset.fillna('VALUE NULL', inplace=True)

print(dataset)