import pandas as pd

dataset01=[1, 2, 3, 4, 5]
dataset02={
    'name1':'ryan',
    'name2':'joao',
    'name3':'carlos'
}
index = ['a', 'b', 'c', 'd', 'e']

#Dados unidimensionais
dataSeries01 = pd.Series(dataset01, index, "float32")
dataSeries02 = pd.Series(dataset02)

print(f"Conjunto 01: \n{dataSeries01}")
print(f"Conjunto 02: \n{dataSeries02}")