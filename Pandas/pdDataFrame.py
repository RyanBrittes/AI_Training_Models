import pandas as pd

dataset01 = [
    ['ryan', 'maria', 'ricardo'],[23, 27, 30]
]
dataset02 = {
    'Name':['ryan', 'joao', 'ana'],
    'Height': [192, 180, 185]
}
index = ['Name', 'Age']

dataFrame01 = pd.DataFrame(dataset01, index)
dataFrame02 = pd.DataFrame(dataset02)

print(f"Conjunto 02:\n{dataFrame02.loc[[0,1]]}")

