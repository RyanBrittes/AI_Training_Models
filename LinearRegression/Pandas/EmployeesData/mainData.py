import pandas as pd
from loadData import LoadData

df = LoadData()
dataF = df.get_data()

print(dataF.loc[[1, 2], ['age']])
