import numpy as np
import pandas as pd

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/DataManipulation/file/Macrofitas - Guaraguacu.csv')

species = dataset['especie'].tolist()

listSpecies = []


for i in range(len(species)):
    if(species[i] != species[i-1]):
        #listSpecies.append(species[i])
        print(species[i])

#print(listSpecies)

list = ['blue', 'cray', 'red', 'purple', 'other']
indexList = []
dictList = {}

for i in range(len(list)):
    indexList.append(i)
    dictList[list[i]] = i

identit = np.eye(len(indexList))

#print(f"Dicionário: {dictList}")
#print(identit)
#print(indexList)