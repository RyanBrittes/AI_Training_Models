import pandas as pd
from loadData import LoadData
from operationsData import OperationsData

class MenuData():
    def __init__(self):
        self.LData = LoadData()
        self.getData = self.LData.get_data()
        self.OpData = OperationsData()
    
    def menuOperations(self):
        opt = '0'
        while(opt != '7'):
            print("Leia as seguintes opções: ")
            print("\n01-Show all the employees\n02-Show an especific employee\n03-Update any data\n04-Add new column\n05-Delete a row\n06-Delete a data\n07-Exit")
            opt = input("Escolha uma opção para prosseguir: ")
            match opt:
                case '1':
                    print("Mostrando todos os dados:\n")
                    self.LData.show_data()
                case '2':
                    name = input("Digite o nome do funcionário que deseja encontrar: ")
                    self.OpData.getEmployeeName(name)
                case '3':
                    name = input("Digite o nome do funcionário: ")
                    print("\nColunas: ")
                    for i in self.getData.columns:
                        print(i)
                    
                    column = input("Digite o nome da coluna que deseja alterar: ")
                    value = input("Digite o valor que deseja atribuir: ")
                    self.OpData.updateEmployeeName(name, column, value)
                case '4':
                    column = input("Digite o nome da nova coluna: ")
                    self.OpData.addNewColumn(column)
                case '5':
                    name = input("Digite o nome do funcionário que deseja excluir do sistema: ")
                    self.OpData.deleteRow(name)
                case '6':
                    name = input("\nDigite o nome do funcionário: ")
                    print("\nColunas: ")
                    for i in self.getData.columns:
                        print(i)
                        
                    column = input("Digite o nome da coluna que deseja alterar: ")
                    self.OpData.deleteData(name, column)
                case '7':
                    print("\nSaindo...")
                case _:
                    print("Opção inválida!")
                    