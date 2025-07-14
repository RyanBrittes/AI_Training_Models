import pandas as pd
from loadData import LoadData

class MenuData():
    def __init__(self):
        self.LData = LoadData()
    
    def menu(self):
        opt = '0'
        while(opt != '7'):
            print("Leia as seguintes opções: ")
            print("\n01-Show all the employees\n02-Show an especific employee\n03-Update any data\n04-Add new column\n05-Delete a row\n06-Delete a data\n07-Exit")
            opt = input("Escolha uma opção para prosseguir: ")
            match opt:
                case '1':
                    self.LData.show_data()
                case '2':
                    print("\nOpt 2")
                case '3':
                    print("\nOpt 3")
                case '4':
                    print("\nOpt 4")
                case '5':
                    print("\nOpt 5")
                case '6':
                    print("\nOpt 6")
                case '7':
                    print("\nSaindo...")
                case _:
                    print("Opção inválida!")
                    