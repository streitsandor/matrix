import numpy as np
import pandas as pd
import icecream as ic
import matplotlib.pyplot as plt


# Prioritásos sor osztály
class Class1:
    def __init__(self):
        return


# Ritka mátrix osztály
class Class2:
    def __init__(self):
        return


# Alsó háromszög mátrix osztály
class LowerTriangularMatrix:
    size = 0
    testMatrix = []
    lowerTriangularMatrixData = []
    def __init__(self,N):
        self.size = N
        self.lowerTriangularMatrixData = [0 for _ in range(self.size * self.size)]
        return    
    def createTestMatrix(self):
        self.testMatrix = [[(i) + j * self.size + 1 for i in range(self.size)] for j in range(self.size)] 
        return
    def printTestMatrix(self,mode):
        if mode == 'byRow':
            for row in self.testMatrix:
                print(row)
        if mode == 'byIndex':
            for j in range(self.size):
                for i in range(self.size):
                    print("["+str(j)+","+str(i)+"]:"+str(self.testMatrix[j][i]))
    def getIndex(self, i, j):
        if j <= i and i < self.size:
            k = ((i * (i + 1)) // 2) + j
        else:
            k = ((self.size - 1) * self.size) // 2 + self.size 
        return k
    def setData(self, i, j, data):
        if j <= i and i < self.size:
            k = self.getIndex(i,j)
            self.lowerTriangularMatrixData[k] = data
        else:
            print("A hivatkozott index " + str(i) + ":" + str(j)+ " nem az alsó háromszög mátrix tartományában van.")
    def convertMatrixToLowerlowerTriangularMatrix(self):
        for j in range(self.size):
            for i in range(self.size):
                self.setData(j,i,self.testMatrix[j][i])
    def displayLowerlowerTriangularMatrix(self):
        s = ""
        for i in self.lowerTriangularMatrixData:
            s = s + str(i) + " "
        print(s)
    def getLowerTruangularMatrixFromLowerTriangularData(self):
        matrix = [[0 for _ in range(self.size)] for _ in range(self.size)]
        index = 0

        for i in range(self.size):
            for j in range(i + 1):
                if index < len(self.lowerTriangularMatrixData):
                    matrix[i][j] = self.lowerTriangularMatrixData[index]
                    index += 1
                else:
                    break
        return matrix    
    def visualizeLowerTriangleMatrixData(self):
        tempMatrix = self.getLowerTruangularMatrixFromLowerTriangularData()
        plt.figure(figsize=(6, 6))
        plt.imshow(tempMatrix, cmap='Blues', interpolation='none')

        for i in range(len(tempMatrix)):
            for j in range(len(tempMatrix[0])):
                plt.text(j, i, str(tempMatrix[i][j]), ha='center', va='center', color='black')

        plt.title("Alsó háromszög mátrix vizualizálása")
        plt.colorbar()
        plt.show()  


# Program belépése
if __name__ == "__main__":
    # Class1
    # Class2
    # Class3
    #LowerTriangularMatrix
    lowerTriangularMatrixData = LowerTriangularMatrix(4)
    lowerTriangularMatrixData.createTestMatrix()
    print("Teszt mátrix")
    lowerTriangularMatrixData.printTestMatrix('byRow')
    print("----------------------------------")
    print("Az alsó háromszög mátrix adatrosának összeállítása.")
    lowerTriangularMatrixData.convertMatrixToLowerlowerTriangularMatrix()
    print("----------------------------------")
    print("Az alsó háromszög mátrix adatsora.")
    lowerTriangularMatrixData.displayLowerlowerTriangularMatrix()
    print("----------------------------------")
    print("Az alsó háromszög mátrix vizualizálása matplotlib.pyplot-al.")
    lowerTriangularMatrixData.visualizeLowerTriangleMatrixData()    
    teszt = Class1
