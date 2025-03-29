import numpy as np
import pandas as pd
import icecream as ic
import matplotlib.pyplot as plt


# Prioritásos sor osztály
class Class1:
    def __init__(self):
        return

# Ritka mátrix osztály
class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = {}

    def insert(self, row, col, value):
        if value != 0:
            self.data[(row, col)] = value

    def get(self, row, col):
        return self.data.get((row, col), 0)

    def delete(self, row, col):
        if (row, col) in self.data:
            del self.data[(row, col)]

    def display(self):
        matrix = np.zeros((self.rows, self.cols))
        for (r, c), v in self.data.items():
            matrix[r, c] = v
        print(matrix)

    def visualize(self):
        matrix = np.zeros((self.rows, self.cols))
        for (r, c), v in self.data.items():
            matrix[r, c] = v

        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="Blues", interpolation="none")
        plt.colorbar()
        plt.title("Ritka mátrix vizualizáció")
        plt.show()

    @staticmethod
    def from_csv(file_path):
        df = pd.read_csv(file_path, header=None)
        matrix = df.to_numpy()
        total_elements = matrix.size
        zero_elements = np.count_nonzero(matrix == 0)

        if zero_elements / total_elements > 0.5:  # Ritka mátrix ellenőrzés
            sparse_matrix = SparseMatrix(*matrix.shape)
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    if matrix[r, c] != 0:
                        sparse_matrix.insert(r, c, matrix[r, c])
            return sparse_matrix
        else:
            print("Nem ritka mátrix.")
            return None

# Alsó háromszög mátrix osztály
class Class3:
    def __init__(self):
        return

# Program belépése
if __name__ == "__main__":
    print("\n--- Ritka mátrix teszt (CSV fájlból) ---")
    sm = SparseMatrix.from_csv("import/matrix.csv")
    if sm:
        sm.display()
        sm.visualize()