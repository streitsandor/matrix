import heapq
import numpy as np
import pandas as pd
import icecream as ic
import matplotlib.pyplot as plt


# Prioritásos sor osztály
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def insert(self, priority, value):
        heapq.heappush(self.queue, (priority, value))

    def pop(self):
        return heapq.heappop(self.queue) if self.queue else None

    def display(self):
        return sorted(self.queue)

    def visualize(self):
        if not self.queue:
            print("A prioritásos sor üres!")
            return
        data = sorted(self.queue)
        priorities, values = zip(*data)

        plt.style.use('classic')
        plt.figure(figsize=(5, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(values)))
        plt.bar(range(len(values)), priorities, tick_label=values, color=colors)
        plt.xlabel("Elemek")
        plt.ylabel("Prioritás")
        plt.title("Prioritásos sor vizualizáció")
        plt.show()

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
    print("--- Prioritásos sor teszt ---")
    pq = PriorityQueue()
    pq.insert(2, "Második")   
    pq.insert(1, "Első")
    pq.insert(3, "Harmadik")
    pq.insert(7, "Hetedik")
    
    print(pq.display())     
    pq.visualize()
    
    print("\n--- Ritka mátrix teszt (CSV fájlból) ---")
    sm = SparseMatrix.from_csv("import/matrix.csv")
    if sm:
        sm.display()
        sm.visualize()
        
