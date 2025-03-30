import os
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from pathlib import Path


# Prioritásos sor osztály
class PriorityQueue:
    def __init__(self) -> None:
        self.queue = []

    def insert(self, priority: int, value: str) -> None:
        """Új adat beszúrás."""
        heapq.heappush(self.queue, (priority, value))

    def pop(self) -> list | None:
        """Queue legkisebb adatának lekérdezése, eltávolítása."""
        return heapq.heappop(self.queue) if self.queue else None

    def display(self) -> list:
        """Rendezett adatok lekérdezése."""
        return sorted(self.queue)

    def visualize(self) -> None:
        """Vizuális megjelenítés plot-ban."""
        if not self.queue:
            print("A prioritásos sor üres!")
            return
        data = sorted(self.queue)
        plt.style.use("classic")
        priorities, values = zip(*data)
        plt.figure(figsize=(5, 8))
        colors = plt.colormaps.get_cmap("tab10_r")(np.linspace(0, 1, len(values)))
        plt.bar(range(len(values)), priorities, tick_label=values, color=colors)
        plt.xlabel("Elemek")
        plt.ylabel("Prioritás")
        plt.title("Prioritásos sor vizualizáció")
        plt.show()

    @staticmethod
    def from_csv(file_path: str) -> "PriorityQueue":
        """CSV file beolvasás, adatok feldolgozása, visszatérés kész prioritásos sorral."""
        df = pd.read_csv("import/PriorityQueue/" + file_path, header=None)
        pq = PriorityQueue()
        for _, row in df.iterrows():
            pq.insert(row[0], row[1])
        return pq


# Ritka mátrix osztály
class SparseMatrix:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.data = {}

    def insert(self, row: int, col: int, value: int) -> None:
        """Beolvasott adatokba új beszúrás."""
        if value != 0:
            self.data[(row, col)] = value

    def get(self, row: int, col: int) -> int:
        """Beolvasott adatok lekérdezése."""
        return self.data.get((row, col), 0)

    def delete(self, row: int, col: int) -> None:
        """Kiválasztott cella adatainak törlése."""
        if (row, col) in self.data:
            del self.data[(row, col)]

    def display(self) -> None:
        """Adatok megjelenítése a console-ban."""
        matrix = np.zeros((self.rows, self.cols))
        for (r, c), v in self.data.items():
            matrix[r, c] = v
        print(matrix)

    def visualize(self) -> None:
        """Vizuális megjelenítés plot-ban."""
        matrix = np.zeros((self.rows, self.cols))
        for (r, c), v in self.data.items():
            matrix[r, c] = v

        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="Blues", interpolation="none")
        plt.colorbar()
        plt.title("Ritka mátrix vizualizáció")
        plt.show()

    @staticmethod
    def from_csv(file_path: str) -> "SparseMatrix | None":
        """CSV file beolvasás, adatok feldolgozása, visszatérés kész mátrixal."""
        df = pd.read_csv("import/SparseMatrix/" + file_path, header=None)
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
class LowerTriangularMatrix:
    def __init__(self, size: int) -> None:
        self.size = size
        self.data = {}

    def insert(self, row: int, col: int, value: int) -> None:
        """Beolvasott adatokba új beszúrás."""
        if row >= col:
            self.data[(row, col)] = value
        else:
            print(f"A hivatkozott index {row} : {col} nem az alsó háromszögmátrix tartományában van.")

    def get(self, row: int, col: int) -> int | Literal[0]:
        """Beolvasott adatok lekérdezése."""
        if row < col:
            return 0
        return self.data.get((row, col), 0)

    def delete(self, row: int, col: int) -> None:
        """Kiválasztott cella adatainak törlése."""
        if (row, col) in self.data:
            del self.data[(row, col)]

    def display(self) -> None:
        """Adatok megjelenítése a console-ban."""
        matrix = np.zeros((self.size, self.size))
        for (r, c), v in self.data.items():
            matrix[r, c] = v
        print(matrix)

    def visualize(self) -> None:
        """Vizuális megjelenítés plot-ban."""
        matrix = np.zeros((self.size, self.size))
        for (r, c), v in self.data.items():
            matrix[r, c] = v

        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="Blues", interpolation="none")
        for r in range(self.size):
            for c in range(self.size):
                plt.text(c, r, str(int(matrix[r][c])), ha="center", va="center", color="black")
        plt.colorbar()
        plt.title("Alsó háromszög mátrix vizualizáció")
        plt.show()

    @staticmethod
    def from_csv(file_path: str) -> "LowerTriangularMatrix":
        """CSV file beolvasás, adatok feldolgozása, visszatérés kész mátrixal."""
        df = pd.read_csv("import/LowerTriangularMatrix/" + file_path, header=None)
        matrix = df.to_numpy()
        size = matrix.shape[0]
        lower_matrix = LowerTriangularMatrix(size)

        for r in range(size):
            for c in range(size):
                if r >= c:
                    lower_matrix.insert(r, c, matrix[r, c])

        return lower_matrix


# Program belépse
if __name__ == "__main__":
    csv_files_by_subfolder = {}

    # Import mappa és CSV fájlok beolvasása
    for subfolder in Path("import").iterdir():
        if subfolder.is_dir():
            csv_files = {i + 1: file.name for i, file in enumerate(subfolder.glob("*.csv"))}
            if csv_files:
                csv_files_by_subfolder[subfolder.name] = csv_files

    def clear_console() -> None:
        """Console tartalmának teljes törlése."""
        os.system("cls" if os.name == "nt" else "clear")

    def showFiles(file_type: str) -> None:
        """Kiválasztható elemek kiíratása."""
        for file_id, filename in csv_files_by_subfolder[file_type].items():
            print(f"{file_id} - {filename}")

    clear_console()

    while True:
        print("\n===== ADATSZERKEZET MENÜ =====")
        print("1 - Prioritásos sor kezelése")
        print("2 - Ritka mátrix betöltése CSV-ből és megjelenítése")
        print("3 - Alsó háromszög mátrix betöltése CSV-ből és megjelenítése")
        print("0 - Kilépés")

        try:
            choice = int(input("Válassz egy opciót: "))

            if choice == 1:
                print("\n--- Prioritásos sor ---")
                showFiles("PriorityQueue")

                selected_file = int(input("Add meg a CSV fájl számát: "))
                pq = PriorityQueue.from_csv(csv_files_by_subfolder["PriorityQueue"][selected_file])
                print("Tartalom:", pq.display())
                pq.visualize()
                clear_console()

            elif choice == 2:
                print("\n--- Ritka mátrix betöltése ---")
                showFiles("SparseMatrix")

                selected_file = int(input("Add meg a CSV fájl számát: "))
                sm = SparseMatrix.from_csv(csv_files_by_subfolder["SparseMatrix"][selected_file])
                if sm:
                    sm.display()
                    sm.visualize()
                else:
                    print("Nem ritka mátrix vagy nem sikerült beolvasni a fájlt.")
                clear_console()

            elif choice == 3:
                print("\n--- Alsó háromszög mátrix betöltése ---")
                showFiles("LowerTriangularMatrix")

                selected_file = int(input("Add meg a CSV fájl számát: "))
                ltm = LowerTriangularMatrix.from_csv(csv_files_by_subfolder["LowerTriangularMatrix"][selected_file])
                ltm.display()
                ltm.visualize()
                clear_console()

            elif choice == 0:
                print("Kilépés...")
                break

            else:
                clear_console()
                print("Érvénytelen opció! Próbáld újra.")

        except ValueError:
            clear_console()
            print("Hibás bevitel! Kérlek, számot adj meg.")
        except KeyError:
            clear_console()
            print("Hibás bevitel! Kérem, a listában szereplő elemek közül válasszon.")
        except (FileNotFoundError, PermissionError) as fileError:
            clear_console()
            print("Hibás bevitel! Ilyen fájl nem létezik!")
