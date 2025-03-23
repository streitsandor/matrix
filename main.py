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

        plt.figure(figsize=(8, 5))
        plt.bar(range(len(values)), priorities, tick_label=values, color="skyblue")
        plt.xlabel("Elemek")
        plt.ylabel("Prioritás")
        plt.title("Prioritásos sor vizualizáció")
        plt.show()


# Ritka mátrix osztály
class Class2:
    def __init__(self):
        return


# Alsó háromszög mátrix osztály
class Class3:
    def __init__(self):
        return

  # Példa használatra


if __name__ == "__main__":
    print("--- Prioritásos sor teszt ---")
    pq = PriorityQueue()
    pq.insert(2, "Második")
    pq.insert(1, "Első")
    pq.insert(3, "Harmadik")
    print(pq.display())
     
    pq.visualize()
  


