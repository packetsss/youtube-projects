import time
import numpy as np
from copy import deepcopy
from reader import load_puzzle

class Solve:
    def __init__(self, puzzle):
        self.puzzle = np.array(puzzle)
        self.attempt = 0
        self.sets = set(range(1, 10))
    
    @staticmethod
    def mini_grid(puzzle):
        return np.array([np.reshape(puzzle[i:i + 3, j: j + 3], (1, 9))[0] for i in range(0, 7 ,3) for j in range(0, 7, 3)])
    
    @staticmethod
    def mini_grid_index(i, j):
        return i // 3 * 3 + i // 3, i % 3 * 3 + j % 3
    
    @staticmethod
    def mini_index_block(i, j):
        return i // 3 * 3, i // 3 * 3 + 3, j // 3 * 3, j // 3 * 3 + 3
    
    def possibilities(self):
        for i in range(9):
            for j in range(9):
                if self.puzzle[i, j] == 0 or len(str(self.puzzle[i, j])) > 1:
                    row = self.puzzle[i, :]
                    column = self.puzzle[:, j]

                    mii, mif, mji, mjf = self.mini_index_block(i, j)
                    mini_puzzle = np.reshape(self.puzzle[mii:mif, mji:mjf], (1, 9))[0]

                    try:
                        self.puzzle[i, j] = int("".join(map(str, self.sets.difference(set(row)).difference(set(column)).difference(set(mini_puzzle)))))

                    except ValueError:
                        print("Failed at possibilities")
                        return False
        
        return True
    
    def find_unique(self):
        for i in range(9):
            for j in range(9):
                if len(str(self.puzzle[i, j])) > 1:
                    for k in str(self.puzzle[i, j]):
                        row = self.puzzle[i, :]
                        column = self.puzzle[:, j]

                        mii, mif, mji, mjf = self.mini_index_block(i, j)
                        mini_puzzle = np.reshape(self.puzzle[mii:mif, mji:mjf], (1, 9))[0]
                        
                        if len(str(self.puzzle[i, j])) > 1 and ("".join(map(str, row)).count(k) == 1 or "".join(map(str, column)).count(k) == 1 or "".join(map(str, mini_puzzle)).count(k) == 1):
                            try:
                                self.puzzle[i, :] = [int(s.replace(k, "")) for s in list(map(str, row))]
                                self.puzzle[:, j] = [int(s.replace(k, "")) for s in list(map(str, column))]
                                self.puzzle[i, j] = int(k)
                            except ValueError:
                                print(f"Failed at find_unique\ni: {i}, j: {j}")
                                return False
                            
        return True
    
    
    
solver = Solve(load_puzzle(is_random=False))

solver.possibilities()
solver.find_unique()

print(solver.puzzle)