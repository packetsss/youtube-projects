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

    def elimination(self):
        for i in range(9):
            for j in range(9):
                if len(str(self.puzzle[i, j])) > 1:
                    l = [str(self.puzzle[i, j])]

                    for k in range(9):
                        if len(str(self.puzzle[i, :][k])) < 2 or k == j:
                            continue

                        if set(str(self.puzzle[i, :][k])).issubset(set(str(self.puzzle[i, j]))):
                            l.append(str(self.puzzle[i, :][k]))
                        
                        if 1 < len(l) == len(max(l, key=len)):
                            ll = []
                            for s in self.puzzle[i, :]:
                                s = str(s)
                                for ss in max(l, key=len):
                                    if s not in l:
                                        s = s.replace(ss, "")
                                
                                ll.append(s)
                            
                            try:
                                self.puzzle[i, :] = np.array([int(i) for i in ll])

                            except ValueError:
                                print(f"Failed at elimination")
                                return False

                    l = [str(self.puzzle[i, j])]
                    for kk in range(9):
                        if len(str(self.puzzle[:, j][kk])) < 2 or kk == i:
                            continue
                            
                        if set(str(self.puzzle[:, j][kk])).issubset(set(str(self.puzzle[i, j]))):
                            l.append(str(self.puzzle[:, j][kk]))
                        
                        if 1 < len(l) == len(max(l, key=len)):
                            ll = []

                            for s in self.puzzle[:, j]:
                                s = str(s)
                                for ss in max(l, key=len):
                                    if s not in l:
                                        s = s.replace(ss, "")
                                ll.append(s)
                            
                            try:
                                self.puzzle[:, j] = np.array([int(i) for i in ll])
                            except ValueError:
                                print("Failed at elimination")
                                return False
        return True
    
    def check(self):
        for i in range(9):
            row = [i for i in self.puzzle[i, :] if len(str(i)) == 1]

            if len(set(row)) != len(row):
                print("Failed at check: row")
                return False
            
            column = [i for i in self.puzzle[:, i] if len(str(i)) == 1]
            if len(set(column)) != len(column):
                print("Failed at check: column")
                return False
            
            mii, mif, mji, mjf = self.mini_index_block(i, i % 3 * 3)
            mini_puzzle = np.reshape(self.puzzle[mii:mif, mji:mjf], (1, 9))[0]
            mini_row = [i for i in mini_puzzle if len(str(i)) == 1]
            if len(set(mini_row)) != len(mini_row):
                print("Failed at check: mini_grid")
                return False
        return True
    
    def get_next_cell(self):
        def min_len(self):
            length = 10
            for i in range(9):
                for j in range(9):
                    if len(str(self.puzzle[i, j])) == 2:
                        return 2
                    if 1 < len(str(self.puzzle[i, j])) < length:
                        length = len(str(self.puzzle[i, j]))
            return length

        minimum_length = min_len(self)
        for i in range(9):
            for j in range(9):
                if minimum_length == len(str(self.puzzle[i, j])):
                    return i, j
        
        return False
    
    def solve(self):
        puzzle_copy = deepcopy(self.puzzle)

        try:
            i, j = self.get_next_cell()
        except TypeError as e:

            if sum(sum(self.puzzle)) == 405:
                return True
            else:
                return False
        
        for k in [int(k) for k in str(self.puzzle[i, j])]:
            print("Try inserting a number")
            self.puzzle[i, j] = k

            if not self.possibilities() or not self.find_unique() or not self.elimination() or not self.check():
                self.puzzle = deepcopy(puzzle_copy)
                continue
            
            self.find_unique()
            if self.solve():
                return True

            self.puzzle = deepcopy(puzzle_copy)
        
        print("Restoring to the previous puzzle")
        self.attempt += 1
        return False

def main():
    start_time = time.time()
    
    solver = Solve(load_puzzle(is_random=True))
    print(solver.puzzle)
    solver.possibilities()
    solver.solve()

    end_time = time.time()
    print(f"{solver.puzzle}\nAttempts: {solver.attempt}\nTime: {end_time - start_time}")


if __name__ == '__main__':
    main()