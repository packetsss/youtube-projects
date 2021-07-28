import random

def load_puzzle(is_random, seed=0):
    if not is_random:
        random.seed(seed)

    with open("new_puzzle.txt", "r") as f:
        ct = 0
        for line in f:
            ct += 1
        random_line = random.randint(1, ct)
    with open("new_puzzle.txt", "r") as f:
        for i ,line in enumerate(f):
            if i == random_line:
                final_line = line
                break
    final_line = eval(final_line)
    return final_line
    
