# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited

d = {}

with open("puzzle.txt", "r") as f:
    for line in f:
        line = str(line).replace(".", "0")
        lst = [[int(i) for i in line[j:(j + 9)]] for j in range(0, 81, 9)]
        print(lst)

        with open("new_puzzle.txt", "a") as ff:
            print(lst, file=ff)