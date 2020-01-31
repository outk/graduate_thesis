with open("./result/qubit/iterative/usedmoq100/resultIterationTime.txt") as f:
    ls = []
    for s in f.readlines():
        ls.extend(map(int, s.strip().split()))

print(ls)
print(sum(ls))