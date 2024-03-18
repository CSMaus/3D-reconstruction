if __name__ == '__main__':
    inputList = []
    names = []
    scores = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        inputList.append([name, score])
        names.append(name)
        scores.append(score)
print(scores)
minv = min(scores)

for val in scores:
    if val == minv:
        scores.remove(val)
print(scores)
minV2 = min(scores)

res = []
for i in range(len(names)):
    if inputList[i][1] == minV2:
        res.append(names[i])
res.sort()

for r in res:
    print(r)