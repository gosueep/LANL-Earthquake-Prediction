
i = 0
for line in open('data/testsplit6.csv'):
    i += 1
    if len(line.split(',')) != 2:
        print(i)

# 38931265
