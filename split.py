with open('data/train.csv', 'r', buffering=10000) as f:
    splitAmt = 63000000

    for i in range(10):
        with open(f'testsplit{i}.csv', 'w') as out:

            for _ in range(splitAmt):
                out.write(f.readline())
