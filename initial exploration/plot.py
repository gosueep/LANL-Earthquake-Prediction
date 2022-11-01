import matplotlib.pyplot as plt

dataFile = open('data/testsplit0.csv', 'r', buffering=10000)
dataFile.readline()

outFile = open('plot.csv', 'w')

acous_x = []
time_y = []
for _ in range(1000):
    line = dataFile.readline()
    outFile.write(line)
    x, y = line.split(',')

    acous_x.append(int(x))
    time_y.append(float(y))


plt.plot(time_y, acous_x)
plt.title('Initial Plot of Time vs Acoustic Data')
plt.ylabel('Acoustic Data')
plt.xlabel('Time in seconds')
plt.show()
