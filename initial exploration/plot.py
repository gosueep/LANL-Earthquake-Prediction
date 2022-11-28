import matplotlib.pyplot as plt
from numpy import arange

dataFile = open('data/testsplit0.csv', 'r', buffering=10000)
dataFile.readline()

outFile = open('plot.csv', 'w')

acous = []
time = []
r = 10000000
s = 1
x = arange(0,r,s)
for _ in range(r):
    line = dataFile.readline()
    outFile.write(line)
    a,t = line.split(',')

    acous.append(int(a))
    time.append(float(t))

plt.subplot(212)
plt.plot(x, acous, 'r')
plt.subplot(211)
plt.plot(x, time, 'b')

plt.title('Initial Plot of Time vs Acoustic Data')
plt.ylabel('Acoustic Data')
plt.xlabel('Time in seconds')
plt.show()
