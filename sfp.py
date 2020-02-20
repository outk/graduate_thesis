import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from matplotlib.dates import DateFormatter

with open('./fpidentityall.txt') as f:
    fidelityls = []
    for s in f.readlines():
        datals = s.strip().split()
        fidelityls.append(float(datals[0]))

fig = plt.figure()

x = np.arange(len(fidelityls))

y = np.array(fidelityls)
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x, y, label='identity')

with open('./fpusedmoqall.txt') as f:
    fidelityls = []
    for s in f.readlines():
        datals = s.strip().split()
        fidelityls.append(float(datals[0]))

x = np.arange(len(fidelityls))

y = np.array(fidelityls)
ax1.plot(x, y, label='usedmoq')

plt.ylabel("fidelity")
plt.xlabel("iteration time")

plt.legend(loc='lower right')

# plt.yscale('log')

plt.show()