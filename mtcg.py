import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['case1', 'case2', 'case3']
identity = [27027, 987551, 662991]
usedmoq = [27814, 982521, 1104963]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, identity, width, label='identity')
rects2 = ax.bar(x + width/2, usedmoq, width, label='usedmoq')

ax.set_ylabel('total iteration time')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()