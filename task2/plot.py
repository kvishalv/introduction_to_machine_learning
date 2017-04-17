import numpy as np
import matplotlib.pyplot as plt

LOG = True

N = 3
col1_means = (1, 1319, 0.0147)
col1_std = (0, 14, 0)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
if LOG:
	rects1 = ax.bar(ind, np.log(col1_means), width, color='r', yerr=np.log(col1_std))
else:
	rects1 = ax.bar(ind, col1_means, width, color='r', yerr=col1_std)

col2_means = (1, 0.0325, 0.0034)
col2_std = (1, 0.000085, 0.00005)
if LOG:
	rects2 = ax.bar(ind + width, np.log(col2_means), width, color='y', yerr=np.log(col2_std))
else:
	rects2 = ax.bar(ind + width, col2_means, width, color='y', yerr=col2_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('log(∆Expression)')
ax.set_title('Expression Level Changes')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('D', '2D', '3D'))

ax.legend((rects1[0], rects2[0]), ('Coll-1', 'Coll-2'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()