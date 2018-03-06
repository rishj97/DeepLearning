import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from scipy.interpolate import interp1d, spline
from scipy import interpolate
from scipy.signal import savgol_filter
# Arguments: 1: Plot title
#            2: Y axis label
#            3: 1st csv file
#            4: 1st csv file label
#            5  2nd csv file
#            6: 2nd csv file label
#            7: ...

plt.title(str(sys.argv[1]))

plt.xlabel('Epoch number')
plt.ylabel(str(sys.argv[2]))
colors = ['r', 'b', 'g', 'c', 'm', 'y']
i = 3
c = 0
file_name = ''
while(i < len(sys.argv)):
    file_name += str(sys.argv[i])
    data = np.genfromtxt(sys.argv[i], delimiter=',', names=True)
    x = np.array(data['Step'])
    y = np.array(data['Value'])
    y_smooth = savgol_filter(y, 7, 2)
    plt.plot(x, y_smooth, c=colors[c], label=str(sys.argv[i + 1]))
    c += 1
    i += 2

plt.legend()
plt.savefig(str(sys.argv[1]) + '.png')
