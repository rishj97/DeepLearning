import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

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
while(i < len(sys.argv)):
    print(i)
    data = np.genfromtxt(sys.argv[i], delimiter=',', names=True)
    plt.plot(data['Step'], data['Value'], c=colors[i-3], label=str(sys.argv[i + 1]))
    i += 2

plt.legend()
plt.savefig(str(sys.argv[1]) + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + '.png')
