import matplotlib.pyplot as plt
import numpy  as np
import math
def distance(x0, y0, x1, y1):
    return ((x0-x1)**2+(y0-y1)**2)**(1/2)

def decision(x, y):
    return math.copysign(1, math.exp(-distance(-1, 1, x, y)**2)- \
        math.exp(-distance(1, 0, x, y)**2))

X = np.linspace(-5, 5)
Y = np.linspace(-5, 5)
for x in X:
    for y in Y:
        if decision(x, y)==1:
            plt.plot(x, y, "r+")
        else:
            plt.plot(x, y, "bo")
plt.show()
