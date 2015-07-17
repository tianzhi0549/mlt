import matplotlib.pyplot as plt
def drawPoints(X, Y):
    for x, y in zip(X, Y):
        x=t(x)
        if y>0:
            style="ro"
        else:
            style="bo"
        plt.plot(x[0], x[1], style)

def drawLine(A, B, C):
    plt.title('A={0}, B={1}, C={2}'.format(A, B, C))
    if A==0 and B!=0:
        plt.plot([-20, 20], [-C/B, -C/B])
    if B==0 and A!=0:
        plt.plot([-C/A, -C/A], [-20, 20])
    else:
        plt.plot([-20, 20], [(-C+20*A)/B, (-C-20*A)/B])

def t(x):
    return [x[0]**2-2*x[0]+3, x[1]**2-2*x[1]-3]

def getLine(A, B, C):
    return lambda x, y: A*x+B*y+C

def getErrCount(X, Y, fn):
    pass


plt.ylabel('y')
plt.ylabel('x')
plt.axis([-20, 20, -20, 20])

X=[[1, 0], [0, 1], \
       [0, -1], [-1, 0], \
       [0, 2], [0, -2], \
       [-2, 0]]

Y=[-1, -1, -1, 1, 1, 1, 1]

drawPoints(X, Y)
drawLine(0, 1, -4.5)
plt.show()

