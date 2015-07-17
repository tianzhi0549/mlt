import math
import numpy as np
def sign(x):
    if x>=0:
        return 1
    else:
        return -1

class KernelRidgeReg:
    def __init__(self, kernel,  lmbd):
        self.kernel=kernel
        self.lmbd=lmbd

    def fit(self, X, Y):
        K=GenKernelMatrix(X, self.kernel)
        self.beta=(self.lmbd*np.eye(K.shape[0])+K).I*np.mat(Y).T
        self.X=X
    
    def decision(self, x):
        a=[self.kernel.get(e, x) for e in self.X]
        return sign(np.dot(a, self.beta.A1))
        #return sign(np.dot(x, self.beta.A1))

def LoadData(fileName):
    X=[]
    Y=[]
    fp=open(fileName)
    for line in fp:
        line=line.strip()
        if line!="":
            arr=line.split(" ")
            X.append([float(e) for e in arr[:-1]])
            Y.append(float(arr[-1]))

    return X, Y

def GetErr(X, Y, U, model):
    errCount=0
    for i, x in enumerate(X):
        if model.decision(x)!=Y[i]:
            errCount=errCount+1*U[i]
    return errCount/sum(U)

class GaussianRBFKernel:
    def __init__(self, gamma):
        self.gamma=gamma

    def get(self, x1, x2):
        return np.exp(-self.gamma* \
            np.linalg.norm(np.array(x1)-np.array(x2))**2)


def GenKernelMatrix(X, kernel):
    N=len(X)
    K=[[0]*N for e in range(0, N)]
    for n in range(N):
        for m in range(N):
            tmp=kernel.get(X[n], X[m])
            K[n][m]=tmp
    return np.matrix(K)


X, Y=LoadData("all.dat")
trainX=X[:400]
trainY=Y[:400]
testX=X[400:]
testY=Y[400:]
for gamma in (32, 2, 0.125):
    for lmbd in (0.001, 1, 1000):
        krr=KernelRidgeReg(GaussianRBFKernel(gamma), lmbd)
        krr.fit(trainX, trainY)
        print("gamma={0}, lambda={1}.".format(gamma, lmbd))
        print("Etest=", GetErr(testX, testY, [1/len(testX)]*len(testX), krr))

