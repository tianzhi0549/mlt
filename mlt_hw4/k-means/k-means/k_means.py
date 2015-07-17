import random as rnd
from numpy.linalg import norm
import numpy as np

class KMeans:
    def __init__(self, k):
        self.k=k

    def getMinDistanceUIndex(x, us):
        minD=float('inf')
        minIndex=0
        for i, u in enumerate(us):
            d=norm(np.array(x)-np.array(u))
            if d<minD:
                minD=d
                minIndex=i
        return minIndex
    def getEmptyIndex(splitedX):
        for i, x in enumerate(splitedX):
            if len(x)==0:
                return i
        return -1

    def split(self, X, u):
        while True:
            Ein=0
            splitedX=[[] for i in range(len(u))]
            for x in X:
                index=KMeans.getMinDistanceUIndex(x, u)
                Ein+=norm(np.array(x)-np.array(u[index]))**2
                splitedX[index].append(x)
            emptyIndex=KMeans.getEmptyIndex(splitedX)
            if emptyIndex==-1:
                return splitedX, Ein/len(X)
            else:
                u[emptyIndex]=X[rnd.randint(0, len(X)-1)]

    def genUs(self, X):
        u=[]
        for i in range(self.k):
            u.append(X[rnd.randint(0, len(X)-1)])
        return u
    
    def computeCenter(self, X):
        sum=np.array([0.0 for i in range(len(X[0]))])
        for x in X:
            #print("oldSum:", sum)
            #print("x:", x)
            sum+=np.array(x)
            #print("newSum:", sum)
        return sum/len(X)


    def fit(self, X):
        u=self.genUs(X)
        lastEin=float('inf')
        while True:
            splitedX, Ein=self.split(X, u)
            
            if abs(Ein-lastEin)<0.01:
                return Ein
            else:
                lastEin=Ein

            u=[]
            for x in splitedX:
                u.append(self.computeCenter(x))


def LoadData(fileName):
    X=[]
    fp=open(fileName)
    for line in fp:
        line=line.strip()
        if line!="":
            arr=line.split(" ")
            X.append([float(e) for e in arr])
    fp.close()
    return X

X=LoadData("hw4_kmeans_train.dat")
kmeans=KMeans(10)
print(kmeans.fit(X))
