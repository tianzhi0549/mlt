from numpy.linalg import norm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math

class KNN:
    def __init__(self, K):
        self.K=K

    def fit(self, X, Y):
        self.X=X
        self.Y=Y

    def predict(self, testX):
        assert(len(self.X)==100)
        XY=sorted(zip(self.X, self.Y), key=lambda xy: \
            norm(np.array(xy[0])-np.array(testX)))
        
        count={}
        for i in range(self.K):
            if XY[i][1] not in count:
                count[XY[i][1]]=1
            else:
                count[XY[i][1]]+=1
        return max(count, key=lambda k: count[k])
    

    def score(self, X, Y):
        correctCount=0
        for i, x in enumerate(X):
            if self.predict(x)==Y[i]:
                correctCount+=1
        return correctCount/len(X)

def LoadData(fileName):
    X, Y=[], []
    fp=open(fileName)
    for line in fp:
        line=line.strip()
        if line!="":
            arr=line.split(" ")
            X.append([float(e) for e in arr[:-1]])
            Y.append(float(arr[-1]))
    fp.close()
    return X, Y

X, Y=LoadData("hw4_knn_train.dat")
testX, testY=LoadData("hw4_knn_test.dat")

knn=KNN(5)
knn.fit(X, Y)
print(1-knn.score(X, Y))
print(1-knn.score(testX, testY))

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X, Y)
print(1-knn.score(X, Y))
print(1-knn.score(testX, testY))
