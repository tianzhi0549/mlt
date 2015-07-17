from decision_tree import *
import random
import math

def Boostrap(size, X, Y):
    newX, newY=[], []
    length=len(X)
    for i in range(int(size)):
        index=random.randint(0, length-1)
        newX.append(X[index])
        newY.append(Y[index])
    return newX, newY

class RandomForest:
    def __init__(self, treeCount):
        self.treeCount=treeCount

    def fit(self, X, Y, maxBranchCount=float('inf')):
        trees=[]
        for i in range(self.treeCount):
            tree=DT(maxBranchCount)
            newX, newY=Boostrap(len(X), X, Y)
            tree.fit(newX, newY)
            trees.append(tree)
        self.trees=trees
    
    def score(self, X, Y):
        errCount=0
        for i, x in enumerate(X):
            if self.predict(x)!=Y[i]:
                errCount+=1
        return 1-errCount/len(X)

    def predict(self, x):
        sum=0
        for tree in self.trees:
            sum+=tree.predict(x)
        return math.copysign(1, sum)

