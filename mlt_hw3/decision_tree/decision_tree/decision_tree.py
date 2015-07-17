class Gini:
    def GetIndex(Y):
        count={}
        for y in Y:
            if y not in count:
                count[y]=0
            count[y]=count[y]+1
        giniIndex=1
        for k in count:
            giniIndex-=(count[k]/len(Y))**2
        return giniIndex

    def GetIndexTotal(Ys):
        sum=0
        for Y in Ys:
            sum+=len(Y)*Gini.GetIndex(Y)
        return sum

class Branch:
    count=0
    def __init__(self):
        Branch.count=Branch.count+1

    def __call__(self, x):
        return self.makeDecision(x, self.threshold, self.dimension)
    
    def makeDecision(self, x, threshold, dimension):
        if x[dimension]<threshold:
            return 0
        else:
            return 1

    def split(self, X, Y, threshold, dimension):
        X1, Y1=[], []
        X2, Y2=[], []
        for i, x in enumerate(X):
            if self.makeDecision(x, threshold, dimension)==0:
                X1.append(x)
                Y1.append(Y[i])
            else:
                X2.append(x)
                Y2.append(Y[i])
        return X1, Y1, X2, Y2

    def getMidPoints(self, X):
        midPoints=[]
        for dimension in range(0, len(X[0])):
            sortedXDimension=sorted(set([e for e in zip(*X)][dimension]))
            midPoints.append([])
            for i in range(len(sortedXDimension)-1):
                midPoint=(sortedXDimension[i]+sortedXDimension[i+1])/2
                midPoints[dimension].append(midPoint)
        return midPoints

    def fit(self, X, Y):
        assert len(X)>1, len(X)
        midPoints=self.getMidPoints(X)
        resultDimension=0
        resultThreshold=0
        resultX1, resultY1=[], []
        resultX2, resultY2=[], []
        minGiniIndex=float('inf')
        for dimension in range(len(X[0])):
            for threshold in midPoints[dimension]:
                X1, Y1, X2, Y2=self.split(X, Y, threshold, dimension)
                newGiniIndex=Gini.GetIndexTotal([Y1, Y2])
                if newGiniIndex<minGiniIndex:
                    minGiniIndex=newGiniIndex
                    resultDimension=dimension
                    resultThreshold=threshold
                    resultX1, resultY1=X1, Y1
                    resultX2, resultY2=X2, Y2
        self.threshold=resultThreshold
        self.dimension=resultDimension
        return resultX1, resultY1, resultX2, resultY2

    def route(self, x):
        if self(x)==0:
            return self.l
        else:
            return self.r

    def setR(self, r):
        self.r=r

    def setL(self, l):
        self.l=l

class Leaf:
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
        count={}
        for y in Y:
            if y not in count:
                count[y]=0
            else:
                count[y]=count[y]+1
        self.value=max(count, key=lambda k: count[k])

    def getValue(self):
        return self.value

class DT:
    def __init__(self, maxBranchCount=float('inf')):
        self.root=None
        self.maxBranchCount=maxBranchCount

    def score(self, X, Y):
        errCount=0
        for i, x in enumerate(X):
            if self.predict(x)!=Y[i]:
                errCount=errCount+1
        return 1-errCount/len(X)
    
    def getBranchCount(self):
        return self.branchCount

    def predict(self, x):
        assert(self.root)
        curNode=self.root
        while not isinstance(curNode, Leaf):
            curNode=curNode.route(x)
        return curNode.getValue()

    def createTree(self, X, Y):
        if Gini.GetIndex(Y)==0 or \
            self.branchCount>=self.maxBranchCount:
            return Leaf(X, Y)
        else:
            self.branchCount+=1
            branch=Branch()
            X1, Y1, X2, Y2=branch.fit(X, Y)
            branch.setL(self.createTree(X1, Y1))
            branch.setR(self.createTree(X2, Y2))
            return branch

    def fit(self, X, Y):
        self.branchCount=0
        self.root=self.createTree(X, Y)

