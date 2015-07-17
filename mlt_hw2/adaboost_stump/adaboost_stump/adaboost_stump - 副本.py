import math

def sign(x):
    if x>=0:
        return 1
    else:
        return -1

class DecisionStump:
    def __init__(self, s, i, threshold):
        self.s=s
        self.i=i
        self.threshold=threshold

    def decision(self, x):
        return self.s*sign(x[self.i]-self.threshold)

    def getErr(self, X, Y, U):
        return GetErr(X, Y, U, self)

    def __str__(self):
        return "s={0}, i={1}, threshold={2}.".format(self.s, self.i, self.threshold)



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

def GetBestDecisionStump(X, Y, U):
    theBestDecisionStump=None
    curErrRate=float('inf')
    for i in range(len(X[0])):
        sortedData=sorted(zip(X, Y, U), key=lambda t: t[0][i])
        sortedX=[e[0] for e in sortedData]
        sortedY=[e[1] for e in sortedData]
        sortedU=[e[2] for e in sortedData]
        midPoints=[(sortedX[index][i]+sortedX[index+1][i])/2 \
            for index in range(len(sortedX)-1)]
        midPoints.insert(0, -float('inf'))
        for midPoint in midPoints:
            for s in [-1, 1]:
                decisionstump=DecisionStump(s, i, midPoint)
                tmpErrRate=decisionstump.getErr(sortedX, sortedY, sortedU)
                
                if tmpErrRate<curErrRate:
                    theBestDecisionStump=decisionstump
                    curErrRate=tmpErrRate
    return theBestDecisionStump, curErrRate

class AdaBoost:
    def __init__(self, GetBestModel):
        self.GetBestModel=GetBestModel

    def getDiamondt(err):
        return ((1-err)/err)**(1/2)

    def fit(self, X, Y, iterationsCount):
        gs=[]
        alphas=[]
        U=[1/len(X)]*len(X)
        minErr=float('inf')
        itorNum=0
        while itorNum<iterationsCount:
            itorNum+=1
            model, err=self.GetBestModel(X, Y, U)
            gs.append(model)
            diamondt=AdaBoost.getDiamondt(err)
            alphas.append(math.log(diamondt))
            for i, x in enumerate(X):
                if(model.decision(x)!=Y[i]):
                    U[i]=U[i]*diamondt
                else:
                    U[i]=U[i]/diamondt
        self.gs=gs
        self.alphas=alphas

    def decision(self, X):
        assert(self.gs)
        assert(self.alphas)
        return sign(sum([g.decision(X)*alpha for g, alpha \
            in zip(self.gs, self.alphas)]))

X, Y=LoadData("train.dat")
testX, testY=LoadData("test.dat")
adaBoost=AdaBoost(GetBestDecisionStump)
adaBoost.fit(X, Y, 300)
print(GetErr(testX, testY, [1/len(testX)]*len(testX), adaBoost))
