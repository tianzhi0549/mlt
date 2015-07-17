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

class SelectBestDecisionStump:
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
        self.midPoints=[]
        for i in range(len(X[0])):
            sortedData=sorted(zip(X, Y), key=lambda t: t[0][i])
            sortedX=[e[0] for e in sortedData]
            sortedY=[e[1] for e in sortedData]
            iMidPoints=[(sortedX[index][i]+sortedX[index+1][i])/2 \
                for index in range(len(sortedX)-1)]
            iMidPoints.insert(0, -float('inf'))
            self.midPoints.append(iMidPoints)

    def get(self, U):
        theBestDecisionStump=None
        curErrRate=float('inf')
        for i in range(len(X[0])):
            for midPoint in self.midPoints[i]:
                decisionstump=DecisionStump(1, i, midPoint)
                tmpErrRate=decisionstump.getErr(self.X, self.Y, U)
                if tmpErrRate>1/2:
                    tmpErrRate=1-tmpErrRate
                    decisionstump=DecisionStump(-1, i, midPoint)
                if tmpErrRate<curErrRate:
                    theBestDecisionStump=decisionstump
                    curErrRate=tmpErrRate
        return theBestDecisionStump, curErrRate

class AdaBoost:
    def __init__(self, SelectBestModel):
        self.SelectBestModel=SelectBestModel

    def getDiamondt(err):
        return ((1-err)/err)**(1/2)

    def fit(self, X, Y, iterationsCount):
        gs=[]
        alphas=[]
        U=[1/len(X)]*len(X)
        itorNum=0
        selectBestModel=self.SelectBestModel(X, Y)
        while itorNum<iterationsCount:
            itorNum+=1
            model, err=selectBestModel.get(U)
            gs.append(model)
            diamondt=AdaBoost.getDiamondt(err)
            alphas.append(math.log(diamondt))
            #update U
            for i, x in enumerate(X):
                if(model.decision(x)!=Y[i]):
                    U[i]=U[i]*diamondt
                else:
                    U[i]=U[i]/diamondt
        self.gs=gs
        self.alphas=alphas

    def decision(self, X):
        return sign(sum([g.decision(X)*alpha for g, alpha \
            in zip(self.gs, self.alphas)]))

X, Y=LoadData("train.dat")
testX, testY=LoadData("test.dat")
adaBoost=AdaBoost(SelectBestDecisionStump)
adaBoost.fit(X, Y, 300)
print(GetErr(testX, testY, [1/len(testX)]*len(testX), adaBoost))
