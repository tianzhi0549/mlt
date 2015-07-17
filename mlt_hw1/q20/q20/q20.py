import sklearn.svm as svm
import random as rnd
def parseLine(line):
	r1=[]
	r2=[]
	line=line.strip()
	if line=="":
		return None
	else:
		if not (line[0].isdigit() or line[0]=='-' or line[0]=='+'):
			return None
		else:
			arr=line.split()
			for e in arr:
				if e!="":
					if ":" in e:
						r2.append(float(e.split(":")[1]))
					else:
						r1.append(float(e))
			return r1, r2

def loadData():
    X=[]
    Y=[]
    try:
        while True:
            line=input()
            r=parseLine(line)
            if r is not None:
                X.append(r[1])
                Y.append(r[0][0])
    finally:
        return X, Y

def getErrCount(clf, X, Y):
    testY=clf.predict(X)
    errCount=0
    for e in zip(Y, testY):
        if e[0]!=e[1]:
            errCount+=1
    return errCount

gammas=[[10**p ,0] for p in range(5)]
X, Y=loadData()

for j in range(0, 100):
    trainX=[e for e in X]
    trainY=[e for e in Y]
    valX=[]
    valY=[]
    length=len(trainX)
    for i in range(1000):
        index=rnd.randint(0, length-1)
        valX.append(trainX[index])
        valY.append(trainY[index])
        del trainX[index]
        del trainY[index]
        length=length-1

    selectedIndex=-1
    score=-1

    for index, g in enumerate(gammas):
        clf=svm.SVC(C=0.1, gamma=g[0])
        clf.fit(trainX, trainY)
        tmp=clf.score(valX, valY)
        if tmp>score:
            selectedIndex=index
            score=tmp
    gammas[selectedIndex][1]+=1
print(gammas)


