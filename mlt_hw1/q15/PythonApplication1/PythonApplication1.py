from sklearn import svm
import numpy as np
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

def loadData(fileName):
    X=[]
    Y=[]
    fp=open(fileName)
    for line in fp:
        r=parseLine(line)
        if r is not None:
            X.append(r[1])
            Y.append(r[0][0])
    fp.close()
    return X, Y
def sum(v):
    sum=0
    for x in v[0]:
        sum+=abs(x)
    return sum


X, Y=loadData("train0.svm")
testX, testY=loadData("test0.svm")
for p in range(-3, 2):
    print("C={0}".format(10**p))
    clf=svm.SVC(C=10**p, kernel="rbf", gamma=100)
    clf.fit(X, Y)
    print(clf.predict(X[0]))
    #print(len(clf.support_))
    #sum=0
    #for i in clf.support_:
    #    sum+=(1-Y[i]*(clf.decision_function(X[i])[0][0]))
    #print(sum)
    #print(clf.score(testX, testY))

    print(np.linalg.norm((np.mat(clf.dual_coef_)*np.mat(clf.support_vectors_)).A[0]))
#for i in clf.dual_coef_[0]:
#    print(i)


