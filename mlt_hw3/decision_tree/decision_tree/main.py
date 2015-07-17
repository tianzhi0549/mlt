from data import *
from random_forest import *
from decision_tree import *
X, Y=LoadData("hw3_train.dat")
testX, testY=LoadData("hw3_test.dat")

def Q17_Q18():
    EinSum=0
    EoutSum=0
    count=1
    for i in range(count):
        rf=RandomForest(300)
        rf.fit(X, Y)
        EinSum+=1-rf.score(X, Y)
        EoutSum+=1-rf.score(testX, testY)
    print(EinSum/count, EoutSum/count)

def Q18():
    for i in range(100):
        newX, newY=Boostrap(len(X), X, Y)
        rf=DT(1)
        rf.fit(newX, newY)
        print(rf.score(X, Y))
    
Q17_Q18()
