import numpy as np
import random as rnd
import math



class Scorer:
    def __init__(self, inputCount, outputCount, r):
        self.inputCount=inputCount
        self.outputCount=outputCount
        self.W=(np.random.rand(inputCount+1, outputCount)*2-1)*r

    def updateW(self, addW):
        self.W=self.W+addW.A

    def getWij(self, i, j):
        return self.W[i][j]
    
    def getScores(self):
        return self.scores

    def addBias(self, X):
        newX=np.array(X).tolist()
        newX.insert(0, 1)
        return newX

    def score(self, X):
        assert len(X)==self.inputCount
        self.scores=(np.matrix(self.addBias(X))*np.matrix(self.W)).A1
        return self.scores

class Layer:
    def __init__(self, inputCount, neuralsCount, transform, r):
        self.neuralsCount=neuralsCount
        self.inputCount=inputCount
        self.scorer=Scorer(inputCount, neuralsCount, r)
        self.transform=transform
        self.deltas=[0]*neuralsCount

    def getX(self):
        return self.output

    def forward(self, X):
        S=self.scorer.score(X)
        self.output=self.transform(S)
        return self.output

    def getScores(self):
        return self.scorer.getScores()

    def updateW(self, addW):
        self.scorer.updateW(addW)

    def getW(self):
        return self.scorer.W

    def setDeltas(self, deltas):
        self.deltas=deltas

    def getDeltas(self):
        return self.deltas

def transform(S, diff=False):
    if diff:
        return 1-np.tanh(S)**2
    return np.tanh(S)

class NeuralNetwork():
    def __init__(self, eta):
        self.layers=[]
        self.eta=eta

    def addLayer(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        newX=X
        for layer in self.layers:
            newX=layer.forward(newX)
        return newX

    def score(self, X, Y):
        correctCount=0
        for i, x in enumerate(X):
            if(math.copysign(1, self.predict(x)[0])==Y[i][0]):
                correctCount+=1
        return correctCount/len(X)
    
    def squareErr(self, X, Y):
        sum=0
        for i, x in enumerate(X):
            predictY=self.predict(x)
            sum+=(predictY[0]-Y[i][0])**2
        return sum

    def fit(self, X, Y, iterationCount):
        length=len(X)
        for t in range(iterationCount):
            index=rnd.randint(0, length-1)
            #前馈
            self.predict(X[index])
            #反向传播
            self.backprop(Y[index])
            input=[e for e in X[index]]
            for layer in self.layers:
                input.insert(0, 1)
                addW=np.matrix(input).T*np.matrix(layer.getDeltas())
                layer.updateW(-self.eta*addW)
                input=[e for e in layer.getX()]
            #print(self.squareErr(X, Y))

    def backprop(self, y):
        layerCount=len(self.layers)
        layers=self.layers

        lastDeltas=[]
        scores=layers[layerCount-1].getScores()
        for i, yi in enumerate(y):
            delta=-2*(yi-layers[layerCount-1].transform(scores[i]))* \
                layers[layerCount-1].transform(scores[i], True)
            lastDeltas.append(delta)
        #print("minus:" ,yi-layers[layerCount-1].transform(scores[i]))
        #print("Diff:", layers[layerCount-1].transform(scores[0], True), "Score:", scores[i])
        layers[layerCount-1].setDeltas(lastDeltas)
        
        for layerIndex in range(layerCount-2, -1, -1):
            nextLayer=layers[layerIndex+1]
            deltas=(np.matrix(nextLayer.getW())* \
                np.matrix(nextLayer.getDeltas()).T).A1
            diffTransforms=layers[layerIndex].transform(layers[layerIndex].getScores(), True)
            deltas=deltas[:-1]*diffTransforms
            layers[layerIndex].setDeltas(deltas)

def LoadData(fileName):
    X, Y=[], []
    fp=open(fileName)
    for line in fp:
        line=line.strip()
        if line!="":
            arr=line.split(" ")
            X.append([float(e) for e in arr[:-1]])
            Y.append([float(arr[-1])])
    fp.close()
    return X, Y

def trainAndAverageScore(M, eta, r):
    sum=0
    for t in range(10):
            NN=NeuralNetwork(eta)
            NN.addLayer(Layer(2, M, transform, r))
            NN.addLayer(Layer(M, 1, transform, r))
            NN.fit(X, Y, T)
            sum+=NN.score(testX, testY)
    return sum/10

def Q11():
    for M in [1, 6, 11, 16, 21]:
        print("M:", M, "averageScore:", trainAndAverageScore(M, 0.1, 0.1))

def Q12():
    M=3
    for r in [0, 0.001, 0.1, 10, 1000]:
        print("r:", r, "averageScore:", trainAndAverageScore(3, 0.1, r))

def Q13():
    for eta in [0.001, 0.01, 0.1, 1, 10]:
        print("eta:", eta, "averageScore:", trainAndAverageScore(3, eta, 0.1))

def Q14():
    sum=0
    for t in range(10):
        r=0.1
        NN=NeuralNetwork(0.01)
        NN.addLayer(Layer(2, 8, transform, r))
        NN.addLayer(Layer(8, 3, transform, r))
        NN.addLayer(Layer(3, 1, transform, r))
        NN.fit(X, Y, T)
        sum+=1-NN.score(testX, testY)
    print(sum/10)

X, Y=LoadData("hw4_nnet_train.dat")
testX, testY=LoadData("hw4_nnet_test.dat")
T=20000

Q14()

