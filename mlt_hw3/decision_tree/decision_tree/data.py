def LoadData(fileName):
    X=[]
    Y=[]
    fp=open(fileName)
    for line in fp:
        line=line.strip()
        if line!="":
            arr=line.split(" ")
            X.append([float(e) for e in arr[:-1] if e])
            Y.append(float(arr[-1]))
    fp.close()
    return X, Y
