#IMPORT RELEVANT MODULES
import numpy as np
import preprocessing
#----------------------------------------------------------------------------------
#FUNCTIONS

def getSoftmaxProbabilities(model, X, isMT):
    if isMT == False:
        ZAll = model.predict(X)
    if isMT == True:
        ZAll = model.predict([X, X])[0]
    return ZAll
        

def getSoftmaxVector(softmaxprobabilities, t):
    ZTranspose = softmaxprobabilities[t,:]
    Z = np.transpose(ZTranspose)
    return Z
    

def getNotePriors(f0labelStream):
    numElements = float(len(f0labelStream))
    countVector = np.bincount(f0labelStream)
    C = countVector / numElements
    return C


def buildf0labelStream(path, pieces):
    f0labelStream = preprocessing.loadLabelArray(path, 'yTrain_ME_'+pieces[0]+'.npy')
    for i in range(1,len(pieces),1):
        selPiece = pieces[i]
        loadedf0labelStream = preprocessing.loadLabelArray(path,'yTrain_ME_'+selPiece+'.npy')
        f0labelStream = np.concatenate((f0labelStream, loadedf0labelStream), axis=0)  
      
    return f0labelStream


def getTransitionMatrix(y, K):   
    yList = y.tolist()
    A = [[0]*K for _ in range(K)]
    for (i,j) in zip(yList,yList[1:]):
        A[i][j] += 1.0
    
    for row in A:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    
    A = np.asarray(A)
    ANorm = np.empty_like(A)
    for i in range(0, K, 1):
        diagElements = []
        i1 = i
        j1 = 0
        while i1 < 49:
            diagElements.append(A[i1,j1])
            i1 += 1
            j1 += 1
                
        diagMean = np.mean(diagElements)
        i2 = i
        j2 = 0
        while i2 < 49:
            ANorm[i2,j2] = diagMean
            i2 += 1
            j2 += 1
    
    for j in range(1, K, 1):
        diagElements = []
        i1 = 0
        j1 = j
        while j1 < 49:
            diagElements.append(A[i1,j1])
            i1 += 1
            j1 += 1
                
        diagMean = np.mean(diagElements)
        i2 = 0
        j2 = j
        while j2 < 49:
            ANorm[i2,j2] = diagMean
            i2 += 1
            j2 += 1
    
    sumANorm = np.sum(ANorm, axis = 1)
    for row in range(0, K, 1):
        ANorm[row,:] = np.divide(ANorm[row,:], sumANorm[row])
        
    return A, ANorm


def getsmoothedf0Traj(f0LabelsTrain, f0LabelsEx, model, isMT, X, numClasses):
    A, ANorm = getTransitionMatrix(f0LabelsTrain, numClasses)
    C = getNotePriors(f0LabelsTrain)
    ZAll = getSoftmaxProbabilities(model, X, isMT)
    newLabels = np.empty_like(f0LabelsEx)
    for t in range(len(newLabels)):
        Z = getSoftmaxVector(ZAll, t)
        posteriorMatrix = np.divide(Z, C)
        newZ = np.dot(ANorm, posteriorMatrix)  
        sumZ = np.sum(newZ)
        newZNorm = np.divide(newZ, sumZ) 
        newLabels[t] = newZNorm.argmax(axis=-1)
        
    return newLabels


def checkSimilarity(yRaw, ySmooth):
    checks = []
    for i in range(len(yRaw)):
        if yRaw[i] == ySmooth[i]:
            checks.append('Same')
        else:
            checks.append('Different')
    
    degreeSim = float(float(checks.count('Same')) / float((checks.count('Same')+checks.count('Different'))))
    return checks, degreeSim


