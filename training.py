#IMPORT RELEVANT MODULES 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import preprocessing
from sklearn import model_selection
from keras.models import Model
from keras.utils import np_utils, plot_model
from keras.layers import Flatten, Dropout, Activation, BatchNormalization, Concatenate
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras import optimizers
from keras import callbacks 
from keras.callbacks import ModelCheckpoint
import keras.backend as K
#-------------------------------------------------------------------------------------------------------------------
#FUNCTIONS

def initialiseWithSeed(num):
    seed = num
    initialisation = np.random.seed(seed)
    return initialisation


def getInputShape(rows=192,columns=11,channels=2):
    inputShape = (rows,columns,channels)
    return inputShape


def splitData(XFull,yFull,valSetSize,outputToStratify):    
    X_Train, X_Val, Y_Train, Y_Val = model_selection.train_test_split(XFull, yFull, 
                                                                      test_size=valSetSize, 
                                                                      stratify=None)
    print("Train-test split complete with "+str(valSetSize)+" validation set")
    return X_Train, X_Val, Y_Train, Y_Val


def encodeLabels(yTrain, yTest, numClasses):
    yTrainEncoded = np_utils.to_categorical(yTrain, numClasses)
    yTestEncoded = np_utils.to_categorical(yTest, numClasses) 
    return yTrainEncoded, yTestEncoded


def encodeLabelsSingle(y, numClasses):
    yEncoded = np_utils.to_categorical(y, numClasses)
    return yEncoded


def buildSingleMECNN(rows,columns,channels):
    inputShape = getInputShape(rows,columns,channels) 
    
    fH_l1 = 4
    fW_l1 = 3
    nC_l1 = 64
    sH_l1 = 1
    sW_l1 = 1
    
    fH_l2 = 4
    fW_l2 = 3
    nC_l2 = 64 
    sH_l2 = 1
    sW_l2 = 1
    
    fH_l2C = 1
    fW_l2C = 2
    sH_l2C = 1
    sW_l2C = 2
    
    fH_l2D = 2
    fW_l2D = 1
    sH_l2D = 2
    sW_l2D = 1
    
    fH_l3 = 4
    fW_l3 = 3
    nC_l3 = 128
    sH_l3 = 1
    sW_l3 = 1
    
    fH_l3C = 1
    fW_l3C = 2
    sH_l3C = 1
    sW_l3C = 2
    
    fH_l3D = 2
    fW_l3D = 1
    sH_l3D = 2
    sW_l3D = 1
    
    pDropout_l4 = 0.25
    fH_l4 = 12
    fW_l4 = 3
    nC_l4 = 256
    sH_l4 = 1
    sW_l4 = 1
    
    fH_l4C = 1
    fW_l4C = 2
    sH_l4C = 1
    sW_l4C = 2
    
    pDropout_l5 = 0.50
    hiddenUnits_l5 = 392
    
    pDropout_l6 = 0.50
    C = 49
    
    input_l0 = Input(shape=inputShape)
    
    Conv_l1 = Conv2D(nC_l1, kernel_size=(fH_l1,fW_l1), strides=(sH_l1,sW_l1), 
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(input_l0)
    
    BatchNorm_l1B = BatchNormalization(axis=-1)(Conv_l1)
    
    Conv_l2 = Conv2D(nC_l2, kernel_size=(fH_l2,fW_l2), strides=(sH_l2,sW_l2), 
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(BatchNorm_l1B)
    
    BatchNorm_l2B = BatchNormalization(axis=-1)(Conv_l2)
    
    AP_l2C = AveragePooling2D(pool_size=(fH_l2C,fW_l2C), strides=(sH_l2C,sW_l2C), 
                              padding='valid')(BatchNorm_l2B)
    MP_l2D = MaxPooling2D(pool_size=(fH_l2D,fW_l2D), strides=(sH_l2D,sW_l2D), 
                          padding='valid')(AP_l2C)

    Conv_l3 = Conv2D(nC_l3, kernel_size=(fH_l3,fW_l3), strides=(sH_l3,sW_l3), 
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(MP_l2D)
    
    BatchNorm_l3B = BatchNormalization(axis=-1)(Conv_l3)
    
    AP_l3C = AveragePooling2D(pool_size=(fH_l3C,fW_l3C), strides=(sH_l3C,sW_l3C), 
                              padding='valid')(BatchNorm_l3B)
    
    MP_l3D = MaxPooling2D(pool_size=(fH_l3D,fW_l3D), strides=(sH_l3D,sW_l3D), 
                          padding='valid')(AP_l3C)
    
    Dropout_l4A = Dropout(pDropout_l4)(MP_l3D)
    
    Conv_l4 = Conv2D(nC_l4, kernel_size=(fH_l4,fW_l4), strides=(sH_l4,sW_l4), 
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(Dropout_l4A)
    
    BatchNorm_l4B = BatchNormalization(axis=-1)(Conv_l4)
    
    AP_l4C = AveragePooling2D(pool_size=(fH_l4C,fW_l4C), strides=(sH_l4C,sW_l4C), 
                              padding='valid')(BatchNorm_l4B)
    
    Flatten_l5A = Flatten()(AP_l4C)
    Dropout_l5B = Dropout(pDropout_l5)(Flatten_l5A)
    FC_l5 = Dense(hiddenUnits_l5, activation=None,
                   kernel_initializer='glorot_uniform', bias_initializer='zeros')(Dropout_l5B) 
    
    Dropout_l6A = Dropout(pDropout_l6)(FC_l5)
    FC_l6 = Dense(C, activation='softmax',
                   kernel_initializer='glorot_uniform', bias_initializer='zeros')(Dropout_l6A)
    
    modelBuilt = Model(inputs=[input_l0], outputs=[FC_l6])
    
    print(modelBuilt.summary())
    return modelBuilt


def setHyperparamsPolyMECNN():
    E = 25
    batchSize = 64
    learningRate = 0.001
    learningDecay = 1e-06
    beta = 0.90
    optimiser = optimizers.SGD(lr=learningRate, decay=learningDecay,
                                momentum=beta, nesterov=True)
    """
    beta1 = 0.90
    beta2 = 0.999
    optimiser = optimizers.Adam(lr=learningRate, decay=learningDecay,
                                beta_1=beta1, beta_2=beta2)
    """
    lossFunction = 'categorical_crossentropy'
    return E, batchSize, optimiser, lossFunction


def setHyperparamsMonoMECNN():
    E = 50
    batchSize = 64
    learningRate = 0.001
    learningDecay = 1e-06
    momentum = 0.90
    optimiser = optimizers.SGD(lr=learningRate, decay=learningDecay,
                               momentum=momentum, nesterov=True)
    """
    beta1 = 0.90
    beta2 = 0.999
    optimiser = optimizers.Adam(lr=learningRate, decay=learningDecay,
                                beta_1=beta1, beta_2=beta2)
    """
    lossFunction = 'categorical_crossentropy'
    return E, batchSize, optimiser, lossFunction


def buildMTMECNN(rows,columns,channels):
    inputShape = getInputShape(rows,columns,channels) 
    
    fH_l1 = 4
    fW_l1 = 3
    nC_l1 = 64
    sH_l1 = 1
    sW_l1 = 1
    
    fH_l2 = 4
    fW_l2 = 3
    nC_l2 = 64 
    sH_l2 = 1
    sW_l2 = 1
    
    fH_l2C = 1
    fW_l2C = 2
    sH_l2C = 1
    sW_l2C = 2
    
    fH_l2D = 2
    fW_l2D = 1
    sH_l2D = 2
    sW_l2D = 1
    
    fH_l3 = 4
    fW_l3 = 3
    nC_l3 = 128
    sH_l3 = 1
    sW_l3 = 1
    
    fH_l3C = 1
    fW_l3C = 2
    sH_l3C = 1
    sW_l3C = 2
    
    fH_l3D = 2
    fW_l3D = 1
    sH_l3D = 2
    sW_l3D = 1
    
    pDropout_l4 = 0.25
    fH_l4 = 12
    fW_l4 = 3
    nC_l4 = 256
    sH_l4 = 1
    sW_l4 = 1
    
    fH_l4C = 1
    fW_l4C = 2
    sH_l4C = 1
    sW_l4C = 2
    
    pDropout_l5 = 0.50
    hiddenUnits_l5 = 392
    
    pDropout_l6 = 0.50
    C = 49
    
    input_l0_Mono = Input(shape=inputShape)
    input_l0_Poly = Input(shape=inputShape)
    
    Conv_l1_Mono = Conv2D(nC_l1, kernel_size=(fH_l1,fW_l1), strides=(sH_l1,sW_l1), 
                          padding='same', 
                          activation='relu')(input_l0_Mono)
    Conv_l1_Poly = Conv2D(nC_l1, kernel_size=(fH_l1,fW_l1), strides=(sH_l1,sW_l1), 
                          padding='same', 
                          activation='relu')(input_l0_Poly)
    
    BatchNorm_l1B_Mono = BatchNormalization(axis=-1)(Conv_l1_Mono)
    BatchNorm_l1B_Poly = BatchNormalization(axis=-1)(Conv_l1_Poly)
    
    Conv_l2_Mono = Conv2D(nC_l2, kernel_size=(fH_l2,fW_l2), strides=(sH_l2,sW_l2), 
                          padding='same', activation='relu')(BatchNorm_l1B_Mono)
    Conv_l2_Poly = Conv2D(nC_l2, kernel_size=(fH_l2,fW_l2), strides=(sH_l2,sW_l2), 
                          padding='same', activation='relu')(BatchNorm_l1B_Poly)
    
    BatchNorm_l2B_Mono = BatchNormalization(axis=-1)(Conv_l2_Mono)
    BatchNorm_l2B_Poly = BatchNormalization(axis=-1)(Conv_l2_Poly)
    
    AP_l2C_Mono = AveragePooling2D(pool_size=(fH_l2C,fW_l2C), strides=(sH_l2C,sW_l2C), 
                                   padding='valid')(BatchNorm_l2B_Mono)
    AP_l2C_Poly = AveragePooling2D(pool_size=(fH_l2C,fW_l2C), strides=(sH_l2C,sW_l2C), 
                                   padding='valid')(BatchNorm_l2B_Poly)
    
    MP_l2D_Mono = MaxPooling2D(pool_size=(fH_l2D,fW_l2D), strides=(sH_l2D,sW_l2D), 
                               padding='valid')(AP_l2C_Mono)
    MP_l2D_Poly = MaxPooling2D(pool_size=(fH_l2D,fW_l2D), strides=(sH_l2D,sW_l2D), 
                               padding='valid')(AP_l2C_Poly)
    
    Conv_l3_Mono = Conv2D(nC_l3, kernel_size=(fH_l3,fW_l3), strides=(sH_l3,sW_l3), 
                          padding='same', activation='relu')(MP_l2D_Mono)
    Conv_l3_Poly = Conv2D(nC_l3, kernel_size=(fH_l3,fW_l3), strides=(sH_l3,sW_l3), 
                          padding='same', activation='relu')(MP_l2D_Poly)
    
    BatchNorm_l3B_Mono = BatchNormalization(axis=-1)(Conv_l3_Mono)
    BatchNorm_l3B_Poly = BatchNormalization(axis=-1)(Conv_l3_Poly)
    
    AP_l3C_Mono = AveragePooling2D(pool_size=(fH_l3C,fW_l3C), strides=(sH_l3C,sW_l3C), 
                                   padding='valid')(BatchNorm_l3B_Mono)
    AP_l3C_Poly = AveragePooling2D(pool_size=(fH_l3C,fW_l3C), strides=(sH_l3C,sW_l3C), 
                                   padding='valid')(BatchNorm_l3B_Poly)
    
    MP_l3D_Mono = MaxPooling2D(pool_size=(fH_l3D,fW_l3D), strides=(sH_l3D,sW_l3D), 
                               padding='valid')(AP_l3C_Mono)
    MP_l3D_Poly = MaxPooling2D(pool_size=(fH_l3D,fW_l3D), strides=(sH_l3D,sW_l3D), 
                               padding='valid')(AP_l3C_Poly)
    
    Dropout_l4A_Mono = Dropout(pDropout_l4)(MP_l3D_Mono)
    Dropout_l4A_Poly = Dropout(pDropout_l4)(MP_l3D_Poly)
    
    Conv_l4_Mono = Conv2D(nC_l4, kernel_size=(fH_l4,fW_l4), strides=(sH_l4,sW_l4), 
                          padding='same', activation='relu')(Dropout_l4A_Mono)
    Conv_l4_Poly = Conv2D(nC_l4, kernel_size=(fH_l4,fW_l4), strides=(sH_l4,sW_l4), 
                          padding='same', activation='relu')(Dropout_l4A_Poly)
    
    BatchNorm_l4B_Mono = BatchNormalization(axis=-1)(Conv_l4_Mono)
    BatchNorm_l4B_Poly = BatchNormalization(axis=-1)(Conv_l4_Poly)
    
    AP_l4C_Mono = AveragePooling2D(pool_size=(fH_l4C,fW_l4C), strides=(sH_l4C,sW_l4C), 
                                   padding='valid')(BatchNorm_l4B_Mono)
    AP_l4C_Poly = AveragePooling2D(pool_size=(fH_l4C,fW_l4C), strides=(sH_l4C,sW_l4C), 
                                   padding='valid')(BatchNorm_l4B_Poly)
    
    Flatten_l5A_Mono = Flatten()(AP_l4C_Mono)
    Flatten_l5A_Poly = Flatten()(AP_l4C_Poly)
    
    Dropout_l5B_Mono = Dropout(pDropout_l5)(Flatten_l5A_Mono)
    Dropout_l5B_Poly = Dropout(pDropout_l5)(Flatten_l5A_Poly)
    
    FC_l5_Mono = Dense(hiddenUnits_l5, activation=None)(Dropout_l5B_Mono) 
    FC_l5_Poly = Dense(hiddenUnits_l5, activation=None)(Dropout_l5B_Poly) 
    
    Dropout_l6A_Mono = Dropout(pDropout_l6)(FC_l5_Mono)
    Dropout_l6A_Poly = Dropout(pDropout_l6)(FC_l5_Poly)
        
    output_l6_Mono = Dense(C, activation='softmax', name='MONOME')(Dropout_l6A_Mono)
    output_l6_Poly = Dense(C, activation='softmax', name='POLYME')(Dropout_l6A_Poly)
    
    modelBuilt = Model(inputs=[input_l0_Poly, input_l0_Mono], 
                       outputs=[output_l6_Poly, output_l6_Mono])
    
    print(modelBuilt.summary())
    return modelBuilt


def getLossWeights(numModel):
    if numModel==1:
        gammaPoly = 0.5
    elif numModel==2:
        gammaPoly = 0.6
    elif numModel==3:
        gammaPoly = 0.7
    elif numModel==4:
        gammaPoly = 0.8
    elif numModel==5:
        gammaPoly = 0.9
    else:
        gammaPoly = 0.5
    gammaMono = 1.0 - gammaPoly    
    lossWeights = [gammaPoly, gammaMono]
    return lossWeights


def setHyperparamsMTMECNN():
    E = 75
    batchSize = 64
    learningRate = 0.001
    learningDecay = 1e-06
    momentum = 0.90
    optimiser = optimizers.SGD(lr=learningRate, decay=learningDecay,
                               momentum=momentum, nesterov=True)
    """
    beta1 = 0.90
    beta2 = 0.999
    optimiser = optimizers.Adam(lr=learningRate, decay=learningDecay,
                                beta_1=beta1, beta_2=beta2)
    """
    lossFunction = ['categorical_crossentropy', 'categorical_crossentropy']
    return E, batchSize, optimiser, lossFunction


def compileMultiTaskModel(modelBuilt, lossFunction, lossWeights, optimiser):
    modelBuilt.compile(loss=lossFunction, 
                       loss_weights=lossWeights,
                       optimizer=optimiser, 
                       metrics=['accuracy'])
    return modelBuilt


def trainMultiTaskModel(modelCompiled, 
                        trainingX1, trainingY1, trainingX2, trainingY2,
                        E, batchSize, nameModel, numModel, pathSave, 
                        vsplitfactor=0.1):
    filenameModel = nameModel+'_'+str(numModel)+'.h5'
    checkpointer = ModelCheckpoint(filepath=pathSave+'/'+filenameModel, 
                                   monitor='val_loss', verbose=2, 
                                   save_best_only=True)
    modelHistory = modelCompiled.fit([trainingX1, trainingX2], 
                                     [trainingY1, trainingY2], 
                                     batch_size=batchSize, epochs=E, 
                                     verbose=2, validation_split=vsplitfactor,
                                     callbacks = [checkpointer])
    print('Model has been trained and will now be evaluated on test set')
    return modelCompiled, modelHistory


def plotMultiTaskModelHistory(modelHistory, nameModel, numModel, E):
    print(modelHistory.history.keys())
    plt.figure(1, figsize=(22, 11))
    plt.rc('axes', axisbelow=True)
    plt.plot(modelHistory.history['POLYME_acc'])
    plt.plot(modelHistory.history['val_POLYME_acc'])
    plt.plot(modelHistory.history['MONOME_acc'])
    plt.plot(modelHistory.history['val_MONOME_acc'])
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 16)
    plt.grid(axis='both', zorder=0)
    plt.xticks(np.arange(0, E+1, 1), fontsize = 12)
    plt.yticks(fontsize=15)
    plt.legend(['Training Polyphonic','Validation Polyphonic', 'Training Monophonic', 'Validation Monophonic'], loc='upper left', fontsize=16)
    #plt.title('Accuracy of '+nameModel+' over '+str(E)+' training epochs')
    plt.savefig(nameModel+'_'+str(numModel)+'_SingleTaskAccuracy'+'.png')
    plt.figure(2, figsize=(22, 11))
    plt.rc('axes', axisbelow=True)
    plt.plot(modelHistory.history['POLYME_loss'])
    plt.plot(modelHistory.history['val_POLYME_loss'])
    plt.plot(modelHistory.history['MONOME_loss'])
    plt.plot(modelHistory.history['val_MONOME_loss'])
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.grid(axis='both', zorder=0)
    plt.xticks(np.arange(0, E+1, 1), fontsize = 12)
    plt.yticks(fontsize=15)
    plt.legend(['Training Polyphonic','Validation Polyphonic', 'Training Monophonic', 'Validation Monophonic'], loc='upper left', fontsize=16)
    #plt.title('Loss of '+nameModel+' over '+str(E)+' training epochs')
    plt.savefig(nameModel+'_'+str(numModel)+'_SingleTaskLoss'+'.png')
    plt.figure(3, figsize=(22, 11))
    plt.rc('axes', axisbelow=True)
    plt.plot(modelHistory.history['loss'])
    plt.plot(modelHistory.history['val_loss'])
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.grid(axis='both', zorder=0)
    plt.xticks(np.arange(0, E+1, 1), fontsize = 12)
    plt.yticks(fontsize=15)
    plt.legend(['Training Multi-Task','Validation Multi-Task'], loc='upper left', fontsize=16)
    #plt.title('Accuracy of '+nameModel+' over '+str(E)+' training epochs')
    plt.savefig(nameModel+'_'+str(numModel)+'_MultiTaskLoss'+'.png')
    
    
def evaluateMultiTaskModel(modelTrained, batchSize, XTest, yTest):
    allMetrics = modelTrained.metrics_names
    scores = modelTrained.evaluate([XTest, XTest], [yTest, yTest], batch_size=batchSize, verbose=2)
    OverallLoss = scores[0]
    TargetLoss = scores[1]
    TargetAcc = scores[3]
    AuxLoss = scores[2]
    AuxAcc = scores[4]
    print("Test Loss for Target Task is " + str(TargetLoss))
    print("Test Accuracy for Target Task is " + str(TargetAcc))
    print("Test Loss for Auxiliary Task is " + str(AuxLoss))
    print("Test Accuracy for Auxiliary Task is " + str(AuxAcc))
    print("Overall Test Loss is " + str(OverallLoss))
    return allMetrics, scores, TargetLoss, TargetAcc


def saveModelDiagram(model, nameModel):
    plot_model(model, to_file=nameModel+'.png')
    print("Saved Model Graph")


def compileModel(modelBuilt, lossFunction, optimiser):
    modelBuilt.compile(loss=lossFunction, 
                       optimizer=optimiser, 
                       metrics=['accuracy'])
    return modelBuilt


def trainModel(modelCompiled, trainingX, trainingY, 
               E, batchSize, nameModel, numModel, pathSave, 
               vsplit=True, vsplitfactor=0.1, 
               XVal=None, yVal=None):
    filenameModel = nameModel+'_'+str(numModel)+'.h5'
    checkpointer = ModelCheckpoint(filepath=pathSave+'/'+filenameModel, 
                                   monitor='val_loss', verbose=2, 
                                   save_best_only=True)
    if vsplit==True:
        modelHistory = modelCompiled.fit(trainingX, trainingY, 
                                  batch_size=batchSize, epochs=E, 
                                  verbose=2, validation_split=vsplitfactor,
                                  callbacks = [checkpointer])
    elif vsplit==False:
        modelHistory = modelCompiled.fit(trainingX, trainingY, 
                                  batch_size=batchSize, epochs=E, 
                                  verbose=2, validation_data=(XVal,yVal),
                                  callbacks = [checkpointer])
    print('Model has been trained and will now be evaluated on test set')
    return modelCompiled, modelHistory


def saveModel(modelTrained, nameModel, numModel, savepath):
    filenameModel = nameModel+'_'+str(numModel)+'.h5'
    print('Saving model under filename '+filenameModel)
    modelTrained.save(savepath+ '/' + filenameModel)
    print('Saved model under filename '+filenameModel)
  
    
def loadpretrainedmodel(modelfilename):
    loadedModel = load_model(modelfilename)
    print("Model loaded")
    return loadedModel


def plotModelHistory(modelHistory, nameModel, numModel, E):
    print(modelHistory.history.keys())
    plt.figure(1, figsize=(16, 8))
    plt.rc('axes', axisbelow=True)
    plt.plot(modelHistory.history['acc'])
    plt.plot(modelHistory.history['val_acc'])
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.grid(axis='both',zorder=0)
    plt.xticks(np.arange(0, E+1, 1), fontsize = 12)
    plt.legend(['Training','Validation'], loc='upper left', fontsize=14)
    #plt.title('Accuracy of '+nameModel+' over '+str(E)+' training epochs')
    plt.savefig(nameModel+'_'+str(numModel)+'_Accuracy'+'.png')
    plt.figure(2, figsize=(16, 8))
    plt.rc('axes', axisbelow=True)
    plt.plot(modelHistory.history['loss'])
    plt.plot(modelHistory.history['val_loss'])
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Loss', fontsize = 14)
    plt.grid(axis='both',zorder=0)
    plt.xticks(np.arange(0, E+1, 1), fontsize = 12)
    plt.legend(['Training','Validation'], loc='upper left', fontsize=14)
    #plt.title('Loss of '+nameModel+' over '+str(E)+' training epochs')
    plt.savefig(nameModel+'_'+str(numModel)+'_Loss'+'.png')
    print("Model has been evaluated on test set")
    

def evaluateModel(modelTrained, batchSize, XTest, yTest):
    score = modelTrained.evaluate(XTest, yTest, batch_size=batchSize, verbose=2)
    testLoss = score[0]
    testAccuracy = score[1]
    print("Test Loss is " + str(testLoss))
    print("Test Accuracy is " + str(testAccuracy))
    return testLoss, testAccuracy
    

def getClassNames(y):
    f0Labels = np.unique(y)
    lowestMIDI = preprocessing.getMIDIfromNote('G3')
    classNamesTest = preprocessing.getNotesfromF0Labels(f0Labels, offset=lowestMIDI-1)
    return lowestMIDI, f0Labels, classNamesTest    


def getAllClassNames():
    classnamesAll = ['None', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 
                     'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 
                     'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5',
                     'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5',
                     'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6',
                     'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
                     'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7']
    return classnamesAll


def plotConfusionMatrixSingle(model, nameModel, numModel, 
                              XTest, YTest, classNames):
    YHat = model.predict(np.array(XTest))
    YHat_nonCategory = [np.argmax(t) for t in YHat]
    YTest_nonCategory = [np.argmax(t) for t in YTest] 
    cnf_matrix = confusion_matrix(YTest_nonCategory, YHat_nonCategory)
    cm = cnf_matrix.astype('float') / (cnf_matrix.astype('float').sum(axis=1))[:, np.newaxis]   
    plt.figure(figsize=(20, 20))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix')
    #plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation = 45, fontsize = 17)
    plt.yticks(tick_marks, classNames, fontsize = 17)
    fmt = '.2f'
    thresh = cm.max() * 0.85
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 17)

    plt.ylabel('Ground Truth f0 Label', fontsize = 17)
    plt.xlabel('Predicted f0 Label', fontsize = 17)
    plt.tight_layout()
    plt.savefig(nameModel+'_'+str(numModel)+'_ConfusionMatrix'+'.png')
    return cm
    

def plotConfusionMatrixMT(model, nameModel, numModel, 
                          XTest, YTest, classNames):
    YHat = model.predict(XTest)[0]
    YTestPoly = YTest[0]
    YHat_nonCategory = [np.argmax(t) for t in YHat]
    YTest_nonCategory = [np.argmax(t) for t in YTestPoly] 
    cnf_matrix = confusion_matrix(YTest_nonCategory, YHat_nonCategory)
    cm = cnf_matrix.astype('float') / (cnf_matrix.astype('float').sum(axis=1))[:, np.newaxis]   
    plt.figure(figsize=(20, 20))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix')
    #plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation = 45, fontsize = 17)
    plt.yticks(tick_marks, classNames, fontsize = 17)
    fmt = '.2f'
    thresh = cm.max() * 0.85
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 17)

    plt.ylabel('Ground Truth f0 Label', fontsize = 17)
    plt.xlabel('Predicted f0 Label', fontsize = 17)
    plt.tight_layout()
    plt.savefig(nameModel+'_'+str(numModel)+'_ConfusionMatrix'+'.png')
    return cm
    

