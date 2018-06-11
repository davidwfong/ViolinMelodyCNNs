#IMPORT ALL REQUIRED MODULES
import chooseRepresentation
import preprocessing
import training
import postprocessing
import evaluation
import predicting
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#DEFINE PATHS (MAINTAIN FOLDER PATH)
path_proj = '/home/davidwfong/SpyderProjects/Violin_Melody_Extraction'
path_data = path_proj + '/data'

path_ChooseFeatures = path_data + '/choose_Features'
path_violin = path_ChooseFeatures + '/violin_file.wav'

path_ME1 = path_data + '/ME1'
path_train_ME1 = path_ME1 + '/TRAINING'
path_trainAugmented_ME1 = path_ME1 + '/TRAINING_AUGMENTED'
path_test_ME1 = path_ME1 + '/TEST'

path_ME2 = path_data + '/ME2'
path_train_ME2 = path_ME2 + '/TRAINING'
path_trainAugmented_ME2 = path_ME2 + '/TRAINING_AUGMENTED'
path_test_ME2 = path_ME2 + '/TEST'

path_MTME = path_data + '/MTME'

#%%
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#HARDCODE PARAMETERS
fs=16000
hopSize_Sec=0.010
noteMin='G3'
noteMax='F#7'
H=2
K=192
B=48
T=11
numClasses = 49

#-------------------------------------------------------------------------------
#%%
#CHOOSE INPUT REPRESENTATION

#Step 1: Create solo violin audio file
path_violin = chooseRepresentation.concatenateAudio(path_ChooseFeatures)
data, fs = chooseRepresentation.convertWavToArray(path_violin)
#Step 2: Plot Spectrograms of solo violin 
SNorm_Mel, SNorm_STFT, SNorm_CQT, SNorm_HCQT = chooseRepresentation.plotSpectrograms(data,fs)
#Step 3:plot HCQT and Input Feature of polyphonic audio file
path_exampleData = path_test_ME1 + '/VS01.wav'
exampleData,exampleFs = chooseRepresentation.convertWavToArrayNew(path_exampleData)
SFinal_HCQT = chooseRepresentation.plotHCQT(exampleData) 
inputFeatureExample = chooseRepresentation.plotInputFeature(exampleData)

#%%
#---------------------------------------------------------------------------------
#PREPROCESSING FOR POLYPHONIC MELODY EXTRACTOR

#Step 1: Clean Ground Truths
preprocessing.cleanTestGTME(path_test_ME1)
preprocessing.cleanTrainGTME(path_train_ME1)

#Step 2: Downmix to mono and Downsample to 16kHz
preprocessing.downmixAudio(path_test_ME1)
preprocessing.downmixAudio(path_train_ME1)
preprocessing.downsampleAudio(path_test_ME1,fs)
preprocessing.downsampleAudio(path_train_ME1,fs)

#Step 3: Augment training data by pitch shifting up by up to 3 semitones
preprocessing.augmentDataME(path_train_ME1,path_trainAugmented_ME1,3)

#Step 4: Extract Input Features & Labels for Test Set
XTest, yTest = preprocessing.extractFeaturesME(path_test_ME1, T, fs, hopSize_Sec, noteMin, H, K, B)
preprocessing.saveInputOutputArrays(XTest, yTest, 'XTest_ME.npy', 'yTest_ME.npy', path_ME1)  

#Step 5: Get list of filenames in training dataset
pieces = preprocessing.listAudioFilenames(path_train_ME1)

#Step 6: Extract Input Features & Labels for each piece in Training Set (do one by one, changing i from 0 to len(pieces)-1)
selPiece = pieces[i]
XTrain, yTrain = preprocessing.extractFeaturesME(path_trainAugmented_ME1+'/'+selPiece, T, fs, hopSize_Sec, noteMin, H, K, B)
preprocessing.saveInputOutputArrays(XTrain, yTrain, 'XTrain_ME_'+selPiece+'.npy', 'yTrain_ME_'+selPiece+'.npy', path_ME1)

#Step 7: Subsample training data by factor 10.00 to get appropriately sized data for RAM (do one by one, changing i from 0 to len(pieces)-1)
selPiece = pieces[i]
XFull, yFull = preprocessing.loadArrays(path_ME1, 'XTrain_ME_'+selPiece+'.npy', 'yTrain_ME_'+selPiece+'.npy')
XSampled, ySampled = preprocessing.pickSubsetfromData(XFull, yFull, reductionFactor=10.00)
preprocessing.saveInputOutputArrays(XSampled, ySampled, 'XTrain_ME_'+selPiece+'subset'+'.npy', 'yTrain_ME_'+selPiece+'subset'+'.npy', path_ME1)

#Step 8: Concatenate subsampled data to form final training set
XProv, yProv = preprocessing.loadArrays(path_ME1, 'XTrain_ME_'+pieces[0]+'subset'+'.npy','yTrain_ME_'+pieces[0]+'subset'+'.npy')
for i in range(1,len(pieces),1):
    selPiece = pieces[i]
    XLoaded, yLoaded = preprocessing.loadArrays(path_ME1, 'XTrain_ME_'+selPiece+'subset'+'.npy','yTrain_ME_'+selPiece+'subset'+'.npy')
    XProv = np.concatenate((XProv, XLoaded), axis=0)
    yProv = np.concatenate((yProv, yLoaded), axis=0)

preprocessing.saveInputOutputArrays(XProv, yProv, 'XTrain_ME'+'.npy', 'yTrain_ME'+'.npy', path_ME1)

#Step 9: Check that all label data is within required range
y_TrainMECNN = preprocessing.loadLabelArray(path_ME1, 'yTrain_ME.npy')
counter=0
for el in y_TrainMECNN:
    if el > 48:
        print("NOTE TOO HIGH AT "+str(el))
    counter = counter+1
    
#Step 10: plot distribution of f0 labels for full and final training set
ySubset = preprocessing.loadLabelArray(path_ME1,'yTrain_ME'+'.npy')
yProv = preprocessing.loadLabelArray(path_ME1,'yTrain_ME_'+pieces[0]+'.npy')
for i in range(1,len(pieces),1):
    selPiece = pieces[i]
    yLoaded = preprocessing.loadLabelArray(path_ME1,'yTrain_ME_'+selPiece+'.npy')
    yProv = np.concatenate((yProv, yLoaded), axis=0)

preprocessing.plotLabelDistributions(yProv,ySubset,'poly')

#%%
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#PREPROCESSING FOR MONOPHONIC MELODY EXTRACTOR

#Step 1: Clean Ground Truths
preprocessing.cleanTestGTME(path_test_ME2)
preprocessing.cleanTrainGTME(path_train_ME2)

#Step 2: Downmix to mono and Downsample to 16kHz
preprocessing.downmixAudio(path_test_ME2)
preprocessing.downmixAudio(path_train_ME2)
preprocessing.downsampleAudio(path_test_ME2,fs)
preprocessing.downsampleAudio(path_train_ME2,fs)

#Step 3: Augment training data by pitch shifting up by up to 3 semitones
preprocessing.augmentDataME(path_train_ME2,path_trainAugmented_ME2,3)

#Step 4: Get list of filenames in training dataset
pieces = preprocessing.listAudioFilenames(path_train_ME2)

#Step 5: Extract Input Features & Labels for each piece in Training Set (do one by one, changing i from 0 to len(pieces)-1)
selPiece = pieces[i]
XTrain, yTrain = preprocessing.extractFeaturesME(path_trainAugmented_ME2+'/'+selPiece, T, fs, hopSize_Sec, noteMin, H, K, B)
preprocessing.saveInputOutputArrays(XTrain, yTrain, 'XTrain_ME_'+selPiece+'.npy', 'yTrain_ME_'+selPiece+'.npy', path_ME2)

#Step 6: Subsample training data by factor 2.22 to get appropriately sized data for RAM (do one by one, changing i from 0 to len(pieces)-1)
selPiece = pieces[i]
XFull, yFull = preprocessing.loadArrays(path_ME2, 'XTrain_ME_'+selPiece+'.npy', 'yTrain_ME_'+selPiece+'.npy')
XSampled, ySampled = preprocessing.pickSubsetfromData(XFull, yFull, reductionFactor=2.22)
preprocessing.saveInputOutputArrays(XSampled, ySampled, 'XTrain_ME_'+selPiece+'subset'+'.npy', 'yTrain_ME_'+selPiece+'subset'+'.npy', path_ME2)

#Step 7: Concatenate subsampled data to form final training set
XProv, yProv = preprocessing.loadArrays(path_ME2, 'XTrain_ME_'+pieces[0]+'subset'+'.npy','yTrain_ME_'+pieces[0]+'subset'+'.npy')
for i in range(1,len(pieces),1):
    selPiece = pieces[i]
    XLoaded, yLoaded = preprocessing.loadArrays(path_ME2, 'XTrain_ME_'+selPiece+'subset'+'.npy','yTrain_ME_'+selPiece+'subset'+'.npy')
    XProv = np.concatenate((XProv, XLoaded), axis=0)
    yProv = np.concatenate((yProv, yLoaded), axis=0)

preprocessing.saveInputOutputArrays(XProv, yProv, 'XTrain_ME'+'.npy', 'yTrain_ME'+'.npy', path_ME2)
 
#Step 8: Check that all label data is within required range
y_TrainMECNN = preprocessing.loadLabelArray(path_ME2, 'yTrain_ME.npy')
counter=0
for el in y_TrainMECNN:
    if el > 48:
        print("NOTE TOO HIGH AT "+str(el))
    counter = counter+1
    
#Step 9: plot distribution of f0 labels for full and final training set
ySubset = preprocessing.loadLabelArray(path_ME2,'yTrain_ME'+'.npy')
yProv = preprocessing.loadLabelArray(path_ME2,'yTrain_ME_'+pieces[0]+'.npy')
for i in range(1,len(pieces),1):
    selPiece = pieces[i]
    yLoaded = preprocessing.loadLabelArray(path_ME2,'yTrain_ME_'+selPiece+'.npy')
    yProv = np.concatenate((yProv, yLoaded), axis=0)

preprocessing.plotLabelDistributions(yProv,ySubset,'mono')

#%%
#--------------------------------------------------------------------------------------------------------------------------------------------
#BUILD, TRAIN & EVALUATE POLYPHONIC MELODY EXTRACTION SINGLE CNN

#Step 1: Load Training and Test Set Arrays
XTrain_PolyMECNN, yTrain_PolyMECNN = preprocessing.loadArrays(path_ME1, 'XTrain_ME'+'.npy','yTrain_ME'+'.npy')
XTest_PolyMECNN, yTest_PolyMECNN = preprocessing.loadArrays(path_ME1, 'XTest_ME'+'.npy','yTest_ME'+'.npy')

#Step 2: One hot encode labels
yTrainOHE_PolyMECNN, yTestOHE_PolyMECNN = training.encodeLabels(yTrain_PolyMECNN,yTest_PolyMECNN,numClasses)

#Step 3: do random initialisation of network
seed = 100
initialisation = training.initialiseWithSeed(seed)

#Step 4: Build Model Architecture
PolyMECNN_Built = training.buildSingleMECNN(rows=K,columns=T,channels=H)

#Step 5: Set model Type and Number, and save model diagram (optional)
modelType = 'PolyMECNN'
modelNum = 1
training.saveModelDiagram(PolyMECNN_Built, nameModel=modelType)

#Step 6: Set hyperparameters for training configuration
E, batchSize, optimiser, lossFunction = training.setHyperparamsPolyMECNN()

#Step 7: Compile Model with loss function and optimiser
PolyMECNN_Compiled = training.compileModel(PolyMECNN_Built, lossFunction, optimiser)

#Step 8: Train CNN Model
PolyMECNN_Trained, PolyMECNN_History = training.trainModel(PolyMECNN_Compiled, 
                                                           XTrain_PolyMECNN, yTrainOHE_PolyMECNN, 
                                                           E, batchSize,  
                                                           nameModel=modelType, numModel=modelNum, pathSave=path_ME1,
                                                           vsplit=False, vsplitfactor=0.1,
                                                           XVal=XTest_PolyMECNN, yVal=yTestOHE_PolyMECNN)

#Step 9: Plot evolution of loss and accuracy
training.plotModelHistory(PolyMECNN_History, 
                          modelType, modelNum,
                          E)
#Step 10: Evaluate Model on Test Set 
PolyMECNN_Loaded = training.loadpretrainedmodel(path_ME1+'/'+modelType+'_'+str(modelNum)+'.h5')
PolyMECNN_TestLoss, PolyMECNN_TestAcc = training.evaluateModel(PolyMECNN_Loaded, batchSize, 
                                                                       XTest_PolyMECNN, yTestOHE_PolyMECNN)

#%%
#--------------------------------------------------------------------------------------------------------------------------------------
#BUILD, TRAIN & EVALUATE MONOPHONIC MELODY EXTRACTION SINGLE CNN

#Step 1: Load Training and Test Set Arrays
XTrain_MonoMECNN, yTrain_MonoMECNN = preprocessing.loadArrays(path_ME2, 'XTrain_ME'+'.npy','yTrain_ME'+'.npy')

#Step 2: One hot encode labels
yTrainOHE_MonoMECNN = training.encodeLabelsSingle(yTrain_MonoMECNN,numClasses)

#Step 3: do random initialisation of network
seed = 100
initialisation = training.initialiseWithSeed(seed)

#Step 4: Build Model Architecture
MonoMECNN_Built = training.buildSingleMECNN(rows=K,columns=T,channels=H)

#Step 5: Set Model Type and Number, and save model diagram (optional)
modelType = 'MonoMECNN'
modelNum = 1
training.saveModelDiagram(MonoMECNN_Built, nameModel=modelType)

#Step 6: Set hyperparameters for training configuration
E, batchSize, optimiser, lossFunction = training.setHyperparamsMonoMECNN()

#Step 7: Compile Model with loss function and optimiser
MonoMECNN_Compiled = training.compileModel(MonoMECNN_Built, lossFunction, optimiser)

#Step 8: Train CNN Model
MonoMECNN_Trained, MonoMECNN_History = training.trainModel(MonoMECNN_Compiled, 
                                                           XTrain_MonoMECNN, yTrainOHE_MonoMECNN, 
                                                           E, batchSize,  
                                                           nameModel=modelType, numModel=modelNum, pathSave=path_ME2,
                                                           vsplit=True, vsplitfactor=0.1)
#Step 9: Plot evolution of loss and accuracy
training.plotModelHistory(MonoMECNN_History, 
                          modelType, modelNum,
                          E)

#Step 10: Evaluate Model on Test Set 
MonoMECNN_Loaded = training.loadpretrainedmodel(path_ME2+'/'+modelType+'_'+str(modelNum)+'.h5')

#%%
#---------------------------------------------------------------------------------------------------------------------------------------
#PLOT CONFUSION MATRICES & GET PREDICTIONS FOR SINGLE CNN

#Step 1: Set Model Type, Number and path
modelType = 'PolyMECNN'
modelNum = 1
selPath = path_ME1

#Step 2: Load labels and one hot encode
XTestPoly, yTestPoly = preprocessing.loadArrays(path_ME1, 'XTest_ME'+'.npy','yTest_ME'+'.npy')
yTestPolyOHE = training.encodeLabelsSingle(yTestPoly, numClasses)

#Step 3: compute confusion matrix for CNN's predictions on test set
loadedCNN = training.loadpretrainedmodel(selPath+'/'+modelType+'_'+str(modelNum)+'.h5')
lowestMIDIGT, labelListGT, classNamesGT = training.getClassNames(yTestPoly)
yPredPoly = predicting.predictOutputSingle(loadedCNN, XTestPoly)
lowestMIDIPred, labelListPred, classNamesPred = training.getClassNames(yPredPoly)
ConfusionMatrix, Predictions = training.plotConfusionMatrixSingle(loadedCNN, modelType, modelNum,
                                                                  XTestPoly, yTestPolyOHE, 
                                                                  classNamesPred)

#%%
#---------------------------------------------------------------------------------------
#POST-PROCESSING FOR SINGLE CNN

#Step 1: save array of all f0 Training Labels
pieces = preprocessing.listAudioFilenames(path_train_ME1)
f0TrainingLabels = postprocessing.buildf0labelStream(path_train_ME1, pieces)
preprocessing.saveLabelArray(f0TrainingLabels, 'f0TrainingLabels'+'.npy', path_data)

#Step 2: Load Test Set Arrays
XTestPoly, yTestPoly = preprocessing.loadArrays(path_ME1, 
                                                'XTest_ME'+'.npy', 'yTest_ME'+'.npy')

#Step 3: Load pre-trained CNN Model
Singlemodel = training.loadpretrainedmodel(path_ME1+'/'+'PolyMECNN_1.h5')

#Step 4: Load f0 Training Labels
f0TrainingLabels = preprocessing.loadLabelArray(path_data, 'f0TrainingLabels'+'.npy')

#Step 5: Get Smoothed Trajectory
yPredPolySmoothed = postprocessing.getsmoothedf0Traj(f0TrainingLabels, yTestPoly, Singlemodel, False, XTestPoly, numClasses)

#Step 6: Compare accuracy obtained with smoothed over raw f0 trajectory on test set
yPredPolyRaw = predicting.predictOutputSingle(Singlemodel, XTestPoly)
checksSmoothed, degreeSimilaritySmoothed = postprocessing.checkSimilarity(yTestPoly, yPredPolySmoothed)
checksRaw, degreeSimilarityRaw = postprocessing.checkSimilarity(yTestPoly, yPredPolyRaw)

#%%
#--------------------------------------------------------------------------------------------------------------------------------
#PREPROCESSING FOR MULTI-TASK MELODY EXTRACTOR 

#Step 1: Subsample training data for Polyphonic Extractor to form Target Task Training set of 150000 features
XPoly, yPoly = preprocessing.loadArrays(path_ME1, 'XTrain_ME'+'.npy', 'yTrain_ME'+'.npy')
XPolyHalved, yPolyHalved = preprocessing.pickNumFeatures(XPoly, yPoly, 150000)
preprocessing.saveInputOutputArrays(XPolyHalved, yPolyHalved, 'XTrainPoly_MTME'+'.npy', 'yTrainPoly_MTME'+'.npy', path_MTME)

#Step 2: Subsample training data for Polyphonic Extractor to form Auxiliary Task Training set of 150000 features
XMono, yMono = preprocessing.loadArrays(path_ME2, 'XTrain_ME'+'.npy', 'yTrain_ME'+'.npy')
XMonoHalved, yMonoHalved = preprocessing.pickNumFeatures(XMono, yMono, 150000)
preprocessing.saveInputOutputArrays(XMonoHalved, yMonoHalved, 'XTrainMono_MTME'+'.npy', 'yTrainMono_MTME'+'.npy', path_MTME)

#%%
#------------------------------------------------------------------------------------------------------------------------------------
#BUILD, TRAIN & EVALUATE MELODY EXTRACTION MULTI-TASK CNN (DO ONE-BY-ONE)

#Step 1: Load Training and Test Set Arrays 
XTrainPoly_MTMECNN, yTrainPoly_MTMECNN = preprocessing.loadArrays(path_MTME, 'XTrainPoly_MTME'+'.npy',
                                                                  'yTrainPoly_MTME'+'.npy')
XTrainMono_MTMECNN, yTrainMono_MTMECNN = preprocessing.loadArrays(path_MTME, 'XTrainMono_MTME'+'.npy',
                                                                  'yTrainMono_MTME'+'.npy')
XTestPoly_MTMECNN, yTestPoly_MTMECNN = preprocessing.loadArrays(path_ME1, 'XTest_ME'+'.npy','yTest_ME'+'.npy')

#Step 2: One hot encode labels
yTrainPolyOHE_MTMECNN = training.encodeLabelsSingle(yTrainPoly_MTMECNN, numClasses)
yTrainMonoOHE_MTMECNN = training.encodeLabelsSingle(yTrainMono_MTMECNN, numClasses)
yTestPolyOHE_MTMECNN = training.encodeLabelsSingle(yTestPoly_MTMECNN, numClasses)

#Step 3: do random initialisation of network
seed = 100
initialisation = training.initialiseWithSeed(seed)

#Step 4: Build Model Architecture
MTMECNN_Built = training.buildMTMECNN(rows=K,columns=T,channels=H)

#Step 5: Set model Type and Number (do one by one, changing i from 0 to 4), and save model diagram (optional)
modelType = 'MTMECNN'
modelNum = [1, 2, 3, 4, 5]
selModelNum = modelNum[i]
training.saveModelDiagram(MTMECNN_Built, nameModel=modelType)

#Step 6: Set hyperparameters for training configuration
E, batchSize, optimiser, lossFunction = training.setHyperparamsMTMECNN()
lossWeights = training.getLossWeights(selModelNum) 

#Step 7: Compile Model with loss function and optimiser
MTMECNN_Compiled = training.compileMultiTaskModel(MTMECNN_Built, lossFunction, lossWeights, optimiser)

#Step 8: Train CNN Model
MTMECNN_Trained, MTMECNN_History = training.trainMultiTaskModel(MTMECNN_Compiled, 
                                                                XTrainPoly_MTMECNN, yTrainPolyOHE_MTMECNN, 
                                                                XTrainMono_MTMECNN, yTrainMonoOHE_MTMECNN,
                                                                E, batchSize,  
                                                                nameModel=modelType, numModel=selModelNum, pathSave=path_MTME,
                                                                vsplitfactor=0.1)

#Step 9: Plot evolution of loss and accuracy
training.plotMultiTaskModelHistory(MTMECNN_History, 
                                   modelType, selModelNum,
                                   E)

#Step 10: Evaluate Model on Test Set
MTMECNN_Loaded = training.loadpretrainedmodel(path_MTME+'/'+modelType+'_'+str(selModelNum)+'.h5')
MTMECNN_Metrics, MTMECNN_Score, MTMECNN_TestLoss, MTMECNN_TestAcc = training.evaluateMultiTaskModel(MTMECNN_Loaded, batchSize, 
                                                                                                    XTestPoly_MTMECNN, yTestPolyOHE_MTMECNN)
#%%
#-------------------------------------------------------------------------------------------------------------------------------------------
#PLOT CONFUSION MATRICES & GET PREDICTIONS FOR MULTI-TASK CNN

#Step 1: Set model Type, model Number and path
modelType = 'MTMECNN'
modelNum = [1, 2, 3, 4, 5]
selPath = path_MTME

#Step 2: Load labels and one hot encode
XTestMT, yTestMT = preprocessing.loadArrays(path_ME1, 'XTest_ME'+'.npy','yTest_ME'+'.npy')
yTestMTOHE = training.encodeLabelsSingle(yTestMT, numClasses)

#Step 3: compute confusion matrices for each CNN's predictions on test set
for num in modelNum:
    loadedCNN = training.loadpretrainedmodel(selPath+'/'+modelType+'_'+str(num)+'.h5')
    lowestMIDIGT, labelListGT, classNamesGT = training.getClassNames(yTestMT)
    yPredMT = predicting.predictOutputMT(loadedCNN, [XTestMT, XTestMT])
    lowestMIDIPred, labelListPred, classNamesPred = training.getClassNames(yPredMT)
    ConfusionMatrix = training.plotConfusionMatrixMT(loadedCNN, modelType, num,
                                                     [XTestMT, XTestMT],
                                                     [yTestMTOHE, yTestMTOHE], 
                                                     classNamesPred)

#%%
#------------------------------------------------------------------------------------------------------------------------------------------
#POST-PROCESSING FOR MULTI-TASK CNN

#Step 1: Load Test Set Arrays
XTestPoly, yTestPoly = preprocessing.loadArrays(path_ME1, 
                                                'XTest_ME'+'.npy', 'yTest_ME'+'.npy')

#Step 2: Load pre-trained CNN Model (do one-by-one, changing i from 0 to 4)
MTmodelfilenames = ['MTMECNN_1.h5', 'MTMECNN_2.h5', 'MTMECNN_3.h5', 'MTMECNN_4.h5', 'MTMECNN_5.h5']
MTmodel = training.loadpretrainedmodel(path_MTME+'/'+MTmodelfilenames[i])

#Step 3: Load f0 Training Labels
f0TrainingLabels = preprocessing.loadLabelArray(path_data, 'f0TrainingLabels'+'.npy')

#Step 4: Get Smoothed Trajectory
yPredPolySmoothed = postprocessing.getsmoothedf0Traj(f0TrainingLabels, yTestPoly, MTmodel, True, XTestPoly, numClasses)

#Step 5: Compare accuracy obtained with smoothed over raw f0 trajectory on test set
yPredPolyRaw = predicting.predictOutputMT(MTmodel, XTestPoly)
checksSmoothed, degreeSimilaritySmoothed = postprocessing.checkSimilarity(yTestPoly, yPredPolySmoothed)
checksRaw, degreeSimilarityRaw = postprocessing.checkSimilarity(yTestPoly, yPredPolyRaw)

#%%
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#EVALUATE MELODY EXTRACTION SYSTEMS

#Step 1: Load Test Set Arrays
XTestPoly, yTestPoly = preprocessing.loadArrays(path_ME1, 
                                                'XTest_ME'+'.npy', 'yTest_ME'+'.npy') 
#Step 2: Define label file, models to evaluate and model names
labelFile = 'f0TrainingLabels.npy'
modelsEval = ['PolyMECNN_1.h5','MTMECNN_1.h5','MTMECNN_2.h5','MTMECNN_3.h5','MTMECNN_4.h5','MTMECNN_5.h5']
modelNames = ['Melodia','Deep Salience Map CNN',
             'Violin Melody Extraction Single CNN without smoothing','Violin Melody Extraction Single CNN with smoothing',
             'Violin Melody Extraction MT-CNN 1 without smoothing','Violin Melody Extraction MT-CNN 1 with smoothing',
             'Violin Melody Extraction MT-CNN 2 without smoothing','Violin Melody Extraction MT-CNN 2 with smoothing',
             'Violin Melody Extraction MT-CNN 3 without smoothing','Violin Melody Extraction MT-CNN 3 with smoothing',
             'Violin Melody Extraction MT-CNN 4 without smoothing','Violin Melody Extraction MT-CNN 4 with smoothing',
             'Violin Melody Extraction MT-CNN 5 without smoothing','Violin Melody Extraction MT-CNN 5 with smoothing',]

#Step 3: get ground truth arrays
f0TrajGT = evaluation.getF0TrajfromF0Labels(yTestPoly)
f0CentsGT, voicingGT = evaluation.getf0CentsVoicingArrays(f0TrajGT)

#Step 4: Evaluate Melodia
timesMelodia, f0TrajMelodia = evaluation.getMelodiaf0Traj(path_test_ME1, fs, noteMin, noteMax, hopSize_Sec, yTestPoly)
f0CentsMelodia, voicingMelodia = evaluation.getf0CentsVoicingArrays(f0TrajMelodia)
VR_Melodia, VFA_Melodia, RPA_Melodia, RCA_Melodia, OA_Melodia = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                        voicingMelodia, f0CentsMelodia, 
                                                                                        modelNames[0])
#Step 5: Evaluate DSM CNN
timesDSMCNN, f0TrajDSMCNN = evaluation.getDSMCNNf0Traj(path_test_ME1, hopSize_Sec, fs, noteMin, noteMax, yTestPoly)
f0CentsDSMCNN, voicingDSMCNN = evaluation.getf0CentsVoicingArrays(f0TrajDSMCNN)
VR_DSMCNN, VFA_DSMCNN, RPA_DSMCNN, RCA_DSMCNN, OA_DSMCNN = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                   voicingDSMCNN, f0CentsDSMCNN, 
                                                                                   modelNames[1])
#Step 6: Evaluate PolyMECNN_1 Raw
timesPolyMECNN1R, f0TrajPolyMECNN1R = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                                 modelsEval[0], path_ME1, labelFile, path_data, 
                                                                 smooth=False, isMT=False)
f0CentsPolyMECNN1R, voicingPolyMECNN1R = evaluation.getf0CentsVoicingArrays(f0TrajPolyMECNN1R)
VR_PolyMECNN1R, VFA_PolyMECNN1R, RPA_PolyMECNN1R, RCA_PolyMECNN1R, OA_PolyMECNN1R = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                            voicingPolyMECNN1R, f0CentsPolyMECNN1R, 
                                                                                                            modelNames[2])
#Step 7: Evaluate PolyMECNN_1 Smoothed
timesPolyMECNN1S, f0TrajPolyMECNN1S = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                                 modelsEval[0], path_ME1, labelFile, path_data, 
                                                                 smooth=True, isMT=False)
f0CentsPolyMECNN1S, voicingPolyMECNN1S = evaluation.getf0CentsVoicingArrays(f0TrajPolyMECNN1S)
VR_PolyMECNN1S, VFA_PolyMECNN1S, RPA_PolyMECNN1S, RCA_PolyMECNN1S, OA_PolyMECNN1S = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                            voicingPolyMECNN1S, f0CentsPolyMECNN1S, 
                                                                                                            modelNames[3])
#Step 8: Evaluate MTMECNN_1 Raw
timesMTMECNN1R, f0TrajMTMECNN1R = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[1], path_MTME, labelFile, path_data, 
                                                             smooth=False, isMT=True)
f0CentsMTMECNN1R, voicingMTMECNN1R = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN1R)
VR_MTMECNN1R, VFA_MTMECNN1R, RPA_MTMECNN1R, RCA_MTMECNN1R, OA_MTMECNN1R = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN1R, f0CentsMTMECNN1R, 
                                                                                                  modelNames[4])
#Step 9: Evaluate MTMECNN_1 Smoothed
timesMTMECNN1S, f0TrajMTMECNN1S = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[1], path_MTME, labelFile, path_data, 
                                                             smooth=True, isMT=True)
f0CentsMTMECNN1S, voicingMTMECNN1S = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN1S)
VR_MTMECNN1S, VFA_MTMECNN1S, RPA_MTMECNN1S, RCA_MTMECNN1S, OA_MTMECNN1S = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN1S, f0CentsMTMECNN1S, 
                                                                                                  modelNames[5])
#Step 10: Evaluate MTMECNN_2 Raw
timesMTMECNN2R, f0TrajMTMECNN2R = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[2], path_MTME, labelFile, path_data, 
                                                             smooth=False, isMT=True)
f0CentsMTMECNN2R, voicingMTMECNN2R = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN2R)
VR_MTMECNN2R, VFA_MTMECNN2R, RPA_MTMECNN2R, RCA_MTMECNN2R, OA_MTMECNN2R = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN2R, f0CentsMTMECNN2R, 
                                                                                                  modelNames[6])

#Step 11: Evaluate MTMECNN_2 Smoothed
timesMTMECNN2S, f0TrajMTMECNN2S = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[2], path_MTME, labelFile, path_data, 
                                                             smooth=True, isMT=True)    
f0CentsMTMECNN2S, voicingMTMECNN2S = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN2S)
VR_MTMECNN2S, VFA_MTMECNN2S, RPA_MTMECNN2S, RCA_MTMECNN2S, OA_MTMECNN2S = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN2S, f0CentsMTMECNN2S, 
                                                                                                  modelNames[7])
#Step 12: Evaluate MTMECNN_3 Raw
timesMTMECNN3R, f0TrajMTMECNN3R = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[3], path_MTME, labelFile, path_data, 
                                                             smooth=False, isMT=True)
f0CentsMTMECNN3R, voicingMTMECNN3R = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN3R)
VR_MTMECNN3R, VFA_MTMECNN3R, RPA_MTMECNN3R, RCA_MTMECNN3R, OA_MTMECNN3R = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN3R, f0CentsMTMECNN3R, 
                                                                                                  modelNames[8])
#Step 13: Evaluate MTMECNN_3 Smoothed
timesMTMECNN3S, f0TrajMTMECNN3S = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[3], path_MTME, labelFile, path_data, 
                                                             smooth=True, isMT=True)
f0CentsMTMECNN3S, voicingMTMECNN3S = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN3S)
VR_MTMECNN3S, VFA_MTMECNN3S, RPA_MTMECNN3S, RCA_MTMECNN3S, OA_MTMECNN3S = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN3S, f0CentsMTMECNN3S, 
                                                                                                  modelNames[9])
#Step 14: Evaluate MTMECNN_4 Raw
timesMTMECNN4R, f0TrajMTMECNN4R = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[4], path_MTME, labelFile, path_data, 
                                                             smooth=False, isMT=True)
f0CentsMTMECNN4R, voicingMTMECNN4R = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN4R)
VR_MTMECNN4R, VFA_MTMECNN4R, RPA_MTMECNN4R, RCA_MTMECNN4R, OA_MTMECNN4R = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN4R, f0CentsMTMECNN4R, 
                                                                                                  modelNames[10])
#Step 15: Evaluate MTMECNN_4 Smoothed
timesMTMECNN4S, f0TrajMTMECNN4S = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[4], path_MTME, labelFile, path_data, 
                                                             smooth=True, isMT=True)    
f0CentsMTMECNN4S, voicingMTMECNN4S = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN4S)
VR_MTMECNN4S, VFA_MTMECNN4S, RPA_MTMECNN4S, RCA_MTMECNN4S, OA_MTMECNN4S = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN4S, f0CentsMTMECNN4S, 
                                                                                                  modelNames[11])
#Step 16: Evaluate MTMECNN_5 Raw
timesMTMECNN5R, f0TrajMTMECNN5R = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[5], path_MTME, labelFile, path_data, 
                                                             smooth=False, isMT=True)
f0CentsMTMECNN5R, voicingMTMECNN5R = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN5R)
VR_MTMECNN5R, VFA_MTMECNN5R, RPA_MTMECNN5R, RCA_MTMECNN5R, OA_MTMECNN5R = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN5R, f0CentsMTMECNN5R, 
                                                                                                  modelNames[12])
#Step 17: Evaluate MTMECNN_5 Smoothed
timesMTMECNN5S, f0TrajMTMECNN5S = evaluation.getVMECNNf0Traj(XTestPoly, 
                                                             modelsEval[5], path_MTME, labelFile, path_data, 
                                                             smooth=True, isMT=True)
f0CentsMTMECNN5S, voicingMTMECNN5S = evaluation.getf0CentsVoicingArrays(f0TrajMTMECNN5S)
VR_MTMECNN5S, VFA_MTMECNN5S, RPA_MTMECNN5S, RCA_MTMECNN5S, OA_MTMECNN5S = evaluation.getMEmetrics(voicingGT, f0CentsGT, 
                                                                                                  voicingMTMECNN5S, f0CentsMTMECNN5S, 
                                                                                                  modelNames[13])
