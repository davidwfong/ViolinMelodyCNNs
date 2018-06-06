#Preliminary Step: import relevant modules
import preprocessing
import training
import postprocessing
import predicting

#Step 1: specify path of violin melody extractor
path = raw_input("Insert path of violin melody extractor\n")

#Step 2: specify audio file name
audiofilename = raw_input("Insert name of audio file including .wav extension\n")
audiofile = path + '/' + audiofilename

#Step 3: Hardcode Input Feature Representation parameters
fs=16000
hopSize_Sec = 0.010
noteMin = 'G3'
H=2
K=192
B=48
T=11
numClasses=49

#Step 4: Extract input features from input polyphonic violin signal
XHCQT = preprocessing.extractInputFeatures(audiofile, 
                                           T, fs, hopSize_Sec, noteMin, H, K, B)

#Step 5: load f0 label array for smoothing, which must be placed within path directory
f0TrainingLabels = preprocessing.loadLabelArray(path, 'f0TrainingLabels'+'.npy')

#Step 6: specify path and name of model file which must be placed within path directory
modelfilename = raw_input("Insert name of model file including .h5 extension\n")

#Step 7: Predict violin trajectory using specified VME CNN model
if(modelfilename=='PolyMECNN_1.h5'):
    loadedModel = training.loadpretrainedmodel(path+'/'+modelfilename)
    predViolinMelodyRaw =  predicting.predictOutputSingle(loadedModel, XHCQT)
    predViolinMelodySmoothed = postprocessing.getsmoothedf0Traj(f0TrainingLabels, predViolinMelodyRaw, 
                                                                loadedModel, False, XHCQT, numClasses)
elif(modelfilename=='MTMECNN_1.h5' or modelfilename=='MTMECNN_2.h5' or 
     modelfilename=='MTMECNN_3.h5' or modelfilename=='MTMECNN_4.h5' or modelfilename=='MTMECNN_5.h5'):
    loadedModel = training.loadpretrainedmodel(path+'/'+modelfilename)
    predViolinMelodyRaw =  predicting.predictOutputMT(loadedModel, XHCQT)
    predViolinMelodySmoothed = postprocessing.getsmoothedf0Traj(f0TrainingLabels, predViolinMelodyRaw, 
                                                                loadedModel, True, XHCQT, numClasses)
else:
    print("Not a valid model filename")
    quit()
    
#Step 8: Create the .csv and MIDI file for violin melody prediction within path directory
predicting.createAnnotation(predViolinMelodySmoothed, hopSize_Sec, path, audiofilename, noteMin)    


