#IMPORT RELEVANT MODULES
from mir_eval import melody
import essentia
import essentia.standard as essentiaMelody
import predict_on_audio as dsm
import preprocessing
import postprocessing
import predicting
import training
import numpy as np
#-----------------------------------------------------------------------------------
#FUNCTIONS

def getMelodiaf0Traj(audioPath, fs, noteMin, noteMax, hopSize_Sec, refArray):
    rawArray = preprocessing.convertWavsToArray(audioPath)
    fmin = preprocessing.getF0fromNote(noteMin)
    fmax = preprocessing.getF0fromNote(noteMax)
    hopSize = int(hopSize_Sec * fs)
    inputArray = essentia.array(rawArray)
    melodia = essentiaMelody.PredominantPitchMelodia(binResolution=10,
                                                     filterIterations=3,
                                                     frameSize=2048,
                                                     guessUnvoiced=False,
                                                     harmonicWeight=0.8,
                                                     hopSize=hopSize,
                                                     magnitudeCompression=1,
                                                     magnitudeThreshold=40,
                                                     maxFrequency=fmax,
                                                     minDuration=100,
                                                     minFrequency=fmin,
                                                     numberHarmonics=20,
                                                     peakDistributionThreshold=0.9,
                                                     peakFrameThreshold=0.9,
                                                     pitchContinuity=27.5625,
                                                     referenceFrequency=55,
                                                     sampleRate=fs,
                                                     timeContinuity=100,
                                                     voiceVibrato=False,
                                                     voicingTolerance=0.2) 
    f0Traj, f0confidence = melodia(inputArray)
    f0Traj = f0Traj[0:len(refArray)]                       
    times = np.arange(0, len(refArray)*hopSize_Sec, hopSize_Sec)
    return times, f0Traj


def getDSMCNNf0Traj(audioPath, hopSize_Sec, fs, noteMin, noteMax, refArray):
    rawArray = preprocessing.convertWavsToArray(audioPath)
    inputHCQT, freqGrid, timeGrid = dsm.compute_hcqt(rawArray, fs)
    modelDSMCNN = dsm.load_model('melody2')
    saliencePred = dsm.get_single_test_prediction(modelDSMCNN, inputHCQT)
    times, f0Traj = dsm.get_singlef0(saliencePred, 
                                 freq_grid=freqGrid, 
                                 time_grid=timeGrid, 
                                 thresh=0.3, 
                                 use_neg=False)
    times = times[0:len(refArray)]     
    f0Traj = f0Traj[0:len(refArray)]  
    return times, f0Traj


def getVMECNNf0Traj(XHCQT, modelFile, modelPath, labelFile, labelPath, smooth, isMT):
    hopSize_Sec = 0.010
    numClasses = 49
    f0TrainingLabels = preprocessing.loadLabelArray(labelPath, labelFile)
    loadedModel = training.loadpretrainedmodel(modelPath+'/'+modelFile)
    if isMT == False:
        predViolinMelodyRaw =  predicting.predictOutputSingle(loadedModel, XHCQT)
        if smooth == False:
            labelTraj = predViolinMelodyRaw
        elif smooth == True:
            labelTraj = postprocessing.getsmoothedf0Traj(f0TrainingLabels, predViolinMelodyRaw, 
                                                         loadedModel, False, XHCQT, numClasses)
    elif isMT == True:
        predViolinMelodyRaw =  predicting.predictOutputMT(loadedModel, XHCQT)
        if smooth == False:
            labelTraj = predViolinMelodyRaw
        if smooth == True:
            labelTraj = postprocessing.getsmoothedf0Traj(f0TrainingLabels, predViolinMelodyRaw, 
                                                         loadedModel, True, XHCQT, numClasses)
    times = np.arange(0, len(XHCQT)*hopSize_Sec, hopSize_Sec)
    offset = preprocessing.getMIDIfromNote('G3') - 1
    f0Traj = preprocessing.getF0fromF0Labels(labelTraj, offset)
    f0Traj = [0 if el=='None' else el for el in f0Traj]
    f0Traj = np.array(f0Traj)
    return times, f0Traj


def getF0TrajfromF0Labels(labelTraj):
    offset = preprocessing.getMIDIfromNote('G3') - 1
    f0Traj = preprocessing.getF0fromF0Labels(labelTraj, offset)
    f0Traj = [0 if el=='None' else el for el in f0Traj]
    f0Traj = np.array(f0Traj)
    return f0Traj


def getf0CentsVoicingArrays(f0Values):
    f0ValuesArray, voicingArray = melody.freq_to_voicing(f0Values)
    f0CentsArray = melody.hz2cents(f0ValuesArray, base_frequency=10.0)
    return f0CentsArray, voicingArray


def getVoicingMeasures(voicing_GT, voicing_Pred):
    VoicingMeasures = melody.voicing_measures(voicing_GT, voicing_Pred)
    VR =  VoicingMeasures[0]
    VFA = VoicingMeasures[1]
    return VR, VFA


def getRPA(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred):
    RPA = melody.raw_pitch_accuracy(voicing_GT, f0Cents_GT, 
                                    voicing_Pred, f0Cents_Pred,
                                    cent_tolerance=50)
    return RPA


def getRCA(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred):
    RCA = melody.raw_chroma_accuracy(voicing_GT, f0Cents_GT, 
                                     voicing_Pred, f0Cents_Pred,
                                     cent_tolerance=50)
    return RCA
    

def getOA(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred):
    OA = melody.overall_accuracy(voicing_GT, f0Cents_GT, 
                                 voicing_Pred, f0Cents_Pred, 
                                 cent_tolerance=50)
    return OA


def getMEmetrics(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred, modelName):
    VR, VFA = getVoicingMeasures(voicing_GT, voicing_Pred)
    print('Voicing Recall Rate for ' +modelName+ ' is ' + str(VR))
    print('Voicing False Alarm Rate for ' +modelName+ ' is ' + str(VFA))
    RPA = getRPA(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred)
    print('Raw Pitch Accuracy for ' +modelName+ ' is ' + str(RPA))
    RCA = getRCA(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred)
    print('Raw Chroma Accuracy for ' +modelName+ ' is ' + str(RCA))
    OA = getOA(voicing_GT, f0Cents_GT, voicing_Pred, f0Cents_Pred)
    print('Overall Accuracy for ' +modelName+ ' is ' + str(OA))
    return VR, VFA, RPA, RCA, OA


    