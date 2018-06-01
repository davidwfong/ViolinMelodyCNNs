#IMPORT RELEVANT MODULES
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random
from pydub import AudioSegment
import librosa
import pumpp
#-------------------------------------------------------------------------------------------------------
# FUNCTIONS

def getChromaRange(instrument):
    if instrument == 'violin':
        startNote = 'G3'
        endNote = 'F#7' 
    if instrument == 'piano':
        startNote = 'A0'
        endNote = 'A7'
    return startNote, endNote


def getMIDIfromNoteRange(startNote, endNote):
    lowest = librosa.note_to_midi(startNote,round_midi=False)
    highest = librosa.note_to_midi(endNote,round_midi=False)
    midiNoteNums = np.array(range(int(lowest), int(highest+1), 1))
    return midiNoteNums


def getMIDIfromNote(note):
    midiNoteNum = librosa.note_to_midi(note)
    return midiNoteNum


def getF0fromMIDI(midiNotes):
    for i in midiNotes:
        frequencies = librosa.core.midi_to_hz(midiNotes)    
    return frequencies 


def getF0fromNote(notes):
    frequencies = librosa.core.note_to_hz(notes)    
    return frequencies 


def getF0LabelsfromMIDI(midiNotes):
    offset = midiNotes[0]
    freqLabels = midiNotes - (offset-1)
    return freqLabels


def getNotesfromF0Labels(F0Labels, offset):
    Notes = []
    for el in F0Labels:
        if el == 0:
            Notes.append('None')
        else:
            Note = librosa.core.midi_to_note(el+offset)
            Notes.append(Note)
            
    return Notes


def getF0fromF0Labels(F0Labels, offset):
    F0s = []
    for el in F0Labels:
        if el == 0:
            F0s.append('None')
        else:
            F0 = librosa.core.midi_to_hz(el+offset)
            F0s.append(F0)
            
    return F0s


def getNotefromF0(frequency):
    note = librosa.core.hz_to_note(frequency)
    return note


def getMIDIfromF0(frequency):
    midiNoteNum = librosa.core.hz_to_midi(frequency)
    return midiNoteNum


def convertWavToArray(audioFile):
    data, fs = librosa.load(audioFile, sr=16000, mono=True) 
    return data, fs


def convertWavsToArray(audioPath):
    counter = 0
    for filename in os.listdir(audioPath):
        if filename.endswith(".wav"):
            data, fs = librosa.core.load(audioPath+'/'+filename, sr=16000, mono=True)
            if counter == 0:
                audioData = data
            else:
                audioData = np.concatenate((audioData, data))
            counter = counter+1
    return audioData


def convertSecToSample(fs,time):
    sample = (time*fs).astype(int)
    return sample
    

def convertSampleToSec(fs,sample):
    time = sample/fs
    return time


def extractFileName(filepath):
    filestrings = filepath.split("/")
    fileIsolExt = filestrings[-1]
    fileIsolComponents = fileIsolExt.split(".")
    fileIsol =  fileIsolComponents[0]
    return fileIsol


def extractFileNameParts(filepath):
    filestrings = filepath.split("_")
    return filestrings
    

def downsampleAudio(audiopath,newFs):
    for filename in os.listdir(audiopath):
        if filename.endswith(".wav"):
            data,fs = librosa.core.load(audiopath+'/'+filename, sr=newFs)
            librosa.output.write_wav(audiopath+'/'+filename, data, fs)
        else:
            continue


def cleanTestGTME(testdatapath, fs=16000):
    for filename in glob.glob(os.path.join(testdatapath, '*.txt')):
        fileIsol = extractFileName(filename)
        groundTruthRaw = pd.read_csv(filename, delimiter="\t", header=None)
        groundTruthRaw.columns = ["StartTime", "EndTime", "F0"]
        startTimes = groundTruthRaw.iloc[:,0]
        endTimes = groundTruthRaw.iloc[:,1]
        frequencies = groundTruthRaw.iloc[:,2]
        startTimesArray = startTimes.values
        endTimesArray = endTimes.values
        startSamplesArray = convertSecToSample(fs,startTimesArray)
        endSamplesArray = convertSecToSample(fs,endTimesArray)
        notesArray = getNotefromF0(frequencies)
        midiNotesArray = getMIDIfromNote(notesArray)
        startSamples = pd.DataFrame(startSamplesArray)
        endSamples = pd.DataFrame(endSamplesArray)
        midiNotes = pd.DataFrame(midiNotesArray)
        groundTruthClean = pd.concat([startSamples, endSamples, midiNotes], axis=1)  
        groundTruthClean.columns = ["StartSample", "EndSample", "MIDINoteNumber"]
        groundTruthClean.to_csv(testdatapath+'/'+fileIsol+'.csv', sep=',')
    
    
def cleanTrainGTME(traindatapath, fs=16000):
    for filename in glob.glob(os.path.join(traindatapath, '*.csv*')):
        fileIsol = extractFileName(filename)
        fileIsolstrings = extractFileNameParts(fileIsol)
        audiofileName = fileIsolstrings[0]
        groundTruthRaw = pd.read_csv(filename, delimiter=",", header=None)
        groundTruthRaw.columns = ["StartSample", "EndSample","Instrument","MIDINoteNumber","StartBeat","EndBeat","NoteValue"]
        groundTruthIsol = groundTruthRaw[groundTruthRaw.Instrument == "41"]  
        startSamplesIsol = groundTruthIsol.iloc[:,0]
        endSamplesIsol = groundTruthIsol.iloc[:,1]
        midiNotesIsol = groundTruthIsol.iloc[:,3]
        startSamplesIsolArray = startSamplesIsol.values
        endSamplesIsolArray = endSamplesIsol.values
        midiNotesIsolArray = midiNotesIsol.values
        startSamplesCleanArray = startSamplesIsolArray.astype(float)
        endSamplesCleanArray =  endSamplesIsolArray.astype(float)
        startSamplesCleanArrayDS = np.round(startSamplesCleanArray*fs/44100).astype(int)
        endSamplesCleanArrayDS = np.round(endSamplesCleanArray*fs/44100).astype(int)
        midiNotesCleanArray = midiNotesIsolArray.astype(int)
        startSamplesClean = pd.DataFrame(startSamplesCleanArrayDS)
        endSamplesClean = pd.DataFrame(endSamplesCleanArrayDS)
        midiNotesClean = pd.DataFrame(midiNotesCleanArray)
        groundTruthClean = pd.concat([startSamplesClean, endSamplesClean, midiNotesClean], axis=1)  
        groundTruthClean.columns = ["StartSample", "EndSample", "MIDINoteNumber"]
        groundTruthClean.to_csv(traindatapath+'/'+audiofileName+'.csv', sep=',')
 
    
def convertAudioToWav(audiopath):
    for filename in os.listdir(audiopath):
        if filename.endswith(".wav"):
            continue
        else:
            filestrings = filename.split('/')
            filenamestrings = filestrings[-1].split('.')
            ext = filenamestrings[-1]
            #actualname = filenamestrings[0:len(filenamestrings)-1]
            actualname = ''.join(filenamestrings[0:len(filenamestrings)-1])
            audioStream = AudioSegment.from_file(audiopath+'/'+filename,format=ext)
            audioStream.export(audiopath+'/'+actualname+'.wav', format="wav")
            os.remove(audiopath+'/'+filename)


def downmixAudio(audiopath):
    for filename in os.listdir(audiopath):
        if filename.endswith(".wav"):
            data,fs = librosa.core.load(audiopath+'/'+filename,sr=44100)
            data_mono = librosa.core.to_mono(data)
            librosa.output.write_wav(audiopath+'/'+filename, data_mono, fs)
        else:
            continue
   
     
def getlistFilenames(audiopath):   
    filenames = []
    for filename in glob.glob(os.path.join(audiopath, '*.wav')):
        namefile = extractFileName(filename)
        filenames.append(namefile)
    return filenames

        
def augmentDataME(audiopath,audionewpath,maxTransposition,fs=16000):
    counter = 0
    for filename in os.listdir(audiopath):
        if filename.endswith("wav"):
            data,fs = librosa.core.load(audiopath+'/'+filename,sr=fs)
            fileIsolname = extractFileName(filename)
            groundTruth = pd.read_csv(audiopath+'/'+fileIsolname+'.csv', delimiter=",")
            startSamples = groundTruth.iloc[:,1]
            endSamples = groundTruth.iloc[:,2]
            midiNotes = groundTruth.iloc[:,3]
            midiNotesArray = midiNotes.values
            for i in range(0,maxTransposition+1,1):
                data_shifted = librosa.effects.pitch_shift(data, fs, n_steps=i, bins_per_octave=12)    
                newfilename = fileIsolname + '_shift' + str(i) 
                librosa.output.write_wav(audionewpath+'/'+newfilename+'.wav', data_shifted, fs)
                midiNotesTransposedArray = midiNotesArray+i
                midiNotesShift = pd.DataFrame(midiNotesTransposedArray)
                groundTruthShift = pd.concat([startSamples, endSamples, midiNotesShift], axis=1)  
                groundTruthShift.columns = ["StartSample", "EndSample", "MIDINoteNumber"]
                groundTruthShift.to_csv(audionewpath+'/'+newfilename+'.csv', sep=',')
            counter = counter+1
            print("shifted File" + str(counter))


def findRelativeSize(biggersetpath,smallersetpath,fs=16000):
    samplesBiggerSet = 0
    samplesSmallerSet = 0
    for filename in os.listdir(biggersetpath):
        if filename.endswith("wav"):
            data,fs = librosa.core.load(biggersetpath+'/'+filename,sr=fs)
            samplesBiggerSet = samplesBiggerSet + len(data)
    for filename in os.listdir(smallersetpath):
        if filename.endswith("wav"):
            data,fs = librosa.core.load(smallersetpath+'/'+filename,sr=fs)
            samplesSmallerSet = samplesSmallerSet + len(data)
    sizingFactor = samplesBiggerSet /samplesSmallerSet
    return sizingFactor

        
def getHCQTParams(fs, hopSize_Sec, noteMin, H, K, B):
    fmin = librosa.core.note_to_hz(noteMin)
    h=[]
    for i in range(1, H+1, 1):
        h.append(i)
    R = int(fs * hopSize_Sec)
    numOctaves = int(K / B)
    numBinsSemitone = int(B / 12)
    return fmin, h, R, numOctaves, numBinsSemitone 


def getHCQTArray(x,fs,fmin,h,R,numOctaves,numBinsSemitone):
    P_HCQT = pumpp.feature.HCQTMag(name='HCQT', 
                                   sr=fs, hop_length=R, 
                                   n_octaves=numOctaves, 
                                   over_sample=numBinsSemitone, 
                                   fmin=fmin, harmonics=h, 
                                   log=False, 
                                   conv='channels_last')
    SDict_HCQT = P_HCQT.transform(y=x,sr=fs)
    S4d_HCQT = SDict_HCQT.values()[0]
    numharmonics_HCQT = np.size(S4d_HCQT, axis=3)
    S3d_HCQT = S4d_HCQT[0,:,:,:]
    SFinal_HCQT = np.transpose(S3d_HCQT,(1,0,2))
    S_Shape = SFinal_HCQT.shape
    S_HCQT = np.empty(S_Shape)
    for i in range(numharmonics_HCQT):
        S_HCQT[:,:,i] = sklearn.preprocessing.robust_scale(SFinal_HCQT[:,:,i],axis=1)
    return S_HCQT


def getInputFeature(SArray,centreframe,context):
    diff = int((context - 1)/2)
    inputFeature = SArray[:,centreframe-diff:centreframe+diff+1,:]
    return inputFeature, diff


def loadGTME(filename):
    GTdata = np.genfromtxt(filename, delimiter=',')
    GTdata= GTdata[1:,1:]
    startsamples = GTdata[:,0].astype(int)
    endsamples = GTdata[:,1].astype(int)
    f0labels = GTdata[:,2].astype(int)
    return GTdata, startsamples, endsamples, f0labels
    

def convertSamplesToFrame(samples,R,fs=16000):
    frames = np.round(samples/R).astype(int)
    return frames


def getShapeHCQT(SArray):
    HCQTShape = SArray.shape
    numBins = HCQTShape[0]
    numFrames = HCQTShape[1]
    numHarmonics = HCQTShape[2]
    return numBins, numFrames, numHarmonics


def convertMIDINoteToLabel(MIDInote,lowestnote='G3'):
    lowestNote = librosa.note_to_midi(lowestnote)
    labelCNN = MIDInote - lowestNote + 1
    return labelCNN


def extractFeaturesME(audiopath,T, fs, hopSize_Sec, noteMin, H, K, B):
    inputFeatures = []
    labels = []
    for filename in os.listdir(audiopath):
        diff = int((T - 1)/2)
        if filename.endswith("wav"):
            fmin, h, R, numOctaves, numBinsSemitone = getHCQTParams(fs, hopSize_Sec, noteMin, H, K, B)
            fileIsol = extractFileName(filename)
            pathcsv = audiopath+'/'+fileIsol+'.csv'
            GTData,startsamples,endsamples,f0labels = loadGTME(pathcsv)
            startframes = convertSamplesToFrame(startsamples,R)
            endframes = convertSamplesToFrame(endsamples,R)
            CNNlabels = convertMIDINoteToLabel(f0labels,lowestnote=noteMin)
            x = convertWavToArray(audiopath+'/'+filename)[0]
            spectrogram = getHCQTArray(x,fs,fmin,h,R,numOctaves,numBinsSemitone)
            numBins, numFrames, numHarmonics = getShapeHCQT(spectrogram)
            for centreframe in range(diff,numFrames-diff,1):
                newFeature,diff = getInputFeature(spectrogram,centreframe,context=T)
                newFeature4D = np.expand_dims(newFeature,axis=0)
                inputFeatures.append(newFeature4D)
                for index, sf in enumerate(startframes):
                    if centreframe < startframes[0] or centreframe > endframes[-1]:
                        newLabel = int(0)
                        newLabel2D = np.expand_dims(newLabel,axis=0)
                        labels.append(newLabel2D)  
                        break
                    else:
                        if centreframe < sf:
                            continue
                        elif centreframe >= sf: 
                            if centreframe <= endframes[index]:
                                newLabel = int(CNNlabels[index])
                                newLabel2D = np.expand_dims(newLabel,axis=0)
                                labels.append(newLabel2D)                   
                                break
                            else:
                                if centreframe < startframes[index+1]:
                                    newLabel = int(0)
                                    newLabel2D = np.expand_dims(newLabel,axis=0)
                                    labels.append(newLabel2D)  
                                    break
                                else:
                                    continue
                    
    X = np.concatenate(inputFeatures, axis=0)
    y = np.concatenate(labels,axis=0)    
    return X, y


def extractInputFeatures(audiofile, T, fs, hopSize_Sec, noteMin, H, K, B):
    inputFeatures = []
    diff = int((T - 1)/2)
    fmin, h, R, numOctaves, numBinsSemitone = getHCQTParams(fs, hopSize_Sec, noteMin, H, K, B)
    x = convertWavToArray(audiofile)[0]
    spectrogram = getHCQTArray(x, fs, fmin, h, R, numOctaves, numBinsSemitone)
    numBins, numFrames, numHarmonics = getShapeHCQT(spectrogram)
    for centreframe in range(diff,numFrames-diff,1):
        newFeature,diff = getInputFeature(spectrogram, centreframe, context=T)
        newFeature4D = np.expand_dims(newFeature, axis=0)
        inputFeatures.append(newFeature4D)
                    
    X = np.concatenate(inputFeatures, axis=0)    
    return X


def saveInputOutputArrays(X, y, filenameX, filenamey, writePath):
    np.save(writePath+'/'+filenameX, X)
    np.save(writePath+'/'+filenamey, y)

    
def saveLabelArray(y, filenamey, writePath):
    np.save(writePath+'/'+filenamey, y)


def listAudioFilenames(directory):
    filenames = []
    for filename in glob.glob(os.path.join(directory, '*.wav')):
        fileIsol = extractFileName(filename)
        filenames.append(fileIsol)
    
    return filenames


def loadArrays(readPath, filenameX, filenamey):
    XArray = np.load(readPath+'/'+filenameX)
    yArray = np.load(readPath+'/'+filenamey)
    return XArray, yArray


def loadLabelArray(readPath, filenamey):
    yArray = np.load(readPath+'/'+filenamey)
    return yArray


def pickSubsetfromData(inputData,outputData,reductionFactor): 
    subset = random.sample(zip(inputData,outputData), int(float(np.size(outputData))/reductionFactor))
    xData = np.array([i[0] for i in subset])
    yData = np.array([i[1] for i in subset])
    return xData, yData


def pickNumFeatures(inputData,outputData,numFeatures): 
    subset = random.sample(zip(inputData,outputData), numFeatures)
    xData = np.array([i[0] for i in subset])
    yData = np.array([i[1] for i in subset])
    return xData, yData


def plotLabelDistributions(labelFull,labelSubset,labelType):
    plt.figure(1)
    if(labelType=='poly'):
        plt.figure(figsize=(15, 21))
    if(labelType=='mono'):
        plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    plt.hist(labelFull, bins=np.arange(min(labelFull)+0-0.5, 48+0.5, 1),
             edgecolor='black', linewidth=0.8)
    plt.xlabel('f0 label')
    plt.xticks(np.arange(0, max(labelFull)+1, 1))
    plt.ylabel('Frequency')
    plt.yticks(np.arange(0, max(labelFull)+1, 10000))
    plt.title('Distributon of f0 labels for full training set (top) and training subset (bottom)') 
    plt.subplot(2,1,2)
    plt.rc('axes', axisbelow=True)
    plt.hist(labelSubset, bins=np.arange(min(labelSubset)+0-0.5, 48+0.5, 1),
             edgecolor='black', linewidth=0.8)
    plt.xlabel('f0 label',fontsize=15)
    plt.xticks(np.arange(0, max(labelSubset)+1, 1),fontsize= 11)
    plt.ylabel('Frequency', fontsize=15)
    if(labelType=='poly'):
        plt.yticks(np.arange(0, 110000, 2500))
    if(labelType=='mono'):
        plt.yticks(np.arange(0, 35000, 2500))
    plt.grid(axis='y',zorder=0)
    if(labelType=='poly'):
        plt.savefig('labelDistributionPoly.png')
    elif(labelType=='mono'):
        plt.savefig('labelDistributionMono.png')




