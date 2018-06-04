#IMPORT RELEVANT MODULES
import pandas as pd
import numpy as np
import preprocessing
from midiutil import MIDIFile
#-------------------------------------------------------------------------------------------------------
#FUNCTIONS

def createGT(yPred, frameduration, audiopath, audiofilename, noteMin):
    track    = 0
    channel  = 0
    time     = 0    
    tempo    = 60   
    MyMIDI = MIDIFile(1) 
    MyMIDI.addTempo(track, time, tempo)
    
    lengthAudioFrames = len(yPred)
    times = np.arange(0.00, lengthAudioFrames, frameduration)
    times = ["%.2f" % (round(t, 2)) for t in times]
    offset = (preprocessing.getMIDIfromNote(noteMin)) - 1
    notesPred = preprocessing.getNotesfromF0Labels(yPred, offset)
    startTimesList = []
    endTimesList = []
    notesList = []
    for index, note in enumerate(notesPred):
        if(note == 'None'):
            continue
        else:
            if(note == notesPred[index-1] and note == notesPred[index+1]):
                continue
            else:
                if(note != notesPred[index-1] and note != notesPred[index+1]):
                    startTimesList.append(times[index])
                    notesList.append(note)
                    endTimesList.append(times[index])
                    MIDIpitch = preprocessing.getMIDIfromNote(note)
                    MIDItime = float(times[index])
                    MIDIduration = 0.01
                    MyMIDI.addNote(track, channel, MIDIpitch, MIDItime, MIDIduration, volume=100)
                elif(note != notesPred[index-1] and note == notesPred[index+1]):
                    startTimesList.append(times[index])
                    notesList.append(note)
                elif(note == notesPred[index-1] and note != notesPred[index+1]):
                    endTimesList.append(times[index])
                    MIDIpitch = preprocessing.getMIDIfromNote(note)
                    MIDItime = float(startTimesList[-1])
                    MIDIduration = float(times[index]) - MIDItime + 0.01
                    MyMIDI.addNote(track, channel, MIDIpitch, MIDItime, MIDIduration, volume=100)
        
    startTimes = pd.DataFrame(startTimesList)
    endTimes = pd.DataFrame(endTimesList)
    notes = pd.DataFrame(notesList)
    groundTruthClean = pd.concat([startTimes, endTimes, notes], axis=1)  
    groundTruthClean.columns = ["Start Time (s)", "End Time (s)", "Note"]
    fileIsol = preprocessing.extractFileName(audiofilename)
    groundTruthClean.to_csv(audiopath+'/'+fileIsol+'.csv', sep=',')
    with open(audiopath+'/'+fileIsol+".mid", "wb") as outputFile:
        MyMIDI.writeFile(outputFile)
    print("Created groundtruth .csv and MIDI under filename in project filepath")


def predictOutputSingle(model, X):
    probabilities = model.predict(X)
    predictions = probabilities.argmax(axis=-1)
    return predictions


def predictOutputMT(model, X):
    probabilities = model.predict([X, X])[0]
    predictions = probabilities.argmax(axis=-1)
    return predictions


