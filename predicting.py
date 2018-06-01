#IMPORT RELEVANT MODULES
import pandas as pd
import numpy as np
import preprocessing
#-------------------------------------------------------------------------------------------------------
#FUNCTIONS

def createGT(yPred, frameduration, audiopath, audiofilename, noteMin):
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
                elif(note != notesPred[index-1] and note == notesPred[index+1]):
                    startTimesList.append(times[index])
                    notesList.append(note)
                elif(note == notesPred[index-1] and note != notesPred[index+1]):
                    endTimesList.append(times[index])
        
    startTimes = pd.DataFrame(startTimesList)
    endTimes = pd.DataFrame(endTimesList)
    notes = pd.DataFrame(notesList)
    groundTruthClean = pd.concat([startTimes, endTimes, notes], axis=1)  
    groundTruthClean.columns = ["Start Time (s)", "End Time (s)", "Note"]
    fileIsol = preprocessing.extractFileName(audiofilename)
    groundTruthClean.to_csv(audiopath+'/'+fileIsol+'.csv', sep=',')
    print("Created groundtruth under filename in project filepath")


def predictOutputSingle(model, X):
    probabilities = model.predict(X)
    predictions = probabilities.argmax(axis=-1)
    return predictions


def predictOutputMT(model, X):
    probabilities = model.predict([X, X])[0]
    predictions = probabilities.argmax(axis=-1)
    return predictions
