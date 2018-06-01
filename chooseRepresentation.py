#IMPORT RELEVANT MODULES
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from pydub import AudioSegment
import librosa
import librosa.display as display
import pumpp
#-----------------------------------------------------------------------------------------------
#FUNCTIONS

def concatenateAudio(filepath):
    audioStream = AudioSegment.empty()
    for filename in glob.glob(os.path.join(filepath, '*.mp3')):
        audio = AudioSegment.from_file(filename,format="mp3")
        audioStream = audioStream + audio
    newFile = filepath + '/violin_file.wav'
    audioStream.export(newFile, format="wav")
    return newFile


def convertWavToArray(audioFile):
    data, fs = librosa.load(audioFile, sr=44100, mono=True) 
    return data, fs

def convertWavToArrayNew(audioFile):
    data, fs = librosa.load(audioFile, sr=16000, mono=True) 
    return data, fs


def plotSpectrograms(data,fs):
    note_G3 = librosa.core.note_to_hz('G3')
    S_Mel = librosa.feature.melspectrogram(y=data, sr=fs, 
                                           n_mels=192, fmin=note_G3, 
                                           n_fft=512, hop_length=int(0.5*512), 
                                           power=1.0)
    SNorm_Mel = sklearn.preprocessing.robust_scale(S_Mel, axis=1)
    plt.figure(1)
    plt.figure(figsize=(12, 7))
    display.specshow(librosa.amplitude_to_db(SNorm_Mel, ref=np.max),
                     sr=fs, hop_length=256,
                     y_axis='cqt_note', x_axis='time',
                     fmin=note_G3, bins_per_octave=48)
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Note',fontsize=14)
    plt.xticks(np.arange(0, 26, 1),fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(format = '%+2.0f dB')
    #plt.title('Mel Spectrogram for Violin')
    plt.tight_layout()
    plt.savefig('Violin_Mel' + '.png')
    
    S_STFT = np.abs(librosa.core.stft(y=data, 
                                      n_fft=512, hop_length=int(0.5*512), 
                                      window='hann'))
    SNorm_STFT = sklearn.preprocessing.robust_scale(S_STFT, axis=1)
    SNorm_STFT = SNorm_STFT[65:257,:]
    plt.figure(2)
    plt.figure(figsize=(12,7))
    display.specshow(librosa.amplitude_to_db(SNorm_STFT, ref=np.max), 
                     sr=fs, hop_length=256,
                     y_axis='cqt_note', x_axis='time', 
                     fmin=note_G3, bins_per_octave=48)
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Note',fontsize=14)
    plt.xticks(np.arange(0, 26, 1),fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(format = '%+2.0f dB')
    #plt.title('STFT Spectrogram for Violin')
    plt.tight_layout()
    plt.savefig('Violin_STFT' + '.png')
    
    S_CQT = np.abs(librosa.core.cqt(y=data, sr=fs,
                                    hop_length=256, 
                                    fmin=note_G3, n_bins=48*4, bins_per_octave=48, 
                                    window='hann'))
    SNorm_CQT = sklearn.preprocessing.robust_scale(S_CQT, axis=1)
    plt.figure(3)
    plt.figure(figsize=(12, 7))
    display.specshow(librosa.amplitude_to_db(SNorm_CQT, ref=np.max),
                     sr=fs, hop_length=256,
                     y_axis='cqt_note',x_axis='time',
                     fmin=note_G3, bins_per_octave=48)
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Note',fontsize=14)
    plt.xticks(np.arange(0, 26, 1),fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(format = '%+2.0f dB')
    #plt.title('CQT spectrogram for Violin')
    plt.tight_layout()
    plt.savefig('Violin_CQT' + '.png')
    
    harmonics = [1,2,3,4]
    P_HCQT = pumpp.feature.HCQTMag(name='HCQT', sr=fs, 
                      hop_length=256, 
                      n_octaves=4, over_sample=4, 
                      fmin=note_G3, harmonics=harmonics, 
                      log=False, conv='channels_last')
    SDict_HCQT = P_HCQT.transform(y=data,sr=fs)
    S4d_HCQT = SDict_HCQT.values()[0]
    numharmonics_HCQT = np.size(S4d_HCQT, axis=3)
    S3d_HCQT = S4d_HCQT[0,:,:,:]
    SFinal_HCQT = np.transpose(S3d_HCQT,(1,0,2))
    S_Shape = SFinal_HCQT.shape
    SNorm_HCQT = np.empty(S_Shape)
    for i in range(numharmonics_HCQT):
        SNorm_HCQT[:,:,i] = sklearn.preprocessing.robust_scale(SFinal_HCQT[:,:,i], axis=1)
    plt.figure(4)
    plt.figure(figsize=(16, 12))
    for i in range(numharmonics_HCQT):
        S_harmonic = SNorm_HCQT[:,:,i]
        plt.subplot(2,2,(i+1))
        display.specshow(librosa.amplitude_to_db(S_harmonic, ref=np.max),
                         sr=fs, hop_length=256,
                         y_axis='cqt_note',x_axis='time',
                         fmin=note_G3, bins_per_octave=48)
        plt.xlabel('Time (s)',fontsize=14)
        plt.ylabel('Note',fontsize=14)
        plt.xticks(np.arange(0, 26, 1),fontsize=12)
        plt.yticks(fontsize=12)
        plt.colorbar(format = '%+2.0f dB')
        #plt.title('HCQT spectrogram for Violin ' + 'of harmonic ' + str(i+1))
        plt.tight_layout()
    plt.savefig('Violin_HCQT' + '.png')
      
    return SNorm_Mel, SNorm_STFT, SNorm_CQT, SNorm_HCQT
 
    
def plotHCQT(data,fs=16000):
    note_G3 = librosa.core.note_to_hz('G3')
    harmonics = [1,2]
    P_HCQT = pumpp.feature.HCQTMag(name='HCQT', sr=fs, 
                      hop_length=160, 
                      n_octaves=4, over_sample=4, 
                      fmin=note_G3, harmonics=harmonics, 
                      log=False, conv='channels_last')
    SDict_HCQT = P_HCQT.transform(y=data,sr=fs)
    S4d_HCQT = SDict_HCQT.values()[0]
    numharmonics_HCQT = np.size(S4d_HCQT, axis=3)
    S3d_HCQT = S4d_HCQT[0,:,:,:]
    SFinal_HCQT = np.transpose(S3d_HCQT,(1,0,2))
    S_Shape = SFinal_HCQT.shape
    SNorm_HCQT = np.empty(S_Shape)
    for i in range(numharmonics_HCQT):
        SNorm_HCQT[:,:,i] = sklearn.preprocessing.robust_scale(SFinal_HCQT[:,:,i], axis=1)

    plt.figure(1)
    plt.figure(figsize=(16, 12))
    for i in range(numharmonics_HCQT):
        S_harmonic = SNorm_HCQT[:,:,i]
        plt.subplot(2,1,(i+1))
        display.specshow(librosa.amplitude_to_db(S_harmonic, ref=np.max),
                         sr=fs, hop_length=160,
                         y_axis='cqt_note',x_axis='time',
                         fmin=note_G3, bins_per_octave=48)
        plt.xlabel('Time (s)',fontsize=14)
        plt.ylabel('Note',fontsize=14)
        plt.yticks(fontsize=12)
        plt.colorbar(format = '%+2.0f dB')
        #plt.title('HCQT spectrogram for Violin harmonic ' + str(i+1) + ' for VS01 excerpt in Su Dataset')
        plt.tight_layout()
    plt.savefig('Violin_finalHCQT' + '.png')
    return SNorm_HCQT    


def plotInputFeature(data,fs=16000):
    note_G3 = librosa.core.note_to_hz('G3')
    harmonics = [1,2]
    P_HCQT = pumpp.feature.HCQTMag(name='HCQT', sr=fs, 
                      hop_length=160, 
                      n_octaves=4, over_sample=4, 
                      fmin=note_G3, harmonics=harmonics, 
                      log=False, conv='channels_last')
    SDict_HCQT = P_HCQT.transform(y=data,sr=fs)
    S4d_HCQT = SDict_HCQT.values()[0]
    numharmonics_HCQT = np.size(S4d_HCQT, axis=3)
    S3d_HCQT = S4d_HCQT[0,:,:,:]
    SFinal_HCQT = np.transpose(S3d_HCQT,(1,0,2))
    S_Shape = SFinal_HCQT.shape
    SNorm_HCQT = np.empty(S_Shape)
    for i in range(numharmonics_HCQT):
        SNorm_HCQT[:,:,i] = sklearn.preprocessing.robust_scale(SFinal_HCQT[:,:,i], axis=1)
    SNorm_HCQT = SNorm_HCQT[:,1000:1011,:]
    S_harmonic = SNorm_HCQT[:,:,0]
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('xtick', labelsize=14)
    plt.figure(1)
    plt.figure(figsize=(10, 20))
    display.specshow(librosa.amplitude_to_db(S_harmonic, ref=np.max),
                         sr=fs, hop_length=160,
                         y_axis='cqt_note',x_axis='time',
                         fmin=note_G3, bins_per_octave=48)
    plt.colorbar(format = '%+2.0f dB')
    plt.xlabel('Time (s)',fontsize=14)
    plt.xticks(np.arange(0, 0.12, 0.01))
    plt.ylabel('Note',fontsize=20)
    #plt.title('Input Feature for Violin harmonic 1 for VS01 excerpt in Su Dataset',fontsize = 20)
    plt.tight_layout()
    plt.savefig('Violin_featureHCQT' + '.png')
    return SNorm_HCQT    




