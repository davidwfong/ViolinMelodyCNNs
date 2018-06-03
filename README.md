DEEP CNNs FOR VIOLIN MELODY EXTRACTION FROM POLYPHONIC MUSIC SIGNALS  - David Weng Wei Fong

Final Year Project submitted in partial fulfillment of requirements for the degree
BEng in Electrical and Electronic Engineering at Imperial College London 

main.py - scripts run for project

chooseRepresentation.py - functions for plotting spectrograms

preprocessing.py - functions for preprocessing of data to be input to CNN

training.py - functions for training Single and Multi-Task CNNs

postprocessing.py - functions for temporal smoothing with HMMs

predicting.py - functions for predicting melody using CNNs

evaluation.py - functions for evaluating melody extraction systems

ViolinMelodyExtractor.py - final product creating estimated violin melody transcription from .wav audio file (use ' ' around input)

predict_on_audio.py and weights folder containing melody2.h5 are borrowed from 
https://github.com/rabitt/ismir2017-deepsalience and used for evaluating the melody extractor

The estimated violin melody from a Brahms Sonata excert in file ViolinSonataDavid.wav is found in:
ViolinRecitalDavid.csv, ViolinRecitalDavid.mid

Package Dependencies (Python 2.7):
- essentia 2.1_beta4
- ffmpeg 3.4.1
- keras 2.1.5 
- librosa 0.6.0
- matplotlib 2.1.2
- MIDIUtil 1.2.1
- numpy 1.14.0
- pandas 0.22.0
- pumpp 0.3.2
- pydub 0.21.0
- scikit-learn 0.19.1

Models: 
- MonoMECNN_1.h5: Single CNN for Violin Melody Extraction from Monophonic Music
- PolyMECNN_1.h5: Single CNN for Violin Melody Extraction from Polyphonic Music
- MTMECNN_1.h5: Multi-Task CNN for Violin Melody Extraction
- MTMECNN_2.h5: Multi-Task CNN for Violin Melody Extraction
- MTMECNN_3.h5: Multi-Task CNN for Violin Melody Extraction (best performing with overall accuracy of 74.94% on Su Dataset VS01 & VS04) 
- MTMECNN_4.h5: Multi-Task CNN for Violin Melody Extraction
- MTMECNN_5.h5: Multi-Task CNN for Violin Melody Extraction