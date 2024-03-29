# Deep CNNs for Violin Melody Extraction from Polyphonic Music Signals 
#### David W. Fong

*Final Year Project submitted in partial fulfillment of the requirements for the degree
BEng in Electrical and Electronic Engineering at Imperial College London* 

### Modules:
- main.py - scripts run for project
- chooseRepresentation.py - functions for plotting spectrograms
- preprocessing.py - functions for preprocessing of data to be input to CNN
- training.py - functions for training Single and Multi-Task CNNs
- postprocessing.py - functions for temporal smoothing with HMMs
- predicting.py - functions for predicting melody using CNNs
- evaluation.py - functions for evaluating melody extraction systems
- ViolinMelodyExtractor.py - final product creating estimated violin melody transcription from .wav audio file
- predict_on_audio.py and weights folder containing melody2.h5 are borrowed from 
https://github.com/rabitt/ismir2017-deepsalience and used for evaluating the melody extractor

### Package Dependencies (Python 2.7):
- essentia 2.1_beta4
- ffmpeg 3.4.1
- keras 2.1.5 
- librosa 0.6.0
- matplotlib 2.1.2
- MIDIUtil 1.2.1
- mir_eval 0.4
- numpy 1.14.0
- pandas 0.22.0
- pumpp 0.3.2
- pydub 0.21.0
- scikit-learn 0.19.1

### CNN Models: 
- MonoMECNN_1.h5: Single CNN for Violin Melody Extraction from Monophonic Music Signals
- PolyMECNN_1.h5: Single CNN for Violin Melody Extraction from Polyphonic Music Signals
- MTMECNN_1.h5: Multi-Task CNN 1 for Violin Melody Extraction from Polyphonic Music Signals
- MTMECNN_2.h5: Multi-Task CNN 2 for Violin Melody Extraction from Polyphonic Music Signals
- MTMECNN_3.h5: Multi-Task CNN 3 for Violin Melody Extraction from Polyphonic Music Signals (Recommended) 
- MTMECNN_4.h5: Multi-Task CNN 4 for Violin Melody Extraction from Polyphonic Music Signals
- MTMECNN_5.h5: Multi-Task CNN 5 for Violin Melody Extraction from Polyphonic Music Signals

### Demo Files:
- ViolinSonataDavid.wav: Audio Excerpt from Brahms Sonata No.2 in A Major, Movement 1
- ViolinRecitalDavid.csv: Estimated Annotation in CSV format 
- ViolinRecitalDavid.mid: Estimated Annotation in MIDI format
