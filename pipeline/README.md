# Pipeline Module

### wav_to_spec.py
Script to transform wav files into mel spectrograms
````bash
python wav_to_spec.py [--params length_sec,stride,n_fft,hop_length,n_mels] INPUT OUTPUT
````
INPUT: directory path containing the wav (can contain sub-directories)\
OUPUT: directory path to save the spectrograms. Re-create the folder structure of INPUT

length_sec: length in seconds of the spectrograms (5 by default)\
stride: stride between each spectrograms. If None: stride = rate * length_sec\
n_fft: window size of the FFT (default 2048)\
hop_length: jump between each window (default 512)\
n_mels: number of mel frequencies (default 128)


### preprocessing.py
Set of functions to:
- load wav files and spectrograms 
- Create tensorflow dataset
- Compute spectrograms