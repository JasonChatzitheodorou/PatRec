import librosa
import numpy as np
import os
import re

# Read file with librosa 
def read_file(file_name):
    y, sr= librosa.load(file_name, sr=16000)
    return y


def data_parser(directory):
    speakers = []
    digits = []
    wavs = []
    for filename in os.listdir(directory):
        l = re.findall('\d*\D+',filename)
        l[1] = l[1][:-4]
        wavs.append(read_file(directory + filename))
        speakers.append(l[1])
        digits.append(l[0])
    return wavs, speakers, digits

# hop_length might be (window - hop_samples) for some reason 
def calc_mfcc(wav, hop_samples, window_samples, n_mfcc=13, fs=16000):
    kwargs = {'n_mfcc': n_mfcc, 'hop_length': hop_samples, 'n_fft': window_samples}
    return librosa.feature.mfcc(y=wav, sr=fs, **kwargs)


def choose_index(chosen_digit, chosen_speaker, digits, speakers):
    for idx, z in enumerate(zip(digits, speakers)):
        digit = z[0]
        speaker = z[1]
        if digit == chosen_digit and speaker == chosen_speaker:
            return idx


