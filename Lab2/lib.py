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
