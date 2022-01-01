import librosa
import numpy as np
import os
import re
from sklearn.base import BaseEstimator, ClassifierMixin

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

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    image = X[index].reshape((16, 16))
    plt.imshow(image)
    plt.axis('off')


def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    fig = plt.figure()
    rows = 2
    columns = 5

    chosenSamples = []
    for i in range(0, 10):
        samplesOfClass = [idx for idx, row in enumerate(X) if y[idx] == i]
        chosenSamples.append(random.choice(samplesOfClass))

    i = 1
    for s in chosenSamples:
        fig.add_subplot(rows, columns, i)
        show_sample(X, s)
        i += 1


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''

    samplesOfClass = [idx for idx, row in enumerate(X) if y[idx] == digit]
    pixelOfSamples = [X[sample].reshape((16, 16))[pixel[0]][pixel[1]] for sample in samplesOfClass]
    return np.mean(pixelOfSamples)


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    samplesOfClass = [idx for idx, row in enumerate(X) if y[idx] == digit]
    pixelOfSamples = [X[sample].reshape(
        (16, 16))[pixel[0]][pixel[1]] for sample in samplesOfClass]
    return np.var(pixelOfSamples)


def digit_mean_at_component(X, y, digit, component):
    samplesOfClass = [idx for idx, row in enumerate(X) if y[idx] == digit]
    componentlOfSamples = [X[sample][component] for sample in samplesOfClass]
    return np.mean(componentlOfSamples)


def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''

    vectorSize = X.shape[1]
    means = [digit_mean_at_component(X, y, digit, i) for i in range(vectorSize)]
    
    #for i in range(16):
    #    for j in range(16):
    #        currentPixel = (i, j)
    #        means.append(digit_mean_at_pixel(X, y, digit, currentPixel))

    return np.array(means)


def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''

    variances = []
    for i in range(16):
        for j in range(16):
            currentPixel = (i, j)
            variances.append(digit_variance_at_pixel(
                X, y, digit, currentPixel))

    return np.array(variances)

def log_gaussian_distribution(x, var, mean):
    """
    If the variance of the characteristic is 0
    it is treated as if it were 1
    """
    var_smoothed = var + 1e-09
    ans = (-0.5 * (x - mean)**2 / var_smoothed) - 0.5 * np.log(var_smoothed * 2 * np.pi)
    
    return ans 

def posteriori_sum(prior, sample, var, mean):
    ans = np.log(prior)
    for idx, x in enumerate(sample):
        ans += log_gaussian_distribution(x, var[idx], mean[idx])
    return ans

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.priors = np.zeros(10)
        self.means = None
        self.variance = None

    def fit(self, X, y):
        """
        Calculate priors for Naive Bayes by dividing 
        number of samples of one class with total samples.

        Also calculates mean and variance for the characteristics
        of the date given.

        fit always returns self.
        """
        
        totalSamples = X.shape[0]
        samplesOfClass = dict((i, []) for i in range(10))

        # Fill dictionary of classes to list of samples
        for idx, x in enumerate(X):
            sampleClass = y[idx]
            samplesOfClass[sampleClass].append(x) 

        # Calculate priors by dividing with total samples
        for i in samplesOfClass:
            sizeOfClass = len(samplesOfClass[i])
            self.priors[i] = sizeOfClass / totalSamples
        
        # Calculate mean and variance 
        # NOTE: Performance of calculation of mean and variance can 
        # be improved by using the dictionary
        self.means = np.zeros((10, X.shape[1]))
        for i in range(10):
            m = digit_mean(X, y, i)
            for j in range(m.shape[0]):
                self.means[i][j] = m[j]
        
        if self.use_unit_variance == True:
            self.variance = np.ones((10, X.shape[1]))
        else:
            self.variance = np.zeros((10, X.shape[1]))
            for i in range(10):
                v = digit_variance(X, y, i)
                for j in range(v.shape[0]):
                    self.variance[i][j] = v[j]

        return self

    def predict(self, X):
        """
        Make predictions assuming Gaussian distribution
        of the characteristics
        """
        predictions = np.zeros(X.shape[0], dtype=int)
        posteriori = np.zeros(10)
        for idx, sample in enumerate(X):
            # Calculate the a-posteriori probabilities of each class
            # by summing the logarithm of the probabilities
            for digit in range(10):
                curr_prior = self.priors[digit]
                curr_var = self.variance[digit]
                curr_mean = self.means[digit]
                posteriori[digit] = posteriori_sum(curr_prior, sample, curr_var, curr_mean)
            
            # Choose the class that is most probable
            temp_sorted = np.flip(np.argsort(np.array(posteriori)))
            for x in temp_sorted:
                if x != 0: 
                    predictions[idx] = x
                    break
            #predictions[idx] = np.argmax(np.array(posteriori))

        
        return predictions


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        raise NotImplementedError
