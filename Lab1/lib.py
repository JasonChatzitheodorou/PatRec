from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

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
    return statistics.fmean(pixelOfSamples)



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
    pixelOfSamples = [X[sample].reshape((16, 16))[pixel[0]][pixel[1]] for sample in samplesOfClass]
    return statistics.variance(pixelOfSamples)



def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    
    means = []
    for i in range(16):
        for j in range(16):
            currentPixel = (i, j)
            means.append(digit_mean_at_pixel(X, y, digit, currentPixel))            

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
            variances.append(digit_variance_at_pixel(X, y, digit, currentPixel))

    return np.array(variances)


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    raise NotImplementedError


def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    raise NotImplementedError



class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        raise NotImplementedError
        #return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        raise NotImplementedError

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        raise NotImplementedError


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    raise NotImplementedError

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    raise NotImplementedError



class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        raise NotImplementedError
        #return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        raise NotImplementedError

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        raise NotImplementedError


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        raise NotImplementedError

    def fit(self, X, y):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        train_loader = ...
        val_loader = ...
        # TODO: Train model
        raise NotImplementedError

    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate
        test_loader = ...
        raise NotImplementedError

    def score(self, X, y):
        # Return accuracy score.
        raise NotImplementedError

        
def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError


def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    
    
def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    
def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError    

    

def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    
    

def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError