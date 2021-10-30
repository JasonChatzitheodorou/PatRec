from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn, sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import random


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


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    return np.linalg.norm(s - m)


def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    predictions = []
    for x in X:
        distances = [euclidean_distance(x, m) for m in X_mean]
        predictions.append(np.argmin(distances))
    
    return np.array(predictions)


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
        means = [digit_mean(X, y, digit) for digit in range(10)]
        self.X_mean_ = np.array(means)
        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        return euclidean_distance_classifier(X, self.X_mean_)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        return sklearn.metrics.accuracy_score(y, self.predict(X))


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    # List that holds scores
    scores = []

    # Split the arrays in 5 equal parts
    X_folds = np.array_split(X, folds, axis=0)
    y_folds = np.array_split(y, folds, axis= 0)

    for fold in range(folds): 
        # Select data for testing by choosing the correct fold
        X_test = X_folds[fold]
        y_test = y_folds[fold]

        # Select data for training by concatenating the rest of the folds
        X_train = np.concatenate([x for idx, x in enumerate(X_folds) if idx != fold])
        y_train = np.concatenate([x for idx, x in enumerate(y_folds) if idx != fold])

        # Train the estimator
        clf.fit(X_train, y_train)

        # Get a score for current fold and append it
        scores.append(clf.score(X_test, y_test))

    # Return the mean of the errors
    return np.mean(scores)


def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    priors = []
    totalSamples = y.size
    samplesOfClass = np.zeros(10)
    
    for sampleClass in y:
        samplesOfClass[sampleClass] += 1 
    
    # Calculate priors by dividing with total samples
    for s in samplesOfClass:
        priors.append(s / totalSamples)

    return np.array(priors)

def log_gaussian_distribution(x, var, mean):
    """
    If the variance of the characteristic is 0
    it is treated as if it were 1
    """
    ans = 0
    if(var == 0): # Assume variance is 1
        ans = (-0.5 * (x - mean)**2) - 0.5 * np.log(2 * np.pi)
    else:
        ans = (-0.5 * (x - mean)**2 / var) - 0.5 * np.log(var * 2 * np.pi)
    
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
            predictions[idx] = np.argmax(np.array(posteriori))
        
        return predictions


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
    euclClassifier = EuclideanDistanceClassifier()
    return evaluate_classifier(euclClassifier, X, y, folds)


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
