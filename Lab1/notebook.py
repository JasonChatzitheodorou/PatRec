# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Αναγνώριση προτύπων
# 
# * Χατζηθεοδώρου Ιάσων 03117089
# * Κουνούδης Δημήτρης
# %% [markdown]
# ## Εργαστήριο 1

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import numpy as np
import matplotlib.pyplot as plt
import sklearn, sklearn.metrics, sklearn.model_selection

import lib
import importlib


# %%
# Useful for reloading lib.py when it is changed
importlib.reload(lib)

# %% [markdown]
# ### Προπαρασκευή
# %% [markdown]
# #### Βήμα 1

# %%
def read_data_to_array(file):
  with open(file) as f:
    rawdata = f.readlines()
  data = [list(map(lambda x: float(x), row.split())) for row in rawdata]
  return np.array(data)

train = read_data_to_array('train.txt')
test = read_data_to_array('test.txt')

X_train = train[:, 1:]
X_test = test[:, 1:]
y_train = np.array([int(row[0]) for row in train])
y_test = np.array([int(row[0]) for row in test])

# %% [markdown]
# #### Βήμα 2

# %%
plt.figure(figsize = (2, 2))
lib.show_sample(X_train, 131)
plt.show()

# %% [markdown]
# #### Βήμα 3

# %%
lib.plot_digits_samples(X_train, y_train)
plt.show()

# %% [markdown]
# #### Βήμα 4

# %%
lib.digit_mean_at_pixel(X_train, y_train, 0, (10, 10))

# %% [markdown]
# #### Βήμα 5

# %%
lib.digit_variance_at_pixel(X_train, y_train, 0, (10, 10))

# %% [markdown]
# #### Βήμα 6

# %%
zero_mean = lib.digit_mean(X_train, y_train, 0)
zero_variance = lib.digit_variance(X_train, y_train, 0)

# %% [markdown]
# #### Βήμα 7

# %%
plt.figure(figsize = (2, 2))
plt.imshow(zero_mean.reshape((16, 16)))
plt.axis('off')
plt.show()

# %% [markdown]
# #### Βήμα 8

# %%
plt.figure(figsize = (2, 2))
plt.imshow(zero_variance.reshape((16, 16)))
plt.axis('off')
plt.show()

# %% [markdown]
# Παρατηρούμε ότι στην περίπτωση του variance διαχωρίζεται η γραμμή του ψηφίου 0 σε δύο μέρη. Αυτό συμβαίνει διότι τα περισσότερα δείγματα του συγκεκριμένου ψηφίου σε εκείνο το κομμάτι συμφωνούν άρα το variance είναι πολύ χαμηλό, ενώ αντίθετα διαφωνούν στο πού ακριβώς είναι τα όρια των γραμμών, για αυτό και είναι πιό έντονο το χρώμα εκεί που θα περιμέναμε να είναι το τέλος της γραμμής.
# %% [markdown]
# #### Βήμα 9

# %%
means = []
variances = []
for i in range(10):
    means.append(lib.digit_mean(X_train, y_train, i))
    variances.append(lib.digit_mean(X_train, y_train, i))


# %%
fig = plt.figure()

i = 1
for x in means:
    fig.add_subplot(2, 5, i)
    plt.imshow(x.reshape((16, 16)))
    plt.axis('off')
    i += 1

plt.show()

# %% [markdown]
# #### Βήμα 10

# %%
plt.figure(figsize = (2, 2))
lib.show_sample(X_test, 101)
plt.show()


# %%
print("The prediction is {}".format(lib.euclidean_distance_classifier(np.array(X_test[101]), means)[0]))
print("The correct answer is {}".format(y_test[101]))

# %% [markdown]
# Το ψηφίο θα έπρεπε να αναγνωριστεί ως 6, όμως δεν είναι ξεκάθαρο ούτε σε άνθρωπο αυτό, επομένως είναι λογικό να αποτύχει ο euclidean classifier
# %% [markdown]
# #### Βήμα 11

# %%
predictions = lib.euclidean_distance_classifier(X_test, means)

print("The accuracy is {}".format(sklearn.metrics.accuracy_score(y_test, predictions)))

# %% [markdown]
# #### Βήμα 12

# %%
# The solution is in lib.py

# %% [markdown]
# #### Βήμα 13

# %%
# Evaluate classifier
score = lib.evaluate_euclidean_classifier(X_train, y_train, folds=5)


# %%
print("The 5-fold accuracy score of the classifier is {}".format(score))


# %%
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
eucl_clf_pca = lib.EuclideanDistanceClassifier()
eucl_clf_pca.fit(X_pca, y_train)

scatter_kwargs = {'s': 40, 'edgecolor': 'k', 'alpha': 0.7}
plot_decision_regions(X_pca, y_train, clf=eucl_clf_pca, 
                      markers="o",
                      scatter_kwargs=scatter_kwargs)
plt.show()


# %%
# In order to plot learning curve use sklearn.learning_curve 
train_sizes, train_scores, test_scores = sklearn.model_selection.learning_curve(
    lib.EuclideanDistanceClassifier(), X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1.0, 5))


# %%
# This cell is from https://github.com/slp-ntua/python-lab/blob/master/Lab%200.3%20Scikit-learn.ipynb

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure(figsize=(5, 3))
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.6, 1))
plt.show()

# %% [markdown]
# ### Εργαστηριακό μέρος
# %% [markdown]
# #### Βήμα 14

# %%
priors = lib.calculate_priors(X_train, y_train)

for idx, x in enumerate(priors):
    print("The a-priori probability of class {} is {:.5f}".format(idx, x))

# %% [markdown]
# #### Βήμα 15

# %%
importlib.reload(lib)
NB_clf = lib.CustomNBClassifier()
NB_clf.fit(X_train, y_train)

# NOTE: Variance smoothing applied is 1e-09
predictions = NB_clf.predict(X_test)
print("The accuracy of the custom implementation is {}".format(sklearn.metrics.accuracy_score(y_test, predictions)))


# %%
from sklearn.naive_bayes import GaussianNB

for smoothing in [1e-9, 1e-5, 1e-2, 0.1, 0.5, 2]:
    sklearn_NB_clf = GaussianNB(var_smoothing=smoothing)
    sklearn_NB_clf.fit(X_train, y_train)
    sklearn_predictions = sklearn_NB_clf.predict(X_test)
    print("The accuracy of the sklearn implementation for var_smoothing={} is {:.5f}".format(smoothing, sklearn.metrics.accuracy_score(y_test, sklearn_predictions)))

# %% [markdown]
# Το accuracy που έβγαλε η δικιά μας υλοποίηση ήταν πρακτικά ίδιο με τον default GaussianNB (`var_smoothing=1e-09`). Στη δικιά μας υλοποίηση προσθέταμε παντού τον ίδιο αριθμό `1e-09` για να επιτύχουμε variance smoothing. 
# 
# Αντίθετα η υλοποίηση του scikit learn προσθέτει ένα συγκεκριμένο ποσοστό της μεγαλύτερης απόκλισης σε όλες τις αποκλίσεις, το οποίο μπορεί να αλλάξει ο χρήστης για να βρει το βέλτιστο. Όπως φαίνεται στην συγκεκριμένη περίπτωση το `var_smoothing=0.1` έχει αρκετά καλό accuracy.
# %% [markdown]
# #### Βήμα 16

# %%
# All variances are 1
NB_clf = lib.CustomNBClassifier(use_unit_variance=True)
NB_clf.fit(X_train, y_train)

predictions = NB_clf.predict(X_test)
print("The accuracy of the custom implementation with unit variances is {}".format(sklearn.metrics.accuracy_score(y_test, predictions)))

# %% [markdown]
# #### Βήμα 17

# %%
evaluators = [  ('Linear SVM', lib.evaluate_linear_svm_classifier), 
                ('RBF SVM', lib.evaluate_rbf_svm_classifier),
                ('5-Nearest Neighbors', lib.evaluate_knn_classifier), 
                ('Naive Bayes', lib.evaluate_sklearn_nb_classifier)
             ]
    
for ev in evaluators:
    print("The 5-fold cross validation accuracy of {} is {:.5f}".format(ev[0], ev[1](X_train, y_train)))

# %% [markdown]
# #### Βήμα 18

# %%
# NOTE: Check what kind of mistakes each classifier makes
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

linear_svm = SVC(kernel='linear')
gaussian_NB = GaussianNB(var_smoothing=0.1)
k_neighbors = KNeighborsClassifier()

print("Linear SVM mispredictions per class are {}".format(lib.classifier_mistakes_per_class(linear_svm, X_train, y_train, X_test, y_test)))
print("Gaussian Naive Bayes mispredictions per class are {}".format(lib.classifier_mistakes_per_class(gaussian_NB, X_train, y_train, X_test, y_test)))
print("5-Nearest Neighbor mispredictions per class are {}".format(lib.classifier_mistakes_per_class(k_neighbors, X_train, y_train, X_test, y_test)))

# %% [markdown]
# Όπως φαίνεται οι παραπάνω ταξινομητές δεν κάνουν τα ίδια λάθη, αν και υπάρχουν κάποιες ομοιότητες στην κατανομή λαθών, οπότε ο συνδυασμός τους με voting classifier μάλλον θα βελτιώσει την ακρίβεια των προβλέψεων

# %%
print("The 5-fold cross validation accuracy of the voting classifier with hard voting is {:.5f}".format(lib.evaluate_voting_classifier(X_train, y_train)))


# %%
print("The 5-fold cross validation accuracy of the bagging classifier using Gaussian Naive Bayes is {:.5f}".format(lib.evaluate_bagging_classifier(X_train, y_train)))

# %% [markdown]
# Η χρήση του bagging classifier δεν φαίνεται να επηρεάζει σημαντικά το accuracy του Naive Bayes παρόλο που το βελτιώνει λίγο
# %% [markdown]
# #### Βήμα 19

# %%
from torch.utils.data import DataLoader

train_data = lib.BlobData(X_train, y_train)
test_data = lib.BlobData(X_test, y_test)

BATCH_SZ = 32
train_dl = DataLoader(train_data, batch_size=BATCH_SZ, shuffle=True)
test_dl = DataLoader(test_data, batch_size=BATCH_SZ, shuffle=True)


# %%
importlib.reload(lib)

nn_layers = [[], [100], [100, 50], [400], [400, 100, 50]]

score_list = []
for layers in nn_layers:
    nfeat = X_train.shape[1]
    nclass = len(set(y_train))
    learning_r = 1e-2
    epochs = 20

    clf = lib.PytorchNNModel(nfeat, nclass, layers, learning_r, epochs)
    clf.fit(X_train, y_train)
    score_list.append(clf.score(X_test, y_test))


# %%
for idx, s in enumerate(score_list):
    print("The accuracy with layers {} is {:.5f}".format([256, *(nn_layers[idx]), 10], s))


