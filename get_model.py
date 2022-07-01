# Copyleft 2016 Forrest Sheng Bao, Wu Jiang, and Stephen Gang Wu
# Released under GNU Affero General Public License Version 3.0
# See http://www.gnu.org/licenses/agpl-3.0.en.html
# 
# For updated version, check our Git repository at 
# https://bitbucket.org/forrestbao/mflux
# or our website
# http://mflux.org
# 
# Cite this work: Wu et al., Rapid prediction of bacterial heterotrophic
# fluxomics using machine learning and constraint programming, 
# PLoS Computational Biology, 2016, DOI: 10.1371/journal.pcbi.1004838
# 
# This file contains functions to extract the training data from spreadsheet,
# building models, grid search, and cross-validation.

from collections import defaultdict
import pickle

import numpy
import random
import json
from regex import W
import re

#from sklearn import cross_validation, preprocessing, grid_search
# old import at Python2 era

from sklearn import model_selection, preprocessing
from numpy import square, mean, sqrt


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class RegressionModel(object):
    """A help class to store model's name and the actual model."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.model = eval(name)(**kwargs)

    def __str__(self):
        return self.name


class RegressionModelFactory(object):
    """A factory to create new instances of RegressionModel."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __str__(self):
        return "{} ({})".format(self.name, self.kwargs)

    def __call__(self):
        """Create a new RegressionModel instance each time."""
        return RegressionModel(self.name, **self.kwargs)

def shuffle_data(Training_data):
    """Shuffle the order of data in training data

    Shuffle by scrambling the index

    (New)_training_data: a dict, keys are EMPs (e.g., v1, v2, etc.),
                   values are 2-tuples (Feature, Label), where
                   Feature is a 2-D list, each sublist is 24-D feature vector for one sample
                   and
                   Label is a 1-D list, labels for all samples.


    """
    New_training_data = {}
    for i, (Feature_vector, Labels) in Training_data.items():
        Num_Samples = len(Feature_vector)
        if Num_Samples != len(Labels):
            print ("Error! Inconsistent numbers of Features Vectors and Labels")
        Shuffled_index = list(range(Num_Samples))
        random.shuffle(Shuffled_index)
        New_Feature_vector = [Feature_vector[j] for j in Shuffled_index]
        New_Label = [Labels[j] for j in Shuffled_index]
        # TODO: new scikit learn can shuffle items directly

        New_training_data[i] = ([New_Feature_vector, New_Label])

    return New_training_data


def read_spreadsheet(filename):
    """Turn spreadsheet into matrixes for training

    Returns
    ========

    training_data: a dict, keys are EMPs (e.g., v1, v2, etc.),
                   values are 2-tuples (Feature, Label), where
                   Feature is a 2-D list, each sublist is 24-D feature vector for one sample
                   and
                   Label is a 1-D list, labels for all samples.

    Notes
    ============
    EMPs are N.A. for some samples, training features were dropped for them.
    That's why we need one training feature matrix for each EMP.

    We have 29 influx values to predict and thus the index/key for training_data goes from 1 to 29

    AA is 26 for 0-index
    BA is 32 for 0-index
    """

    training_data = {}
    for i in range(1, 29+1):# prepare the data structure
        training_data[i] = ([],[]) # the 1st list is the features and the 2nd the labels for i-th influx

    reports = defaultdict(list)
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.strip()
            line = line.split("\t")
            vector = line[2:26+1] # training vector, from Species (C) to Other carbon (AA).
                              # one empty column
            key = ", ".join(vector)
            reports[key].append(i)

            if "" in vector:
                vector.remove("")
            if not vector :
                print (line)
                exit()

            labels = line[26+3: 26+3+26+5] # AD to BF, v1 to v29
#            print Labels

            try:
                vector = list(map(float, vector))
            except ValueError:
                print (vector)

            # Now create the dictionaries we need, one dictionary for each influx
            for i in range(1, 29+1):
                label = labels[i-1]
                try:
                    label = float(label)
                except ValueError:
#                    print Label, "=>"
#                    print Line
                    continue # this label for this influx is not numeric

                training_data[i][0].append(vector) # add a row to feature vectors
                training_data[i][1].append(float(label)) # add one label

    print("checking duplicate lines...", end=" \t ")
    for k, v in reports.items():
        if len(v) > 1:
            print("line number: {}".format(v))
    print("Done.")
    return training_data


def one_hot_encode_features(training_data):
    """Use one-hot encoder to represent categorical features

    Features 1 to 7 are categorical features:
    Species, reactor, nutrient, oxygen, engineering method, MFA and extra energy

    """
    encoded_training_data, encoders = {}, {}
    for vid, (vectors, targets) in training_data.items():
            encoder = preprocessing.OneHotEncoder()
            vectors = numpy.array(vectors) # 2-D array
            encoded_categorical_features = encoder.fit_transform(vectors[:, 0:6+1])
            encoded_categorical_features = encoded_categorical_features.toarray()
            encoded_vectors = numpy.hstack((encoded_categorical_features, vectors[:, 6+1:]))
            encoded_training_data[vid] = (encoded_vectors, targets)
            encoders[vid] = encoder
    return encoded_training_data, encoders

def standardize_features(training_data):
    """Standarize feature vectors for each influx

    Later, a new feature vector X for i-th influx can be normalized as:
    Scalers[i].transform(X)

    """
    std_training_data, scalers = {}, {}
    for vid, (vectors, labels) in training_data.items():
        vectors_scaled = preprocessing.scale(vectors)
        std_training_data[vid] = (vectors_scaled, labels)

        scalers[vid] = preprocessing.StandardScaler().fit(vectors)

    return std_training_data, scalers


def train_model(training_data, Parameters):
    """Train a regression model for each of the 29 influxes

    Returns
    ================
    Models: dict, keys are influx indexes and values are regression models
    parameters: dict, keys are intergers 1 to 29, values are dicts, such as
                'epsilon': 0.01, 'c': 100.0, 'gamma': 0.001, 'kernel': 'rbf'


    Notes
    ===========
    Parameters are not in use. Now use same parameters for all v's.

    """
    models = {}
    for i in range(1, 29+1):
        vectors, label = training_data[i]
        Parameter = Parameters[i]
        model_gen = RegressionModelFactory("SVR", **Parameter)
#        model_gen = RegressionModelFactory("SVR", kernel="linear", C=0.1, epsilon=0.01)
#        model_gen = RegressionModelFactory("KNeighborsRegressor", n_neighbors=10, weights="distance")
        model = model_gen().model
        model.fit(vectors, label) # train the model
        models[i] = model
    return models


def cross_validation_model(training_data, model_gen, Folds, N_jobs):
    """Do a cross validation on a model using the given training data.

    :param training_data: A dict with keys as v, and values as [vectors, label].
    :param model_gen: A RegressionModel generator or a list of that.
    :param Folds: number of CV folds


    """
#    print("model: {}".format(model_gen))
    print("v\tscore_accuracy")
    folds = Folds
    for i in range(1, 29 + 1):
        vectors, label = training_data[i]
        label = numpy.asarray(label)
        if type(model_gen) == dict:
            model = model_gen[i]()
        else:
            model = model_gen()
        # allow shuffleSplit on dataset
#        print len(label)
        if Folds < 1:
            folds = sklearn.cross_validation.ShuffleSplit(len(label))

        scores = model_selection.cross_val_score(model.model, vectors, label,
                                                  cv=folds,
                                                  scoring="mean_squared_error",
                                                  n_jobs = N_jobs
#                                                  scoring="r2"
                                                  )
        print("{}\t{} (+/- {})"
              .format(i, scores.mean(), scores.std() * 2))
# This needs to scaled back to real range of fluxes.

def grid_search_cv(training_data, model_gen, params, SCORINGS, CORE_NUM, FOLDS):
    """Do a grid search to find best params for the given model.

    :param training_data: A dict with keys as v, and values as [vectors, label].
    :param model_gen: A RegressionModel generator.
    :param params: All parameters the grid search needs to find. It's a subset
        of all the optional params on each model. i.e. for KNeighborsRegressor
        model, it's a subset of

        ```
        {
            "n_neighbors": [1, 5, 10, ...],
            "weights": ["uniform", ...],
            "algorithm": ["auto", ...],
            "leaf_size": [30, 50, ...],
            "p": [2, 5, ...],
            "metric": ["minkowski", ...],
            "metric_params": [...],
        }
        ```

    """
    print("model: {}".format(model_gen))
    print("v\tscoring\tbest_score\tbest_params")
    for i in range(1, 29 + 1):
        vectors, label = training_data[i]
        model = model_gen()
        for scoring in SCORINGS:
            clf = model_selection.GridSearchCV(model.model, params, scoring=scoring, n_jobs=CORE_NUM, cv=FOLDS)
            clf.fit(vectors, label)
            print("{}\t{}\t{}\t{}".format(i, scoring, clf.best_score_,
                                          clf.best_params_))

def grid_search_tasks(std_training_data):
    """One function to run grid search on different regressors

    CORE_NUM: int, number of CPU cores to be used
    FOLDS: int, number of folds for cross validate

    """
    knn_model_gen = RegressionModelFactory("KNeighborsRegressor", n_neighbors=10, weights="distance")
    svr_model_gen = RegressionModelFactory("SVR", kernel="linear", C=10, epsilon=0.2)
    dtree_model_gen = RegressionModelFactory("DecisionTreeRegressor", random_state=0)

    KNN_PARAMS = {
        "n_neighbors": range(1, 16),
        "weights": ["distance", "uniform"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "metric": ["euclidean", "chebyshev", "minkowski", ]
    }

    SVR_PARAMS = {
        "C": 10.0 ** numpy.arange(-4,4),
        "epsilon": [0., 0.0001, 0.001, 0.01, 0.1],  # experience: epsilon>=0.1 is not good.
        "kernel": [
       "linear",
#        "rbf",
#        "poly",  # polynomial kernel sucks. Never use it.
#        "sigmoid",
        # "precomputed"
        ],
#        "degree": [5,], # because polynomial kernel sucks. Never use it.
        "gamma": 10.0 ** numpy.arange(-4, 4),
  }

    DTREE_PARAMS = {
#        "criterion": ["mse"],
        "splitter": ["best", "random"],
        "min_samples_split": range(2, 16),
        "min_samples_leaf": range(1, 16),
        "max_features": ["sqrt", "log2"],
#        "random_state": [0, 1, 10, 100],
    }

    SCORINGS = ["mean_squared_error",
#                "mean_absolute_error"
    ]


    TRAINING_PARMAS = [
#        (knn_model_gen, KNN_PARAMS),
        (svr_model_gen, SVR_PARAMS),
#      (dtree_model_gen, DTREE_PARAMS),
    ]

    FOLDS = 10
    CORE_NUM = 8

    [grid_search_cv(std_training_data, k, v, SCORINGS, CORE_NUM, FOLDS) for k, v in TRAINING_PARMAS]

def cv_tasks(std_training_data, Folds, N_jobs, Label_scalers, Parameters):
    """Cross-validation on all v's

    :param Folds: number of CV folds
    :param N_jobs: number of CPU cores
    :param label_scaler: dict, keys are fluxes and values are sklearn scaler objects
    :param Parameters: dict, keys are fluxes and values are parameters for all fluxes

    """
    knn_model_gen = RegressionModelFactory("KNeighborsRegressor", n_neighbors=10, weights="distance")
#    dtree_model_gen = RegressionModelFactory("DecisionTreeRegressor", random_state=0)

    if Parameters != None: # need to create one instances for one flux
        svr_model_gen = {}
        for i in range(1, 29+1):
            svr_model_gen[i] = RegressionModelFactory("SVR", **(Parameters[i]))
    else: # same set of parameters for all SVR models.
        svr_model_gen = RegressionModelFactory("SVR", kernel="linear", C=0.1, epsilon=0.01)

    Classifier_models = [
#        knn_model_gen,
        svr_model_gen,
#        dtree_model_gen,
    ]


    [cross_validation_model(std_training_data, m, Folds, N_jobs) for m in Classifier_models]

def svr_training_test(std_training_data, Parameters, Label_scalers=None):
    """Test SVR training accuracy

    Parameters
    =============
    std_training_data: dict, keys are vID, values are tuples (vector, label)
                       each vector is 2-D array and label is a 1-D array

    Parameters: dict, keys are intergers 1 to 29, values are dicts, such as
                'epsilon': 0.01, 'c': 100.0, 'gamma': 0.001, 'kernel': 'rbf'

    Label_scalers: dict, keys are int 1 to 29, value sare sklearn scaler objects 

    """
    Models = train_model(std_training_data, Parameters)
    Influxes = {}

    for vID, Model in Models.items():
        (Vectors_for_this_v, Label_for_this_v) = std_training_data[vID]
        Label_predict = Model.predict(Vectors_for_this_v)
        if Label_scalers != None:
            Label_predict = Label_scalers[vID].inverse_transform(Label_predict)
            Label_for_this_v = Label_scalers[vID].inverse_transform(Label_for_this_v)

        MSE = Label_predict - Label_for_this_v
#        if vID==2:
#            print Label_predict
#            print Label_for_this_v
#            print MSE
        MSE = sqrt(mean(square(MSE)))

        print ("\t&\t".join(map(str, [vID, MSE
        , max(Label_for_this_v), min(Label_for_this_v)
        ])) + "\t\\\\")
#        for i, j in enumerate(list(MSE)):
#            print i+1, j
#        print list(square(MSE))
#        print Label_predict
#        break

def _validate_training_data(training_data):
    reports = []
    for _, d in training_data.items():
        report = defaultdict(list)
        vectors = d[0]
        for i, v in enumerate(vectors):
            key = ", ".join(map(str, v))
            report[key].append(i)
        # only keep duplicated rows
        report_ = {k: v for k, v in report.items() if len(v) > 1}
        reports.append(report_)

    return reports

def label_std(Training_data, Method="Norm"):
    """standardize the labels in training data
     training_data: a dict, keys are EMPs (e.g., v1, v2, etc.),
                   values are 2-tuples (Feature, Label), where
                   Feature is a 2-D list, each sublist is 24-D feature vector for one sample
                   and
                   Label is a 1-D list, labels for all samples.

    Label_scalers: dict, keys are vIDs and values are sklearn.preprocessing.MinMaxScaler instances for 29 influxes

    sklearn's preprocessing MixMaxScaler does column-wise Minmax scaling.
    Since influxes have different number of intances, we must loop thru the 29.

    """
    import sklearn
    Label_scaled_data = {}
    Label_scalers = {}
    if Method == "None": # No label std needed 
        return Training_data, None

    for vID, (Vector, Label) in Training_data.items():

        Label = numpy.array(Label).reshape(1,-1) # to cope with new Numpy that Label must be 2D 

        if Method == "Norm":
            Label_scaler = sklearn.preprocessing.StandardScaler().fit(Label)
#            Label_scaled = sklearn.preprocessing.scale(Label)  # option 1 of standarization
        elif Method == "MinMax":
            Label_scaler = sklearn.preprocessing.MinMaxScaler().fit(Label)
        else:
             print ("Unrecognized label standarization method ")
        Label_scaled = Label_scaler.transform(Label) # Option 2, MinMax scaler

        Label_scaled = Label_scaled.ravel() # flatten it back to 1D from 2D due to reshape above

        Label_scaled_data[vID] = (Vector, Label_scaled)
        Label_scalers[vID] = Label_scaler

    return Label_scaled_data, Label_scalers

def load_parameters(File):
    """Load a parameter file from grid search print out

    The format of grid search print out:
	checking duplicate lines
	model: SVR ({'epsilon': 0.2, 'C': 10, 'kernel': 'linear'})
	v	scoring	best_score	best_params
	1	mean_squared_error	-0.00462588529703	{'epsilon': 0.01, 'C': 100.0, 'gamma': 0.001, 'kernel': 'rbf'}
	2	mean_squared_error	-0.0103708930608	{'epsilon': 0.01, 'C': 1000.0, 'gamma': 0.0001, 'kernel': 'rbf'}
	3	mean_squared_error	-0.00713773093885	{'epsilon': 0.01, 'C': 1000.0, 'gamma': 0.0001, 'kernel': 'rbf'}
	4	mean_squared_error	-0.0115793576617	{'epsilon': 0.001, 'C': 1000.0, 'gamma': 0.0001, 'kernel': 'rbf'}

    """
    Parameters = {}
    with open(File, 'r') as F:
        F.readline() # Skip first line
        F.readline() # Skip second line
        F.readline() # Skip 3rd line
        for Line in F.readlines():
            [v, _, _, Parameter] = Line.split("\t")
            v = int(v)
            Parameter = json.loads(Parameter.replace("\'", "\""))            

            # exec ("Parameter = " + Parameter)
            # exec changed meaning in Python 3

            Parameters[v] = Parameter

    return Parameters

def test_label_std():
    """Test the accuracy on labels using different label std methods

    The 3 methods are: no std, normalization, MinMax.
    We will study RMSE under different normalization 
   """
    training_data = read_spreadsheet("wild_and_mutant.csv")
    training_data = shuffle_data(training_data)
    encoded_training_data, encoders = one_hot_encode_features(training_data)
    std_training_data, Feature_scalers = standardize_features(encoded_training_data)

    Parameters = load_parameters("svr_both_rbf_shuffle.log")
    
    for Std_method in ["None", "Norm", "MinMax"]:
        final_training_data, Label_scalers = label_std(std_training_data, Method=Std_method)  # standarize the labels/targets as well.

#    grid_search_tasks(std_training_data)
#	    cv_tasks(std_training_data, 10, 32)
        svr_training_test(final_training_data, Parameters, Label_scalers=Label_scalers)

def prepare_data(Datasheet, Parameter_file=None, Label_std_method="MinMax"):
    """Prepare all data including scaling

    Patermeters
    ============
    Datasheet: str, full path to database spreadsheet file
    Parameters_file: str, full path to file that defines best parameters for different v. 
    Label_std_method: str, label preprocessing method, one in ["None", "Norm", "MinMax"] 
    Feature_std_method: str, feature preprocessing method, currentlyl not used

    """
    Training_data = read_spreadsheet("wild_and_mutant.csv")
    Training_data = shuffle_data(Training_data)
    Encoded_training_data, Encoders = one_hot_encode_features(Training_data)
    Std_training_data, Feature_scalers = standardize_features(Encoded_training_data)

    if Parameter_file != None:
        Parameters = load_parameters(Parameter_file)
    else:
        Parameters = None    

    Final_training_data, Label_scalers = label_std(Std_training_data, Method=Label_std_method)  # standarize the labels/targets as well.

    return Final_training_data, Feature_scalers, Label_scalers, Encoders, Parameters

if __name__ == "__main__":

    Datasheet = "wild_and_mutant.csv"
    Parameter_file = "svr_both_rbf_shuffle.log"
    Training_data, Feature_scalers, Label_scalers, Encoders, Parameters\
    = prepare_data(Datasheet, Parameter_file=Parameter_file, Label_std_method="MinMax")

#    grid_search_tasks(std_training_data)
#    cv_tasks(Training_data, 10, 4, Label_scalers, Parameters)


#    reports = _validate_training_data(std_training_data)
#    for i, report in enumerate(reports, 1):
#        print("v = {}, duplicate data index = {}".format(i, report.values()))

    models = train_model(Training_data, Parameters)
    pickle.dump(models, open("models_svm.p", "wb"))
    pickle.dump(Feature_scalers, open("feature_scalers.p", "wb"))
    pickle.dump(Encoders, open("encoders.p", "wb"))
    pickle.dump(Label_scalers, open("label_scalers.p", "wb"))

#    pickle.dump(training_data, open("training_data.p", "wb"))
#    pickle.dump(encoded_training_data, open("encoded_training_data.p", "wb"))
#    pickle.dump(std_training_data,  open("std_training_data.p", "wb"))
