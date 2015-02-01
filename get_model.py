# extract the training data from spreadsheet

from collections import defaultdict
import cPickle

import numpy
from sklearn import cross_validation, preprocessing, grid_search
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

    We have 29 influxes values to predict and thus the index/key for training_data goes from 1 to 29

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
            vector = line[2:26+1] # training vector, from Purpose (C) to Other carbon (AA).
                              # one empty column
            key = ", ".join(vector)
            reports[key].append(i)

            if "" in vector:
                vector.remove("")
            if not vector :
                print line
                exit()

            labels = line[26+3: 26+3+26+5] # AD to BF, v1 to v29
#            print Labels

            try:
                vector = map(float, vector)
            except ValueError:
                print vector

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

    print("duplicate lines")
    for k, v in reports.iteritems():
        if len(v) > 1:
            print("line number: {}".format(v))
    return training_data


def one_hot_encode_features(training_data):
    """Use one-hot encoder to represent categorical features

    Feature from 1 to 7 are categorical features:
    Species, reactor, nutrient, oxygen, engineering method, MFA and extra energy

    """
    encoded_training_data, encoders = {}, {}
    for vid, (vectors, targets) in training_data.iteritems():
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
    for vid, (vectors, labels) in training_data.iteritems():
        vectors_scaled = preprocessing.scale(vectors)
        std_training_data[vid] = ( vectors, labels)

        scalers[vid] = preprocessing.StandardScaler().fit(vectors)

    return std_training_data, scalers


def train_model(training_data):
    """Train a regression model for each of the 29 influxes

    Returns
    ================
    Models: dict, keys are influx indexes and values are regression models

    """
    models = {}
    for i in range(1, 29+1):
        vectors, label = training_data[i]
        model_gen = RegressionModelFactory("SVR", kernel="linear", C=10, epsilon=0.2)
#        model_gen = RegressionModelFactory("KNeighborsRegressor", n_neighbors=10, weights="distance")
        model = model_gen().model
        model.fit(vectors, label) # train the model
        models[i] = model
    return models


def cross_validation_model(training_data, model_gen):
    """Do a cross validation on a model using the given training data.

    :param training_data: A dict with keys as v, and values as [vectors, label].
    :param model_gen: A RegressionModel generator.

    """
    print("model: {}".format(model_gen))
    print("v\tscore_accuracy")
    for i in range(1, 29 + 1):
        vectors, label = training_data[i]
        label = numpy.asarray(label)
        model = model_gen()
        scores = cross_validation.cross_val_score(model.model, vectors, label,
                                                  cv=4)
        print("{}\t{} (+/- {})"
              .format(i, scores.mean(), scores.std() * 2))


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
            clf = grid_search.GridSearchCV(model.model, params, scoring=scoring, n_jobs=CORE_NUM, cv=FOLDS)
            clf.fit(vectors, label)
            print("{}\t{}\t{}\t{}".format(i, scoring, clf.best_score_,
                                          clf.best_params_))

def grid_search_tasks(std_training_data):
    """One function to run grid search on different regressors

    CORE_NUM: int, number of CPU cores to be used
    FOLDS: int, number of folds for cross validate

    """
    import numpy
    knn_model_gen = RegressionModelFactory("KNeighborsRegressor", n_neighbors=10, weights="distance")
    svr_model_gen = RegressionModelFactory("SVR", kernel="rbf", C=10, epsilon=0.2)
    dtree_model_gen = RegressionModelFactory("DecisionTreeRegressor", random_state=0)

    KNN_PARAMS = {
        "n_neighbors": [1, 2, 3, 4, 5, 10],
        "weights": ["distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    }

    SVR_PARAMS = {
        "C": 10.0 ** numpy.arange(-3,5),
        "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        "kernel": [
        "linear",
        "rbf",
        "poly",
        "sigmoid",
        # "precomputed"
        ],
        "degree": [3, 5, 10],
        "gamma": 10.0 ** numpy.arange(-5,5),
        "random_state": [0, 1, 10, 100],
    }

    DTREE_PARAMS = {
        "criterion": ["mse"],
        "splitter": ["best"],
        "min_samples_split": [2, 3, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["auto", "sqrt", "log2"],
        "random_state": [0, 1, 10, 100],
    }

    SCORINGS = ["mean_squared_error",
#                "mean_absolute_error"
    ]

    training_models = [
        knn_model_gen,
        svr_model_gen,
        dtree_model_gen,
    ]

    TRAINING_PARMAS = [
        (knn_model_gen, KNN_PARAMS),
#        (svr_model_gen, SVR_PARAMS),
#      (dtree_model_gen, DTREE_PARAMS),
    ]

    FOLDS = 10
    CORE_NUM = 1

    [grid_search_cv(std_training_data, k, v, SCORINGS, CORE_NUM, FOLDS) for k, v in TRAINING_PARMAS]

def svr_test(std_training_data):
    """Test SVR training accuracy

    Parameters
    =============
        std_training_data: dict, keys are vID, values are tuples (vector, label) 
                            each vector is 2-D array and label is a 1-D array

    """


    from numpy import square, mean, sqrt
    Models = train_model(std_training_data)
    print len(Models)
    Influxes = {}

    for vID, Model in Models.iteritems():
        (Vectors_for_this_v, Label_for_this_v) = std_training_data[vID]
        Label_predict = Model.predict(Vectors_for_this_v)
        MSE = Label_predict - Label_for_this_v
        MSE = sqrt(mean(square(MSE)))
    
        print vID, MSE
#        for i, j in enumerate(list(MSE)):
#            print i+1, j 
#        print list(square(MSE))
#        print Label_predict
#        break

def _validate_training_data(training_data):
    reports = []
    for _, d in training_data.iteritems():
        report = defaultdict(list)
        vectors = d[0]
        for i, v in enumerate(vectors):
            key = ", ".join(map(str, v))
            report[key].append(i)
        # only keep duplicated rows
        report_ = {k: v for k, v in report.iteritems() if len(v) > 1}
        reports.append(report_)

    return reports


if __name__ == "__main__":
    training_data = read_spreadsheet("wild_type.csv")
    encoded_training_data, encoders = one_hot_encode_features(training_data)
    std_training_data, scalers = standardize_features(encoded_training_data)
    svr_test(std_training_data)

#    reports = _validate_training_data(std_training_data)
#    for i, report in enumerate(reports, 1):
#        print("v = {}, duplicate data index = {}".format(i, report.values()))

#    [cross_validation_model(std_training_data, m) for m in training_models]

#    models = train_model(std_training_data)
#    cPickle.dump(models, open("models_knn.p", "wb"))
#    cPickle.dump(scalers, open("scalers.p", "wb"))
#    cPickle.dump(encoders, open("encoders.p", "wb"))
#    cPickle.dump(training_data, open("training_data.p", "wb"))
