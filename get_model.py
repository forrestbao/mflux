# extract the training data from spreadsheet

import cPickle

import numpy
from sklearn import cross_validation, preprocessing, grid_search
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class RegressionModel(object):
    """A help class to store model's name and the actual model.
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.model = eval(name)(**kwargs)

    def __str__(self):
        return self.name


KNN_MODEL = RegressionModel("KNeighborsRegressor", n_neighbors=5,
                            weights="distance")
KNN_PARAMS = {
    "n_neighbors": [1, 5, 10],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}
SVR_MODEL = RegressionModel("SVR", kernel="linear", C=1)
SVR_PARAMS = {
    "C": [1, 10, 100, 1000],
    "epsilon": [0.1, 0.5],
    # "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
}

def read_spreadsheet(File):
    """Turn spreadsheet into matrixes for training

    Returns
    ========

	Training_data: a dict, keys are EMPs (e.g., v1, v2, etc.),
                   values are 2-tuples (Feature, Label), where
                   Feature is a 2-D list, each sublist is 24-D feature vector for one sample
                   and
                   Label is a 1-D list, labels for all samples.


    Notes
    ============
    EMPs are N.A. for some samples, training features were dropped for them.
    That's why we need one training feature matrix for each EMP.

    We have 29 influxes values to predict and thus the index/key for Training_data goes from 1 to 29

    AA is 26 for 0-index
    BA is 32 for 0-index
    """

    Training_data = {}
    for i in range(1, 29+1):# prepare the data structure
        Training_data[i] = ([],[]) # the 1st list is the features and the 2nd the labels for i-th influx

    with open(File, 'r') as F:
        for Line in F.readlines():
            Line = Line.strip()
            Line = Line.split("\t")
            Vector = Line[2:26+1] # training vector, from Purpose (C) to Other carbon (AA).
                              # one empty column

            if "" in Vector:
                Vector.remove("")
            if not Vector :
                print Line
                exit()

            Labels = Line[26+3: 26+3+26+5] # AD to BF, v1 to v29
#            print Labels

            try:
                Vector = map(float, Vector)
            except ValueError:
                print Vector

            # Now create the dictionaries we need, one dictionary for each influx
            for i in range(1, 29+1):
                Label = Labels[i-1]
                try:
                    Label = float(Label)
                except ValueError:
#                    print Label, "=>"
#                    print Line
                    continue # this label for this influx is not numeric

                Training_data[i][0].append(Vector) # add a row to feature vectors
                Training_data[i][1].append(float(Label)) # add one label

    return Training_data


def one_hot_encode_features(Training_data):
    """Use one-hot encoder to represent categorical features

    Feature from 1 to 7 are categorical features:
    Species, reactor, nutrient, oxygen, engineering method, MFA and extra energy

    """
    Encoded_training_data, Encoders = {}, {}
    for vID, (Vectors, Targets) in Training_data.iteritems():
            Encoder = preprocessing.OneHotEncoder()
            Vectors = numpy.array(Vectors) # 2-D array
            Encoded_Categorical_Features = Encoder.fit_transform(Vectors[:, 0:6+1])
            Encoded_Categorical_Features = Encoded_Categorical_Features.toarray()
            Encoded_Vectors = numpy.hstack((Encoded_Categorical_Features, Vectors[:, 6+1:]))
            Encoded_training_data[vID] = (Encoded_Vectors, Targets)
            Encoders[vID] = Encoder
    return Encoded_training_data, Encoders


def standardize_features(Training_data):
    """Standarize feature vectors for each influx

    Later, a new feature vector X for i-th influx can be normalized as:
    Scalers[i].transform(X)

    """
    Std_training_data, Scalers = {}, {}
    for vID, (Vectors, Labels) in Training_data.iteritems():
        Vectors_scaled = preprocessing.scale(Vectors)
        Std_training_data[vID] = ( Vectors, Labels)

        Scalers[vID] = preprocessing.StandardScaler().fit(Vectors)

    return Std_training_data, Scalers


def train_model(Training_data):
    """Train a regression model for each of the 29 influxes

    Returns
    ================
    Models: dict, keys are influx indexes and values are regression models

    """
    Models = {}
    for i in range(1, 29+1):
        (Vectors, Label) = Training_data[i]
        Model = KNN_MODEL.model
        Model.fit(Vectors, Label) # train the model
        Models[i] = Model
        return Models


def cross_validation_model(training_data, model):
    print("v\tmodel\tscore_accuracy")
    for i in range(1, 29 + 1):
        vectors, label = training_data[i]
        label = numpy.asarray(label)
        scores = cross_validation.cross_val_score(model.model, vectors, label,
                                                  cv=4)
        print("{}\t{}\t{} (+/- {})"
              .format(i, model, scores.mean(), scores.std() * 2))


def grid_search_cv(training_data, model, params):
    print("v\tmodel\tbest_estimator")
    for i in range(1, 29 + 1):
        vectors, label = training_data[i]
        clf = grid_search.GridSearchCV(model.model, params)
        clf.fit(vectors, label)
        print("{}\t{}\t{}".format(i, model, clf.best_estimator_))


if __name__ == "__main__":
    Training_data = read_spreadsheet("1111_forrest.csv")
    Encoded_training_data, Encoders = one_hot_encode_features(Training_data)
    Std_training_data, Scalers = standardize_features(Encoded_training_data)

    cross_validation_model(Std_training_data, KNN_MODEL)
    cross_validation_model(Std_training_data, SVR_MODEL)

    grid_search_cv(Std_training_data, KNN_MODEL, KNN_PARAMS)
    grid_search_cv(Std_training_data, SVR_MODEL, SVR_PARAMS)

    Models = train_model(Std_training_data)
    cPickle.dump(Models, open("models_knn.p", "wb"))
    cPickle.dump(Scalers, open("scalers.p", "wb"))
    cPickle.dump(Encoders, open("encoders.p", "wb"))
    cPickle.dump(Training_data, open("training_data.p", "wb"))
