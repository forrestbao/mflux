# extract the training data from spreadsheet

def read_spreadsheet(File):
    """Turn spreadsheet into matrixes for training

    Returns
    ========

	Training_data: a dict, keys are EMPs (e.g., v1, v2, etc.),
                   values are 2-tuples (Feature, Label), where
                   Feature is a 2-D list, each sublist is 27-D feature vector for one sample
                   and 
                   Label is a 1-D list, labels for all samples.

    Notes
    ============
    EMPs are N.A. for some samples, training features were dropped for them.
    That's why we need one training feature matrix for each EMP.
    
    We have 29 influxes values to predict 

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

def standardize_features(Training_data):
    """Standarize feature vectors for each influx

    Later, a new feature vector X for i-th influx can be normalized as:
    Scalers[i].transform(X)

    """
    from sklearn import preprocessing
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
    from sklearn.neighbors import KNeighborsRegressor
    Models = {}
    for i in range(1, 29+1):
        (Vectors, Label) = Training_data[i]
        Model = KNeighborsRegressor(n_neighbors=4) # initialize the model
        Model.fit(Vectors, Label) # train the model
        Models[i] = Model

    return Models

if __name__ == "__main__":
    Training_data = read_spreadsheet("1111_forrest.csv")
    Std_training_data, Scalers = standardize_features(Training_data)
    Models = train_model(Std_training_data)
    import cPickle
    cPickle.dump(Models, open("models_knn.p", "wb"))
    cPickle.dump(Scalers, open("scalers.p", "wb"))