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

    We have 27 influx values, ending at LAC

    16 values in feature vector are substrate density

    Now we do not use Oxygen level as a feature. So feature vector length is 26 instead of 27.

    """
    Training_data = {}
    for i in range(1, 27+1):# prepare the data structure
        Training_data[i] = ([],[])

    with open(File, 'r') as F:
        for Line in F.readlines():
            Line = Line.strip()
            Line = Line.split(",")
            Vector = Line[1:29] # training vector, from Purpose (B) to Other carbon (AC).
                              # one empty column
            if Vector[27] == "":
                Vector[27] = "0" # no other carbon source
            Vector.remove("") # delete the element from empty column
            Labels = Line[26+5: 26+3+16+13] # AF to BF, last one is LAC
            if not Vector :
                print Line
                exit()

            Vector = Vector[:3+1] + Vector[5:] # drop the 4th feature value which is Oxygen

            Vector = map(float, Vector)
            # Now create the dictionaries we need
            for i in range(1, 27+1):
                Label = Labels[i-1]
                try:
                    Label = float(Label)
                except ValueError:
                    print Label, "=>"
                    print Line 
                    continue # this label for this influx is not numeric

                Training_data[i][0].append(Vector) # add a row to feature vectors
                Training_data[i][1].append(float(Label)) # add one label

    return Training_data

def train_model(Training_data):
    """Train a regression model for each of the 27 influxes

    Returns 
    ================
    Models: dict, keys are influx indexes and values are regression models

    """
    from sklearn.neighbors import KNeighborsRegressor
    Models = {}
    for i in range(1, 27+1):
        (Vectors, Label) = Training_data[i]
        Model = KNeighborsRegressor(n_neighbors=4) # initialize the model
        Model.fit(Vectors, Label) # train the model
        Models[i] = Model

    return Models

if __name__ == "__main__":
    Training_data = read_spreadsheet("1111_forrest.csv")
    Models = train_model(Training_data)
    import cPickle
    cPickle.dump(Models, open("models.p", "wb"))
