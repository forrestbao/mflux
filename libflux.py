def test(S):
	print S

def predict(Vector, Substrates):
    """ Predict and adjust all influx values

    Vector: list of floats, the feature vector, including substrate matrix
    Substrates: dict of floats, 1-indexed part of Feature_vector, ratio of substrates
    Models: dict of models, 1-indexed. 
    """
    import cPickle
    import time

#    print Vector
#    print Substrates

    Models = cPickle.load(open("models.p", "r"))
    #  Models: dict, keys are influx indexes and values are regression models

    for i, Model in Models.iteritems():
        Influx_value =  Model.predict(Vector)[0]
        print """\
        v%s = %s, &nbsp; 
        """ % (i, Influx_value)

#  T = time.clock() -T
#    print """</p>\
#    <p>Using a k-NN model where k=5, uniform weights for all neighbors, BallTree of leaf size 30 and Minkowski distance. </p>
#    <p>Regression done in %s seconds.</p>
#    """ % T
    return None 

