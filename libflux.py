def test(S):
	print S

def print_influxes(Influxes):
    """Print influxes
    
    Influxes: dict, keys are influx id, values are floats
    """
#    print Influxes
    for ID, Value in Influxes.iteritems():
        print """\
        v%s = %s, &nbsp; 
        """ % (ID, Value)

def adjust_influxes(Influxes, Substrates):
    """Adjust influxes values
    """
    Substrate2Index= {"glucose":1, "galactose":3, "gluconate":4, "citrate":6, "xylose":7, "succinate":8, "malate":9, "lactate":10, "acetate":13}
 
    #Step 1: Compute dependent influxes 
    Influxes[1] = 100 * Substrates[Substrate2Index["glucose"]]
    Influxes[13] = Influxes[11] - Influxes[12]
    Influxes[16] = Influxes[14]
    Influxes[25] = Influxes[10] - Influxes[11] + 100 * Substrates[Substrate2Index["gluconate"]]
    Influxes[18] = Influxes[17] + 100 * Substrates[Substrate2Index["citrate"]]
    Influxes[15] = Influxes[12] + 100 * Substrates[Substrate2Index["xylose"]]
    Influxes[24] = Influxes[18] - Influxes[19]
    Influxes[21] = Influxes[20] + Influxes[24] + 100 * Substrates[Substrate2Index["succinate"]]
    Influxes[22] = Influxes[21]
    Influxes[29] = Influxes[22] + Influxes[24] - Influxes[23] + 100 * Substrates[Substrate2Index["malate"]]

    # Step 2: Correct flux values
    if Substrates[Substrate2Index["acetate"]] != 0:
        Influxes[9] = -100 * Substrates[Substrate2Index["acetate"]]
    if  Substrates[Substrate2Index["lactate"]] != 0:
        Influxes[27] = -100 * Substrates[Substrate2Index["lactate"]]

    return Influxes

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


    T = time.clock()
    Influxes = {Index:Model.predict(Vector)[0] for Index,Model in Models.iteritems()}# use dictionary because influx IDs are not consecutive
    Influxes = adjust_influxes(Influxes, Substrates)
    print_influxes(Influxes)

    T = time.clock() -T
    print """</p>\
    <p>Using a k-NN model where k=5, uniform weights for all neighbors, BallTree of leaf size 30 and Minkowski distance. </p>
    <p>Regression done in %s seconds.</p>
    """ % T
    return Influxes 

