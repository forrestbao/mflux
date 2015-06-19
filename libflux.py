import cvxopt
import numpy


Substrate2Index= {
    "glucose": 1,
    "fructose": 2,
    "galactose": 3,
    "gluconate": 4,
    "glutamate": 5,
    "citrate": 6,
    "xylose": 7,
    "succinate": 8,
    "malate": 9,
    "lactate": 10,
    "pyruvate": 11,
    "glycerol": 12,
    "acetate": 13,
}

 # average fluxes from literature suvery
average_fluxes = [
    88.6,
    40.1,
    56.3,
    56.2,
    137.3,
    130.6,
    120.3,
    98.9,
    16.7,
    46.6,
    26.2,
    13.7,
    11.6,
    8.2,
    6.2,
    8.2,
    64.6,
    64.6,
    60.3,
    53.4,
    54.0,
    54.5,
    46.8,
    4.0,
    20.5,
    5.9,
    3.5,
    25.8,
    14.1,
]
average_fluxes = numpy.array(average_fluxes)

ub = [
    100,
    99.5,
    99.3,
    99.3,
    216.6,
    196.2,
    232,
    213.1,
    135,
    151.4,
    113.7,
    94.1,
    41.2,
    47.5,
    71,
    47.5,
    189,
    189,
    189,
    194,
    194,
    194,
    181.5,
    55,
    148,
    193.2,
    151,
    149.8,
    104.2043714,
]

ub = numpy.array(ub)

lb = [
    0,
    -99.9,
    -51.5,
    -51.5,
    -13.5,
    -23.3,
    -36,
    -7.9,
    -144,
    0,
    0,
    -33,
    -94.4,
    -2,
    -6.6,
    -2,
    0,
    -0.1,
    -0.1,
    0,
    -105,
    -106,
    -144.3,
    0,
    0,
    0,
    -100,
    -67.60986805,
    -13.5,
]
lb = numpy.array(lb)


def quadprog(H, f, A, b, Aeq, beq, lb, ub, x0):
    """Simulate matlab quadprog.

    http://www.mathworks.com/help/optim/ug/quadprog.html
    """
    dim = H.shape[1]
    P = H
    q = f
    G = numpy.vstack([A, -numpy.eye(dim), numpy.eye(dim)])
    h = numpy.hstack([b, -lb, ub])
    A = Aeq
    b = beq
    sol = cvxopt.solvers.qp(*map(cvxopt.matrix, [P, q, G, h, A, b]))
    return sol["x"]


# Substrates = {
#     1:1,
#     2:0,
#     3:0,
#     4:0,
#     5:0,
#     6:0,
#     7:0,
#     8:0,
#     9:0,
#     10:0,
#     11:0,
#     12:0,
#     13:0,
#     14:0,
# }
# Fluxes = {
#     1: 100.0,
#     2: -2.7159,
#     3: 15.2254,
#     4: 17.7016,
#     5: 110.9973,
#     6: 91.8578,
#     7: 137.7961,
#     8: 91.1558,
#     9: -0.7373,
#     10: 94.1518,
#     11: 24.1126,
#     12: 21.231,
#     13: 2.8816,
#     14: 11.0324,
#     15: 10.1986,
#     16: 11.0324,
#     17: 79.4203,
#     18: 79.4203,
#     19: 67.9442,
#     20: 67.8806,
#     21: 79.3567,
#     22: 79.3567,
#     23: 64.0876,
#     24: 11.4761,
#     25: 70.0392,
#     26: -1.2424,
#     27: 0.0059,
#     28: 23.2159,
#     29: 26.7451,
# }


def quadprog_adjust(Substrates, Fluxes):
    # specific setting on boundary, based on the user's knowledge on bacteria
    lb[29-1], ub[9-1], ub[24-1], ub[29-1] = 0, 0, 0, 0

    dim = len(average_fluxes)
    H = numpy.eye(dim)
    # TODO: what's 12
    Aineq = numpy.zeros((12 + 1, dim + 1)) # the plus 1 is to tackle MATLAB 1-index
    Aineq[1,1] = 1; Aineq[1,2] = -1; Aineq[1,10] = -1;
    Aineq[2,2] = 1; Aineq[2,3] = -1; Aineq[2,16] = 2;
    Aineq[3,3] = 1; Aineq[3,4] = 1; Aineq[3,5] = -1; Aineq[3,14] = 1; Aineq[3,25] = 1;
    Aineq[4,5] = 1; Aineq[4,6] = -1;
    Aineq[5,6] = 1; Aineq[5,7] = -1; Aineq[5,28] = -1;
    Aineq[6,7] = 1; Aineq[6,8] = -1; Aineq[6,25] = 1; Aineq[6,27] = -1; Aineq[6,29] = 1;
    Aineq[7,8] = 1; Aineq[7,9] = -1; Aineq[7,17] = -1; Aineq[7,24] = -1; Aineq[7,26] = -1;
    Aineq[8,13] = 1; Aineq[8,14] = -1;
    Aineq[9,16] = 1; Aineq[9,15] = -1;
    Aineq[10,19] = 1; Aineq[10,20] = -1;
    Aineq[11,23] = 1; Aineq[11,17] = -1; Aineq[11,28] = 1;
    Aineq[12,21] = -1; Aineq[12,22] = 1;
    Aineq = Aineq[1:, 1:] # convert 1-index to 0-index
    Aineq = -1 * Aineq # because in standarized formulation, it's Ax<=b but in our paper it is Ax>=

    # TODO: what's 12
    bineq = numpy.zeros(12 + 1)
    bineq[2]= 100 * Substrates[Substrate2Index["fructose"]]
    bineq[6]= 100 * Substrates[Substrate2Index["pyruvate"]]
    bineq[10] = 100 * Substrates[Substrate2Index["glutamate"]]
    bineq = bineq[1:] # convert 1-index to 0-index

    # TODO: what's 10
    Aeq = numpy.zeros((10 + 1, dim + 1))
    Aeq[1,1] = 1;
    Aeq[2,3] = 1; Aeq[2,4] = -1;
    Aeq[3,11] = 1; Aeq[3,12] = -1; Aeq[3,13] = -1;
    Aeq[4,14] = 1; Aeq[4,16] = -1;
    Aeq[5,10] = 1; Aeq[5,11] = -1; Aeq[5,25] = -1;
    Aeq[6,18] = 1; Aeq[6,17] = -1;
    Aeq[7,15] = 1; Aeq[7,12] = -1; Aeq[7,14] = 1;
    Aeq[8,24] = 1; Aeq[8,18] = -1; Aeq[8,19] = 1;
    Aeq[9,22] = -1; Aeq[9,23] = 1; Aeq[9,24] = -1; Aeq[9,29] = 1;
    Aeq[10,20] = 1; Aeq[10,24] = 1; Aeq[10,21] = -1;
    Aeq = Aeq[1:, 1:] # convert 1-index to 0-index

    # TODO: what's 10
    beq = numpy.zeros((10+1,1+1))
    beq[1,1] = 100 * (Substrates[Substrate2Index["glucose"]] + Substrates[Substrate2Index["galactose"]])
    beq[2,1] = -100 * Substrates[Substrate2Index["glycerol"]]
    beq[5,1] = -100 * Substrates[Substrate2Index["gluconate"]]
    beq[6,1] = 100 * Substrates[Substrate2Index["citrate"]]
    beq[7,1] = 100 * Substrates[Substrate2Index["xylose"]]
    beq[9,1] = 100 * Substrates[Substrate2Index["malate"]]
    beq[10,1]= -100 * Substrates[Substrate2Index["succinate"]]
    beq = beq[1:, 1:] # convert 1-index to 0-index

    Solution = quadprog(H, -average_fluxes, Aineq, bineq, Aeq, beq, lb, ub, average_fluxes)
    Solution = numpy.array(Solution)[:,0]

    numpy.set_printoptions(precision=5, suppress=True)

    for Idx, Value in enumerate(Solution):
        print Idx+1, Value, Fluxes[Idx+1] # convert from 0-index to 1-index

    return Solution


# def quadprog_adjust(Substrates, Fluxes):
#     """adjust values from ML

#     Substrates: OrderedDict, keys as integers and values as floats, e.g., {1:0.25, 2:0, 3:0.75, ...}
#     Fluxes: Dict, keys as integers and values as floats, e.g., {1:99.5, 2:1.1, ...}

#     Returns
#     =========
#      Solution: an ndarray of 29 floats, adjusted flux values

#     Notes
#     ========
#     In Substrates, the mapping from keys to real chemicals is as follows:
#  	    1. Glucose
#         2. Fructose
#         3. Galactose
#         4. Gluconate
#         5. Glutamate
#         6. Citrate
#         7. Xylose
#         8. Succinate
#         9. Malate
#         10. Lactate
#         11. Pyruvate
#         12. Glycerol
#         13. Acetate
#         14. NaHCO3

#     Formulation of quadratic problems in MATLAB optimization toolbox are different from that in cvxopt.
#     Here is a mapping between variables
#     * H => P (the quadratic terms in objective function)
#     * f => q (the linear terms in objective function)
#     * A and eyes for boundaries => G (coefficients for linear terms in inequality constraints)
# 	* b, -lb, ub => h (coefficients for constant terms in inequality constraints)
#     * Aeq => A
#     * Beq => b

#     Example
#     ============
#     >>> Substrates = {1:1, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
#     >>> Fluxes = {1: 100.0, 2: -2.7159, 3: 15.2254, 4: 17.7016, 5: 110.9973, 6: 91.8578, 7: 137.7961, 8: 91.1558, 9: -0.7373, 10: 94.1518, 11: 24.1126, 12: 21.231, 13: 2.8816, 14: 11.0324, 15: 10.1986, 16: 11.0324, 17: 79.4203, 18: 79.4203, 19: 67.9442, 20: 67.8806, 21: 79.3567, 22: 79.3567, 23: 64.0876, 24: 11.4761, 25: 70.0392, 26: -1.2424, 27: 0.0059, 28: 23.2159, 29: 26.7451}
#     >>> import libflux
#     >>> libflux.quadprog_adjust(Substrates, Fluxes)
#     """

#     import numpy
#     import cvxopt, cvxopt.solvers


#     Substrate2Index= {
#         "glucose": 1,
#         "fructose": 2,
#         "galactose": 3,
#         "gluconate": 4,
#         "glutamate": 5,
#         "citrate": 6,
#         "xylose": 7,
#         "succinate": 8,
#         "malate": 9,
#         "lactate": 10,
#         "pyruvate": 11,
#         "glycerol": 12,
#         "acetate": 13,
#     }

#     Ubs = numpy.array([
#         [
#             100,
#             99.5,
#             99.3,
#             99.3,
#             216.6,
#             196.2,
#             232,
#             213.1,
#             0, # 135,
#             151.4,
#             113.7,
#             94.1,
#             41.2,
#             47.5,
#             71,
#             47.5,
#             189,
#             189,
#             189,
#             194,
#             194,
#             194,
#             181.5,
#             0, #55,
#             148,
#             193.2,
#             151,
#             149.8,
#             0, # 104.2043714,
#         ]
#     ])
#     Ubs = Ubs.transpose() # turn it into column vector, 29x1

#     Lbs = numpy.array([
#         [
#             0,
#             -99.9,
#             -51.5,
#             -51.5,
#             -13.5,
#             -23.3,
#             -36,
#             -7.9,
#             -144,
#             0,
#             0,
#             -33,
#             -94.4,
#             -2,
#             -6.6,
#             -2,
#             0,
#             -0.1,
#             -0.1,
#             0,
#             -105,
#             -106,
#             -144.3,
#             0,
#             0,
#             0,
#             -100,
#             -67.60986805,
#             0, #-13.5,
#         ]
#     ])
#     Lbs = Lbs.transpose() # turn it into column vector, 29x1

#     Aineq = numpy.zeros((12+1, 29+1)) # the plus 1 is to tackle MATLAB 1-index

# #     Aineq[1,1] = 1; Aineq[1,2] = -1; Aineq[1,10] = -1;
# # #    Aineq[2,2] = 1;Aineq[2,3] = -1; Aineq[2,15] = 1; Aineq[2,16] = 1;
# # #    Aineq[3,3] = 1; Aineq[3,4] = 1; Aineq[3,5] = -1;Aineq[3,14] = 1; Aineq[3,15] = 1; Aineq[3,16] = 1; Aineq[3,25] = 1;
# #     Aineq[2,2] = 1;Aineq[2,3] = -1; Aineq[2,16] = 1;
# #     Aineq[3,3] = 1; Aineq[3,4] = 1; Aineq[3,5] = -1;Aineq[3,14] = 1; Aineq[3,25] = 1;
# #     Aineq[4,5] = 1; Aineq[4,6] = -1;
# #     Aineq[5,6] = 1; Aineq[5,7] = -1; Aineq[5,28] = -1;
# #     Aineq[6,7] = 1; Aineq[6,8] = -1; Aineq[6,25] = 1;Aineq[6,27] = -1; Aineq[6,29] = 1;
# #     Aineq[7,8] = 1; Aineq[7,9] = -1; Aineq[7,17] = -1;Aineq[7,24] = -1; Aineq[7,26] = -1;
# #     Aineq[8,13] = 1; Aineq[8,14] = -1;
# #     Aineq[9,16] = 1; Aineq[9,15] = -1;
# #     Aineq[10,19] = 1; Aineq[10,20] = -1;
# #     Aineq[11,23] = 1; Aineq[11,17] = -1;Aineq[11,28] = 1;
# #     Aineq[12,21] = -1; Aineq[12,22] = 1;

#     Aineq[1,1] = 1; Aineq[1,2] = -1; Aineq[1,10] = -1;
#     Aineq[2,2] = 1;Aineq[2,3] = -1;  Aineq[2,16] = 2;
#     Aineq[3,3] = 1; Aineq[3,4] = 1; Aineq[3,5] = -1;Aineq[3,14] = 1; Aineq[3,25] = 1;
#     Aineq[4,5] = 1; Aineq[4,6] = -1;
#     Aineq[5,6] = 1; Aineq[5,7] = -1; Aineq[5,28] = -1;
#     Aineq[6,7] = 1; Aineq[6,8] = -1; Aineq[6,25] = 1;Aineq[6,27] = -1; Aineq[6,29] = 1;
#     Aineq[7,8] = 1; Aineq[7,9] = -1; Aineq[7,17] = -1;Aineq[7,24] = -1; Aineq[7,26] = -1;
#     Aineq[8,13] = 1; Aineq[8,14] = -1;
#     Aineq[9,16] = 1; Aineq[9,15] = -1;
#     Aineq[10,19] = 1; Aineq[10,20] = -1;
#     Aineq[11,23] = 1; Aineq[11,17] = -1;Aineq[11,28] = 1;
#     Aineq[12,21] = -1; Aineq[12,22] = 1;



#     Aineq = Aineq[1:, 1:] # convert 1-index to 0-index
#     Aineq = -1 * Aineq # because in standarized formulation, it's Ax<=b but in our paper it is Ax>=b

#     Aineq = numpy.vstack([Aineq, -numpy.eye(29), numpy.eye(29)]) # add eye matrixes for Lbs and Ubs

#     bineq = numpy.zeros((12+1, 1+1))
#     bineq[2,1]= 100 * Substrates[Substrate2Index["fructose"]]
#     bineq[6,1]= 100 * Substrates[Substrate2Index["pyruvate"]]
#     bineq[10,1] = 100 * Substrates[Substrate2Index["glutamate"]]
#     bineq = bineq[1:, 1:] # convert 1-index to 0-index
#     bineq = numpy.vstack([bineq, -Lbs, Ubs])

#     Aeq = numpy.zeros((10+1, 29+1))
#     Aeq[1,1] = 1;
#     Aeq[2,3] = 1; Aeq[2,4] = -1;
#     Aeq[3,11] = 1; Aeq[3,12] = -1; Aeq[3,13] = -1;
#     Aeq[4,14] = 1; Aeq[4,16] = -1;
#     Aeq[5,10] = 1; Aeq[5,11] = -1; Aeq[5,25] = -1;
#     Aeq[6,18] = 1; Aeq[6,17] = -1;
#     Aeq[7,15] = 1; Aeq[7,12] = -1; Aeq[7,14] = 1;
#     Aeq[8,24] = 1; Aeq[8,18] = -1; Aeq[8,19] = 1;
#     Aeq[9,22] = -1; Aeq[9,23] = 1; Aeq[9,24] = -1;Aeq[9,29] = 1;
#     Aeq[10,20] = 1; Aeq[10,24] = 1;Aeq[10,21] = -1;
#     Aeq = Aeq[1:, 1:] # convert 1-index to 0-index
#     Aeq = numpy.matrix(Aeq)

# #    Aeq = Aeq.transpose().tolist()

#     beq = numpy.zeros((10+1,1+1))
#     beq[1,1] = 100 * (Substrates[Substrate2Index["glucose"]] + Substrates[Substrate2Index["galactose"]])
#     beq[2,1] = -100 * Substrates[Substrate2Index["glycerol"]]
#     beq[5,1] = -100 * Substrates[Substrate2Index["gluconate"]]
#     beq[6,1] = 100 * Substrates[Substrate2Index["citrate"]]
#     beq[7,1] = 100 * Substrates[Substrate2Index["xylose"]]
#     beq[9,1] = 100 * Substrates[Substrate2Index["malate"]]
#     beq[10,1]= -100 * Substrates[Substrate2Index["succinate"]]
#     beq = beq[1:, 1:] # convert 1-index to 0-index
#     beq = numpy.matrix(beq)

#     P = numpy.eye((29))
#     q = [[Fluxes[i] for i in range(1, 29+1)]]
#     q = numpy.array((q)).transpose()

# #    print map(numpy.shape, [Aineq, bineq, Aeq, beq, P, q])
# #    print map(type, [Aineq, bineq, Aeq, beq, P, q])

# #    [bineq] = map(cvxopt.matrix, [bineq])

# #    [beq] = map(cvxopt.matrix, [beq])

# #    [Aineq, bineq, Aeq, beq] = map(cvxopt.matrix, [Aineq, bineq, Aeq, beq])

#     [Aineq, bineq, Aeq, beq, P, q] = map(cvxopt.matrix, [Aineq, bineq, Aeq, beq, P, q])

#     Solv = cvxopt.solvers.qp(P, q, G=Aineq, h=bineq, A=Aeq, b=beq)

#     Solution = Solv['x']

#     Solution = numpy.array(Solution)[:,0]

#     numpy.set_printoptions(precision=5, suppress=True)

#     for Idx, Value in enumerate(Solution):
#         print Idx+1, Value, Fluxes[Idx+1] # convert from 0-index to 1-index

#     return Solution

def test(S):
    print S

def print_influxes(Influxes):
    """Print influxes

    Influxes: dict, keys are influx id, values are floats
    """

    import sys
    sys.stderr = sys.stdout

#    print Influxes
    print """\
    <h2>Influx values based on given parameters:</h2>
    """# % len(Vector)  #"\t".join(map(str, Vector))

    print """\
    <table border=0 border-spacing=5px>
      <tr>
       <td>

    """

#    for x in range(5):
#        print x

    for ID, Value in Influxes.iteritems():
        print """\
        v%s = %.4f, <br>
        """ % (ID, Value)

    print """\
      </td>
       <td>
          <img src=\"centralflux.png\">
       </td>
      </tr>
    </table>
    """

#    for ID, Value in Influxes.iteritems():
#        print """\
#        v%s = %.4f, <br>
#        """ % (ID, Value)


def process_input(Features):
    """Process the result from CGI parsing to form feature vector including substrate matrixi

    Substrates: OrderedDict, keys as integers and values as floats
        1. Glucose
        2. Fructose
        3. Galactose
        4. Gluconate
        5. Glutamate
        6. Citrate
        7. Xylose
        8. Succinate
        9. Malate
        10. Lactate
        11. Pyruvate
        12. Glycerol
        13. Acetate
        14. NaHCO3
    """

    Num_substrates = 14 # excluding other carbon
    # Generate substrate matrix
    import collections
    Substrates = collections.OrderedDict([(i,0) for i in range(1, Num_substrates+1)]) # substrate values, initialization
    Substrates[int(Features["Substrate_first"])] += Features["Ratio_first"]
    Substrates[int(Features["Substrate_sec"])] += Features["Ratio_sec"]

    # Form the feature vector
    Vector = [Features[Feature_name] for Feature_name in ["Species", "Reactor", "Nutrient", "Oxygen", "Method", "MFA", "Energy", "Growth_rate", "Substrate_uptake_rate"]]
    Vector += [Substrates[i] for i in range(1, Num_substrates+1)]
    Vector.append(Features["Substrate_other"]) # Other carbon source

    # Print debug info

    Substrate_names = ["glucose", "fructose", "galactose", "gluconate", "glutamate", "citrate", "xylose", "succinate", "malate", "lactate", "pyruvate", "glycerol", "acetate",  "NaHCO3"]
    Substrate_dict = collections.OrderedDict([(i+1,Name) for i, Name in enumerate(Substrate_names)])
    print "<p>Feature Vector (pre-one-hot-encoding and prescaled):", Vector, "</p>"
    print "<p>in which the substrates ratios are:", [(Substrate_dict[Index],Ratio) for Index, Ratio in Substrates.iteritems()], "</p>"
    print "<p>Feature vector size is ", len(Vector), "</p>"

    return Vector, Substrates

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
    Influxes[15] = Influxes[12] - Influxes[14] + 100 * Substrates[Substrate2Index["xylose"]]
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

    Vector: 1-D list of floats, the feature vector, including substrate matrix, size = 24
    Substrates: dict of floats, 1-indexed part of Feature_vector, ratio of substrates
    Models: dict of models, 1-indexed, 29 moddels for 29 influxes.

    Calls adjust_influxes() to compute dependent influxes.
    """
    import cPickle
    import time
    import collections
    import sys

    Models = cPickle.load(open("models_svm.p", "r"))
    Feature_scalers = cPickle.load(open("feature_scalers.p", "r"))
    Encoders = cPickle.load(open("encoders.p", "r"))
    Label_scalers = cPickle.load(open("label_scalers.p", "r"))

    print "Models, feature and label Scalers and one-hot Encoder loaded.."
    #  Models: dict, keys are influx indexes and values are regression models

    T = time.clock()
    Influxes = {}
#    Influxes = {Iundex:Model.predict(Scalers[Index].transform(Vector))[0] for Index, Model in Models.iteritems()}# use dictionary because influx IDs are not consecutive

    for vID, Model in Models.iteritems():
        Vector_local = list(Vector) # make a copy; o/w Vector will be changed in one-hot encoding and standarization for different models
        One_hot_encoding_of_categorical_features =  Encoders[vID].transform([Vector[:6+1]]).toarray().tolist()[0]  # one-hot encoding for categorical features
#        print len(One_hot_encoding_of_categorical_features), "\n"
        Vector_local =  One_hot_encoding_of_categorical_features + Vector_local[6+1:] # combine one-hot-encoded categorical features with continuous features (including substrate matrix)
#        print Vector_local, len(Vector_local)
        Vector_local = Feature_scalers[vID].transform(Vector_local) # standarization of features
#        print Vector_local
        Influx_local = Model.predict(Vector_local)[0] # prediction
        print "v{0:d}={1:.5f}, ".format(vID, Influx_local)
        Influx_local = Label_scalers[vID].inverse_transform([Influx_local])[0]
        Influxes[vID] = Influx_local

#    Influxes = adjust_influxes(Influxes, Substrates) # do not adjust as of 2015-05-10
    T = time.clock() -T

    print_influxes(Influxes)

    print """</p>\
    <p>Using RBF kernel SVM as regressor. Parameters vary for different fluxes. For details, refer to <a href="svr_both_rbf_shuffle.log">this document generated by grid search on SVM parameters</a>. </p>
    <p>Standarization and Regression done in %s seconds.</p>
    """ % T
    return Influxes
