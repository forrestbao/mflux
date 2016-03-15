# Copyleft 2016 Forrest Sheng Bao 
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
# This file contains functions for constraint programming in MFlux

def process_species_db(File):
    """Load the species database which is Suppliment Information I

    Format
    ========
    Fields separated by tab

    Species Spcies name Oxygen condition    Substrate uptake rate upper bound (mmol/gDW*mol )   1   2   3   4   5   6   7   8   9   10  11  12  13  14  Growth rate upper bound (h-1)   Reference
    1   Escherichia coli    1,3,2   20  Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   N   1.2 1
    2   Corynebacterium glutamicum  1,3,2   40  Y   Y   Y   Y   Y   Y   Y   Y   N   Y   N   Y   Y   N   1   2

    Returns
    ========
    DB: A list of tuples. Each tuple is (species, Oxygen, rate, Carbon1, Carbon 2, ..., Carbon 14, Growth_rate_upper)
        Oxygen itself is a string, e.g., "1,2,3"

    """
    Carbon_sub = {"Y":True, "N":False}
    DB = []
    with open(File, "r") as F:
        F.readline() # skip the header
        for Line in F:
            Field = Line.split("\t")
#            print Field
            [Species, Substrate_rate] = map(int, [Field[0], Field[3]])
            Oxygen = Field[2] #map(int, Field[2].split(","))
            Carbon_src = [ Carbon_sub.get(x, False) for x in Field[4:4+13+1]  ]
            Growth_rate_upper = Field[4+14] 
            DB.append(tuple([Species, Oxygen, Substrate_rate]+ Carbon_src + [Growth_rate_upper]))

    return DB

def species_db_to_constraints(DB, Debug=False):
    """Turn the species DB into a CSP problem (constraints only, no variable ranges)

    Parameters
    =============
    DB: list of tuples
        Each tuple is (species, Oxygen, rate, Carbon1, Carbon 2, ..., Carbon 14)
        Oxygen itself is a tuple, e.g., (1,3)

    Returns
    =========
    problem: an instance of python-constraint
        containing only constraints but no variable domains

    Notes
    ========
    the problem has a solution if any of the rules set in species database is VIOLATED. 
    In other words, if the problem has solution, then the input does NOT make sense. 

    """
    import constraint # python-constraint module
    problem = constraint.Problem() # initialize the CSP problem

    # create variables
#    problem.addVariable("Species", range(1,41+1))
#    problem.addVariable("Substrate_rate", range(0, 100+1))
#    problem.addVariable("Oxygen", [1,2,3])
#    for i in xrange(1, 14+1):
#        problem.addVariable("Carbon"+str(i), [True, False]) # create one variable for each carbon source
    # This part should be from user input

    # add constraints, where each entry in DB is a constraint. 
    #   create the lambda functions
    All_vars= ["Species", "Substrate_rate", "Oxygen"] + ["Carbon"+str(i) for i in xrange(1, 14+1)] + ["Growth_rate_upper"]
    for Entry in DB:
        Oxygen_values = Entry[1] # as string
        Foo = "lambda "
        Foo += ", ".join(All_vars) # done with listing all variables
        Foo += ": " 
        Logic_exp = ["Substrate_rate<=" + str(Entry[2]), "Species==" + str(Entry[0]), "Growth_rate_upper<=" + str(Entry[4+13])]

        for i in range(3, 3+14): # carbon sources
            if not Entry[i]: # only use false ones to create the constraint
                Logic_exp.append( ( All_vars[i] + "==" + str(Entry[i]) ) )

        Logic_exp.append( ( "Oxygen in [" + Oxygen_values + "]" )  ) 

        Logic_exp = " and ".join(Logic_exp)
        Logic_exp = "not (" + Logic_exp + ")"  # De Morgan's Law
        if Debug: 
            print Logic_exp
        problem.addConstraint(eval(Foo + Logic_exp), tuple(All_vars))

    return problem # just return one solution, if no solution, return is NoneType

def input_ok(problem, Vector):
    """Turn user inputs into domains of variables for the CSP problem and then solve. 

    Parameters
    ============
    problem: a python-constraint instance with constraints built 
    Vector: the feature vector, float numbers, [Species, Reactor, Nutrient, Oxygen, Method, MFA, Energy, Growth_rate, Substrate_uptake_rate] + ratio of 14 carbon sources in the order: "glucose", "fructose", "galactose", "gluconate", "glutamate", "citrate", "xylose", "succinate", "malate", "lactate", "pyruvate", "glycerol", "acetate",  "NaHCO3"

    Notes
    ========
    In current formulation, the problem has a solution if any of the rules set in species database is VIOLATED. 
    In other words, if the problem has solution, then the input does NOT make sense. 

    Example
    =========
    >>> import clp 
    >>> DB = clp.process_species_db("SI_1_species_db.csv")
    >>> P  = clp.species_db_to_constraints(DB)
    >>> Vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72, 10.47, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]  
    >>> print clp.input_ok(P, Vector)
    True
    >>> P.reset() # another test, violating the carbon source it takes
    >>> P  = clp.species_db_to_constraints(DB)
    >>> Vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72, 17, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.0] 
    >>> print clp.input_ok(P, Vector)
    False
    >>> P.reset() # another test, violating growth rate upper boundary
    >>> P  = clp.species_db_to_constraints(DB)
    >>> Vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72, 10.47, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]
    >>> print clp.input_ok(P, Vector)
    False


    """

    problem.addVariable("Species", [Vector[0]])
    problem.addVariable("Substrate_rate", [Vector[8]])
    problem.addVariable("Oxygen", [Vector[3]])
    problem.addVariable("Growth_rate_upper", [Vector[7]])
    for i in xrange(1, 14+1):
        problem.addVariable("Carbon"+str(i), [True if Vector[i+8]>0 else False]) # create one variable for each carbon source

    Solutions = problem.getSolution()
    
    if Solutions == None:# no a single solution, pass test
        return True
    else:
        return False


