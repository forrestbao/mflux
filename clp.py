def process_species_db(File):
    """Load the species database which is Suppliment Information I

    Format
    ========
    Fields separated by tab

    Species Spcies name Oxygen condition    Substrate uptake rate upper bound (mmol/gDW*mol )   1   2   3   4   5   6   7   8   9   10  11  12  13  14      Reference
    1   Escherichia coli    1,3,2   20  Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   N       1
    2   Corynebacterium glutamicum  1,3,2   40  Y   Y   Y   Y   Y   Y   Y   Y   N   Y   N   Y   Y   N       2
    3   Bacillus subtilis   1,3,2   30  Y   Y   Y   Y   Y   N   Y   Y   Y   Y   Y   Y   Y   N       3
    4   Pseudomonas putida  1   20  Y   Y   N   Y   Y   Y   N   Y   N   Y   Y   Y   Y   N       4
    5   Synechocystis 6803  1,3,2   5   Y   Y   Y   ND  Y   Y   N   Y   Y   Y   Y   Y   Y   Y       5, 6
    6   Shewanella oneidensis   1,3,2   20  N   N   N   N   Y   N   N   Y   Y   Y   Y   N   Y   N       7
    7   Rhodobacter sphaeroides     1,3,2   20  Y   Y   -   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   N       8
    8   Actinobacillus succinogenes     1,3,2   20  Y   Y   Y   Y   Y   N   Y   ND  Y   ND  ND  Y   ND  N       9
    9   Rhodopseudomonas palustris  1,3,2   10  N   N   N   ND  N   N   N   Y   Y   Y   Y   Y   Y   Y       10 - 14

    Returns
    ========
    DB: A list of tuples. Each tuple is (species, Oxygen, rate, Carbon1, Carbon 2, ..., Carbon 14)
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
            Carbon_src = [ Carbon_sub.get(x, False) for x in Field[4:4+14+1]  ]
            DB.append(tuple([Species, Oxygen, Substrate_rate]+ Carbon_src))

    return DB

def species_db_to_constraints(DB):
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

    """
    import constraint # python-constraint module
    problem = constraint.Problem() # initialize the CSP problem

    # create variables
    problem.addVariable("Species", range(1,41+1))
    problem.addVariable("Substrate_rate", range(0, 100+1))
    problem.addVariable("Oxygen", [1,2,3])
    for i in xrange(1, 14+1):
        problem.addVariable("Carbon"+str(i), [True, False]) # create one variable for each carbon source
    # This part should from user input


    # add constraints, where each entry in DB is a constraint. 
    #   create the lambda functions

    All_vars= ["Species", "Substrate_rate", "Oxygen"] + ["Carbon"+str(i) for i in xrange(1, 14+1)]
    for Entry in DB:
        Oxygen_values = Entry[1] # as string
        Foo = "lambda "
        Foo += ", ".join(All_vars) # done with listing all variables
        Foo += ": " 
        Logic_exp = ["Substrate_rate<=" + str(Entry[2])]
        for i in [0,2] + range(3, 3+14): # skip Oxygen
            Logic_exp.append( ( All_vars[i] + "==" + str(Entry[i]) ) )
        Logic_exp.append( ( "Oxygen in [" + Oxygen_values + "]" )  ) 
        Logic_exp = " and ".join(Logic_exp)
        Logic_exp = "not (" + Logic_exp + ")"
        print Logic_exp
        problem.addConstraint(eval(Foo + Logic_exp), tuple(All_vars))

    return problem # just return one solution, if no solution, return is NoneType
