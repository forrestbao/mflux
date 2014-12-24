#!/usr/bin/env python

import cgi
form  = cgi.FieldStorage() # instantiate only once!

Feature_names =  ["Purpose", "Reactor", "Species", "Nutrient", "Method", "Energy", "MFA", "Growth_rate", "Substrate_uptake_rate", "Substrate_first", "Ratio_first", "Substrate_sec", "Ratio_sec", "Substrate_other"]

Features = {}

# Avoid script injection escaping the user input
#Purpose ="sdfsd"

print "Content-Type: text/html"
print
print """\
<html>
<head><title>Result of Influx analysis </title></head>

<body>
<h2>Parameters entered:</h2>
"""

# Process the form values
for Feature_name in Feature_names:
    Feature_value = form.getfirst(Feature_name)
    Feature_value = cgi.escape(Feature_value) 
    Features[Feature_name] = float(Feature_value) # convert all string to numbers

    print """\
    %s is %s, 
     """ % (Feature_name, Feature_value)

# Generate substrate matrix
Substrates = {i:0 for i in range(1, 27+1)} # substrate values, initialization
Substrates[int(Features["Substrate_first"])]= Features["Ratio_first"]
Substrates[int(Features["Substrate_sec"])]= Features["Ratio_sec"]

# Form the feature vector
Vector = [Features[Feature_name] for Feature_name in ["Purpose", "Reactor", "Species", "Nutrient", "Method", "Energy", "MFA", "Growth_rate", "Substrate_uptake_rate"]]
Vector += [Substrates[i] for i in range(1, 16+1)]
Vector.append(Features["Substrate_other"])

print """\
<h2>Influx values based on given parameters:</h2>
"""# % len(Vector)  #"\t".join(map(str, Vector))

import libflux 
#libflux.test("hello, world")
libflux.predict(Vector, Substrates) # use the feature vector to predict influx values 

print """\
<p><a href="index.html">Go back to submission page</a></p>

<hr>
<p>
This project is supported by National Science Foundation. <a href="http://www.nsf.gov/awardsearch/showAward?AWD_ID=1356669">More info</a> <br>
Information on this website only relects the perspectives of the individuals.<br>
Built by Forrest Sheng Bao http://fsbao.net 
</p>
</body>
</html>
"""

