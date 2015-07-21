#!/usr/bin/env python

import cgi
form  = cgi.FieldStorage() # instantiate only once!

import cgitb; cgitb.enable()

#Feature_names =  ["Purpose", "Reactor", "Species", "Nutrient", "Method", "Energy", "MFA", "Growth_rate", "Substrate_uptake_rate", "Substrate_first", "Ratio_first", "Substrate_sec", "Ratio_sec", "Substrate_other"]
Feature_names =  ["Species", "Reactor", "Nutrient", "Oxygen", "Method", "MFA", "Energy", "Growth_rate", "Substrate_uptake_rate", "Substrate_first", "Ratio_first", "Substrate_sec", "Ratio_sec", "Substrate_other"]


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

import libflux 
Vector, Substrates = libflux.process_input(Features)
Boundary_dict = libflux.process_boundaries(form)

#libflux.test("hello, world")
Influxes = libflux.predict(Vector, Substrates, Boundary_dict) # use the feature vector to predict influx values 

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

