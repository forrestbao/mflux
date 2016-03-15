#!/usr/bin/env python

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
# This file processes user input submitted from webform in MFlux

import cgi
form  = cgi.FieldStorage() # instantiate only once!

import cgitb; cgitb.enable()

# Feature_names =  ["Species", "Reactor", "Nutrient", "Oxygen", "Method", "MFA", "Energy", "Growth_rate", "Substrate_uptake_rate", "Substrate_first", "Ratio_first", "Substrate_sec", "Ratio_sec", "Substrate_other"]
Feature_names =  ["Species", "Reactor", "Nutrient", "Oxygen", "Method", "Growth_rate", "Substrate_uptake_rate", "Substrate_first", "Ratio_first", "Substrate_sec"]

Features = {"Energy":1.0, "MFA":1.0, "Substrate_other":0.0, "Ratio_sec":0.0}

# Avoid script injection escaping the user input
#Purpose ="sdfsd"

print "Content-Type: text/html"
print
print """\
<html>
<head><title>Result of Influx analysis </title></head>

<body>
"""

# Process the form values
for Feature_name in Feature_names:
    Feature_value = form.getfirst(Feature_name)
    Feature_value = cgi.escape(Feature_value) 
    Features[Feature_name] = float(Feature_value) # convert all string to numbers

    print """\
    %s is %s, 
     """ % (Feature_name, Feature_value)

Features["Ratio_sec"] = 1- Features["Ratio_first"]

import libflux 
Vector, Substrates = libflux.process_input(Features)
Boundary_dict = libflux.process_boundaries(form, Substrates)

#libflux.test("hello, world")
Influxes = libflux.predict(Vector, Substrates, Boundary_dict) # use the feature vector to predict influx values 

print """\
<p><a href="index.html">Go back to submission page</a></p>

<hr>
<p>
This project is supported by National Science Foundation. <a href="http://www.nsf.gov/awardsearch/showAward?AWD_ID=1356669">More info</a> <br>
Information on this website only relects the perspectives of the individuals.<br>
Built by Forrest Sheng Bao <a href="http://fsbao.net">http://fsbao.net </a>
</p>
</body>
</html>
"""

