#!/usr/bin/env python

import cgi
form = cgi.FieldStorage() # instantiate only once!
#name = form.getfirst('name', 'empty')

#name = cgi.escape(name)

name = "Bao"

import constraint
problem = constraint.Problem()
problem.addVariable("a", range(6))
problem.addVariable("b", range(6)) 

problem.addConstraint(lambda a: a<4, ("a"))

Solutions = problem.getSolutions() # A dictionary 



print "Content-Type: text/html"
print
print """\
<html>
<head><title>A Python CLP test script </title></head>

<body>
<h2>Hello World, %s!</h2>
""" % name

#print """\
#<p>The solution is </p>
#""" % 

#print Solutions

for Solution in Solutions:
	for Key, Value in Solution.iteritems(): 
		print """%s=%s""" % (str(Key), str(Value))
	print """<br>"""


print """\
</body>
</html>
""" % name
