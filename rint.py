#!/usr/bin/python3
##
## rint.py - Integrates gnuplot "table" output with cumtrapz.
##
## Copyright(c) 2017 Kevin Croker
## GPL v3
##
## Integrates all data, or data beyond a lower bound, if specified.
## Outputs to stdout.
##

# (Careful with negative valued functions)
from scipy import integrate
import sys
import math

f=open(sys.argv[1], 'r')
dats = []
for line in f:
    if line[0] == "#" or line[0] == "\n":
        continue

    # Otherwise, add it
    dats.append((line.strip()).split())

# Get the lower bound
lbound = float(sys.argv[2])

# Organize it (I'm sure theres a pythonic way to do this)
x = []
y = []
for dat in dats:
    xm = float(dat[0])
    if xm < lbound:
        continue
    ym = float(dat[1])
    if not math.isnan(xm) and not math.isnan(ym):
        x.append(xm)
        y.append(ym)

# Usually produces cumtrapz from x[0] -> x[len(x) - 1]
# 
# So each line is: F(a) = \int_x[0] ^ x (input data) dx
#
# If we want: F(a) = \int_x^x[l] (input data) dx
if len(sys.argv) > 3:
    x = x[::-1]
    y = [-1*z for z in y[::-1]]

# Now perform the running integration
cum = integrate.cumtrapz(y, x, initial=0)
l = len(x)
for i in range(0, l):
    print(x[i], cum[i])
