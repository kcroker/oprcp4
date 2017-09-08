#!/usr/bin/python3
##
## orpcp4.py - v.1 Integrate the Friedmann equations with a point-like diluting DE sourced by stellar collapse
##             v.2 and subequent accretion
##             v.3 uses only a single diffeq, \LambdaCDM only (no variable DE EOS, no splining)
##             v.4 uses the Madau-Dickinson 2014 comoving stellar density model, extrapolated back
##
## DISCLAIMER: I did not use the accretion code for the publication after making adjustments for v4, 
##             I just set accretion rate to 0.  
##             It may be broken. 
##
## Copyright(c) 2017 Kevin Croker
## GPL v3
##

import numpy as np
import sys
from scipy import integrate

## Read in the parameters for the run
if len(sys.argv) < 9:
    print("oprcp4.py - Integrate the Friedmann equations with a point-like diluting")
    print("            perfect fluid with fixed equation of state, ")
    print("            sourced by stellar collapse and possibly subsequent accretion.")
    print("            Optionally include cosmological constant.")
    print("Usage: %s <a0> <Omega_m> <w_0> <Omega_Lambda> <SFD slope parameter> <accretion rate> <final fraction consumed> <accretion onset z>" % sys.argv[0])
    exit(1)

## Don't try to read the future
amax = 1

a0 = float(eval(sys.argv[1]))
z0 = 1/a0 - 1

Omega_m = float(eval(sys.argv[2]))
w0 = float(eval(sys.argv[3]))
Omega_Lambda = float(eval(sys.argv[4]))
k = float(eval(sys.argv[5]))
kappa = float(eval(sys.argv[6]))
Xi = float(eval(sys.argv[7]))
zcrit = float(eval(sys.argv[8]))

## Definitions of the SFD as a function of scale, 
## approximated from A&A 554, A70 (2013)
##
## This is used in the accretion model, but not for the stellar model, which uses
## Madau and Dickinson in v.4
##
## This ("A") is the ratio of the stellar fraction to the present day 
## matter density.
##
## This is the computed value from: the Planck 2015 \Omega_m, the 2011 PDG value
## for the critical density, and the y-intercept of the Herschel data 10^9 M_\sol/Mpc^3 
A = 2.32e-2

## Note that we start from zero correctly now because we are integrating
##
## We want to use the Madau-Dickinson density
## but we need the integrated density, which depends on
## the Hubble factor up until a.  So we need to stash these values as we are called
## and then cumtrapz.
##

# We need to perform running integrations
a_stack = []
dbhda_stack = []

def bh(a, v):
    # Push another dbhda(a,v)
    a_stack.append(a)
    dbhda_stack.append(dbhda(a,v))

    # Now cumtrapz
    cum = integrate.cumtrapz(dbhda_stack, a_stack, initial=0)
    
    # Return the final value
    return cum[-1]
    
def madau_psi(z):
    return 0.015*(1+z)**2.7/(1 + ( (1+z)/2.9 )**5.6)

# (Note the 0.277 correction to go from Astronomer units to cosmological units)
# R is Madau's return fraction upon stellar death
R = 0.27
def dbhda(a,v):
    return 0.277*(1-R)*madau_psi(1/a - 1)/v

## Definitions of the individual BH accretion rate 
## approximated from ApJ (2007) Li, Hernquist, etc.
##
## Assuming a Mr of 1 is the same as saying that all of the
## characteristic star mass becomes BH seed.  Seed values are between
## 100Msol and 1000Msol.  Setting this to one keeps the model 
## a single parameter fit, which makes Enrico Fermi the hap.
Mr = 1

##
## This is reconstructed from Fig. 10, bottom (3rd) panel.  There
## is some flexibility in this value, and its actually superexponential,
## but we don't need specifics here, just a range of viabilities and 
## insensitivity of consumed fraction to the specifics.

## It is likely that accretion dominates the buildup process.  In
## other words, that the continued formation of BHs at late z will be
## inconsequential.
zcut = 6

if z0 <= zcrit:
    zcrit = z0

def accrete(a):
    z = 1/a - 1
    
    # Turn off accretion at quasar visibility time
    # (the purge, where feedback generates enough whind to blow
    #  of the accrete)
    if z < zcut:
        z = zcut

    # Turn on accretion, as a check on the systematic dependence 
    # This should be done because you need galaxies before you
    # efficiently accrete, so zcrit can be taken as the time
    # when galaxy mergers begin to happen
    if z > zcrit:
        z = zcrit

    # 
    # The integrated accretion function as seen on TV
    kd = kappa - k
    return A*Mr*Xi/kd*(k*np.exp(zcrit*(kappa - k) - kappa*z) + kd*np.exp(-k*zcrit) - kappa*np.exp(-k*z))

def daccreteda(a):
    z = 1/a - 1

    # Do the cut
    if z <= zcut or z > zcrit:
        return 0
    else:
        # Notice the -1/a^2 factor because we took the z derivative...
        kd = kappa - k
        return A*Mr*k*kappa*Xi/kd*(np.exp(-k*z) - np.exp(zcrit*(kappa - k) - kappa*z))*(-1/a**2)

## So we can turn off BH formation
## (e.g. making reference cosmological histories)
if k > 0:
    bhfxn = bh
    dbhfxn = dbhda
else:
    bhfxn = lambda a,v : 0
    dbhfxn = lambda a,v : 0

## And we can turn off accretion
## (e.g. studying how dominant the effect is)
if kappa > 0:
    accretefxn = accrete
    daccretefxn = daccreteda
else:
    accretefxn = lambda a : 0
    daccretefxn = lambda a : 0

## Scale-dependent comoving density
##
## In order for the initial condition for q0 to be correct, we must
## start from perfect matter domination.
##
def rho(a, v, Omega_m):
    return Omega_m*(1 - accretefxn(a) - bhfxn(a,v))

## 
## Here is the simplified eqution of state w(a)
##
def w(a):
    return w0

##
## Definitions: q = f + rho + f_Lambda (which is set to Omega_Lambda*a^3)
##

## These dynamical equations are completely de-dimensionalized.
## Taking the time unit to be: 1/H_0
## Taking density unit to be: \rho_cr(a=1)
##  
## Since we are spatially flat, this means that \dot{a}(a = 1) = 1
## when you're doing it right.
##
## Note that we explicitly burn in Omega_Lambda as equation of state -1
##
## Note that v is usually a vector of the linear system being solved
## so here we just want to send the value to rho()
##
def friedmann(v, a, Omega_m):
    return 3.0*w(a)/2*(rho(a, v[0], Omega_m)/(v*a**2) - v/a) - v/(2*a) + 3*Omega_Lambda*a*(w(a) + 1)/(2*v)

## Initial summed energy density.
##
## NOTE: We now must start from when formation onset happens.
##       This is because \rho now depends on v via the Hubble rate in
##       Madau and Dickinson.   
q0 = Omega_m + Omega_Lambda*a0**3

## v0 \equiv sqrt(q0/a0)  [from algebraic constraint]
v0 = np.sqrt(q0/a0)
y0 = [v0]

## Generate the lattice for scale factors
N = 1000
a = np.linspace(a0, amax, N)

## Integrate and output
from scipy.integrate import odeint
sol = odeint(friedmann, y0, a, args=tuple([Omega_m]))

## Give some information for orientation
print("## a0 = %g\n## w0 = %g\n## Omega_Lambda = %g\n## Omega_m = %g" % (a0, w0, Omega_Lambda, Omega_m))
print("## a | \dot{a} | processed fraction of total matter density")
for n,vec in enumerate(sol):
    print(a[n], vec[0], accretefxn(a[n]) + bhfxn(a[n], vec[0]))
