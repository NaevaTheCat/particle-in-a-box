#!/usr/bin/env python
"""
Using the Hartree approximation to solve the problem
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=3)
import scipy as sci
import scipy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.integrate
import HFroutines as HF
"""
Starting parameters
"""
# Box length
L = 5
# Number of basis functions
n_basis= 10
# Number of spatial orbitals
n_orbitals = 2
# Divisions of x in L
xsteps = 50*n_basis
# Potential inside the box
V_0 = np.zeros((xsteps))
# Array of x values
x = np.linspace(0,L,xsteps)

F = np.zeros(n_basis*n_basis).reshape(n_basis,n_basis)
C = np.ones(n_basis*n_basis).reshape(n_basis,n_basis)
C = HF.normCol(C,n_basis)
"""
Next stage involves composition of basis functions
Will use the solutions to the 1D infinite well which
Take the form Chi_n(x)=sin(n*pi*x/L)
"""
# The ith column of chi will be the ith basis function
Chi = np.zeros((xsteps*n_basis)).reshape(xsteps,n_basis)
for i in xrange(0,n_basis):
        Chi[:,i] = np.sqrt(2./L)*np.sin((i+1)*np.pi*x/L)
        # Correct boundary condition for finite resolution
        Chi[-1,:] = 0.

"""
Outcome of the Kinetic energy operator T is attained analytically
T_ij = <Chi_i|-1/2*d^2/dx^2|Chi_j> = E_n(Chi) for i=j=n or 0 for i!=j
E_n = n^2*pi^2/(2*L^2)
"""
T = np.zeros(n_basis*n_basis).reshape(n_basis,n_basis)
V = np.zeros(n_basis*n_basis).reshape(n_basis,n_basis)
for i in xrange(0,n_basis):
        T[i,i] = ((i+1)*np.pi/L)**2/2
       # V[i,i] = V_0*np.inner(Chi[:,i].Chi[:,i]) #placeholder for potential

"""
Computation of the Coulomb integrals
"""
# Fourier coefficients
FourierCp=np.zeros(31)
FourierCn=np.zeros(31)
for g in xrange(0,31):
                FourierCp[g]=(HF.Vg(0.1,g,L))
                FourierCn[g]=(HF.Vg(0.1,-g,L))
# Coulomb integral matrix
Q=np.zeros(n_basis**4).reshape(n_basis,n_basis,n_basis,n_basis)
# Loop iterates for the integrals <pr|U|qs>
for p in xrange(1,n_basis+1):
        for q in xrange(1,n_basis+1):
                for r in xrange(1,n_basis+1):
                        for s in xrange (1,n_basis+1):
                                newQ=0+0j
                                newQ+=FourierCp[0]*HF.I_gmn(0,p,q)*HF.I_gmn(-0,r,s)
                                pg=1
                                while pg <=30:
                                        newQ+=FourierCp[pg]*HF.I_gmn(pg,p,q)*HF.I_gmn(-pg,r,s)+FourierCn[pg]*HF.I_gmn(-pg,p,q)*HF.I_gmn(pg,r,s)
                                        pg+=1
                                Q[p-1,r-1,q-1,s-1]=np.real(newQ/L*4)
print "Q calculated"
"""
Main program: Iterates making and solving the Fock matrix F untill
Convergence is achieved across all Fock levels
"""
OldFockEnergy=np.ones(n_basis)
FockEnergy=np.zeros(n_basis)
timesrun=0
P=HF.MakeDenseMat(C,n_basis,n_orbitals)
while np.sum(abs(FockEnergy-OldFockEnergy)) > 1.e-11:
        OldFockEnergy=FockEnergy
        FockEnergy=np.zeros(n_basis)
        for p in xrange(0,n_basis):
                for q in xrange(0,n_basis):
                        F[p,q]=T[p,q]
                        for r in xrange(0,n_basis):
                              for s in xrange(0,n_basis):
                                      # Excludes each orbital from its own sum
                                      if r==p and s==q:
                                              F[p,q]+=0.
                                      else:
                                              F[p,q]+=P[r,s]*Q[p,r,q,s]/2
        eigenvalues, eigenvectors = LA.eig(F)
        FockEnergy=eigenvalues[np.argsort(eigenvalues)]
        oldC=C
        C = eigenvectors[:,np.argsort(eigenvalues)]
        P=HF.MakeDenseMat(C,n_basis,n_orbitals)
        timesrun+=1
        
"""
Makes the spatial part of the wavefunction
"""
psi1=np.zeros(xsteps)
psi2=np.zeros(xsteps)
for i in xrange(0,n_basis):
        psi1+=C[i,0]*Chi[:,i]
        psi2+=C[i,1]*Chi[:,i]
psi1=psi1
psi2=psi2
#Overall Wavefunction PSI(x1,x2)=psi1(x1)psi2(x2)
PSI=np.zeros(xsteps**2).reshape(xsteps,xsteps)
for x1 in xrange(0,xsteps):
        for x2 in xrange(0,xsteps):
                PSI[x1,x2]=psi1[x1]*psi2[x2]
x1, x2 = np.meshgrid(x,x)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf = ax.plot_surface(x1,x2,PSI,cmap=cm.Spectral,linewidth=0,antialiased=False,shade=True)
plt.xlabel('x_1')
plt.ylabel('x_2')
Energy=0
Tenergy=0
Cenergy=0
"""
Energy of the orbitals is calculated
"""
for p in xrange(0,n_basis):
        for q in xrange(0,n_basis):
                Tenergy+=P[p,q]*T[p,q]/2
                for r in xrange(0,n_basis):
                        for s in xrange(0,n_basis):
                                if r==p and s==q:
                                        # again with the excluding
                                        Energy+=0.
                                else:
                                        Cenergy+=0.25*(P[p,q]*P[r,s]*Q[p,r,q,s])
Energy=Tenergy+Cenergy
print "Convergence was achieved in: ",timesrun," steps","\n","Energy is: ",Energy
plt.show()
