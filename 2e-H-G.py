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
import scipy.integrate
import HFroutines as HF
"""
Starting parameters
"""
# Box length
L = 2
# Number of basis functions
n_basis= 5
# Number of spatial orbitals
n_orbitals = 1
# Divisions of x in L
xsteps = 100*n_basis
# Potential inside the box
V_0 = np.zeros((xsteps))
# Array of x values
x = np.linspace(0,L,xsteps)

F = np.zeros(n_basis*n_basis,dtype=complex).reshape(n_basis,n_basis)
C = np.ones(n_basis*n_basis,dtype=complex).reshape(n_basis,n_basis)
for i in xrange(0,n_basis):
        C[i,i]=1.
C = HF.normCol(C,n_basis)
"""
Next stage involves composition of basis functions
Will use the solutions to the 1D infinite well which
Take the form Chi_n(x)=sin(n*pi*x/L)
"""
# The ith column of chi will be the ith basis function
Chi = np.zeros((xsteps*n_basis)).reshape(xsteps,n_basis)
for i in xrange(0,n_basis):
        Chi[:,i] = np.sin((i+1)*np.pi*x/L)
        # Correct boundary condition for finite resolution
        Chi[-1,:] = 0.
        # Normalise. Do I need to?
        Chi[:,i] = Chi[:,i]/np.sqrt(np.inner(Chi[:,i],Chi[:,i]))

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
FourierC=np.zeros(201,dtype=complex)
for g in xrange(0,201):
        FourierC[g]=(HF.Vg(0.1,g-100))
# Coulomb integral matrix
Q=np.zeros(n_basis**4,dtype=complex).reshape(n_basis,n_basis,n_basis,n_basis)
# Loop iterates for the integrals <pr|U|qs>
for p in xrange(0,n_basis):
        for q in xrange(0,n_basis):
                for r in xrange(0,n_basis):
                        for s in xrange (0,n_basis):
                                # Includes more of the sum until convergance
                                OldQ=-1
                                newQ=FourierC[100]*HF.I_gmn(0,p+1,q+1)*HF.I_gmn(-0,r+1,s+1)
                                g=1
                                while abs(newQ-OldQ)>1e-9:
                                        newQ+=FourierC[100+g]*HF.I_gmn(g,p+1,q+1)*HF.I_gmn(-g,r+1,s+1)+FourierC[100-g]*HF.I_gmn(g,p+1,q+1)*HF.I_gmn(-g,r+1,s+1)
                                        if g%5==0:
                                                OldQ=newQ
                                        g+=1
                                Q[p,q,r,s]=newQ/L*4 
"""
Main program: Iterates making and solving the Fock matrix F untill
Convergence is achieved across all Fock levels
"""
OldFockEnergy=np.ones(n_basis)
FockEnergy=np.zeros(n_basis)
timesrun=0
while np.sum(abs(FockEnergy-OldFockEnergy)) > 1.e-11:
        OldFockEnergy=FockEnergy
        P=HF.MakeDenseMat(C,n_basis,n_orbitals)
        FockEnergy=np.zeros(n_basis,dtype=complex)
        for p in xrange(0,n_basis):
                for q in xrange(0,n_basis):
                        F[p,q]=T[p,q]
                        for r in xrange(0,n_basis):
                              for s in xrange(0,n_basis):
                                        F[p,q]+=P[s,r]*Q[p,q,r,s]
        eigenvalues, eigenvectors = LA.eig(F)
        FockEnergy=eigenvalues[np.argsort(eigenvalues)]
        oldC=C
        C = eigenvectors[:,np.argsort(eigenvalues)]
        timesrun+=1
        
"""
Makes the spatial part of the wavefunction
"""
psi1=np.zeros(xsteps,dtype=complex)
psi2=np.zeros(xsteps,dtype=complex)
for i in xrange(0,n_basis):
        psi1+=C[i,0]*Chi[:,i]
        psi2+=C[i,0]*Chi[:,i]
psi1=psi1*np.conj(psi1)
psi2=psi2*np.conj(psi2)
# Overall Wavefunction PSI(x1,x2)=psi1(x1)psi2(x2)
PSI=np.zeros(xsteps**2).reshape(xsteps,xsteps)
for x1 in xrange(0,xsteps):
        for x2 in xrange(0,xsteps):
                PSI[x1,x2]=psi1[x1]*psi2[x2]
x1, x2 = np.meshgrid(x,x)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf = ax.plot_surface(x1,x2,PSI)
plt.xlabel('x_1')
plt.ylabel('x_2')
Energy=0
"""
Energy of the orbitals is calculated
"""
for p in xrange(0,n_basis):
        for q in xrange(0,n_basis):
                Energy+=P[p,q]*T[p,q]
                for r in xrange(0,n_basis):
                        for s in xrange(0,n_basis):
                                Energy+=0.5*(P[p,q]*P[r,s]*Q[p,q,r,s])
print "Convergence was achieved in: ",timesrun," steps","\n","Energy is: ",Energy
plt.show()
