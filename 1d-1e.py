"""
This program aims to solve the 1D electron in a box through
variational methods using matrix diagonalisation. This is
computationally expensive but a good working basis for extending
the model
"""
# Importing packages
import numpy as np
import scipy as sci
import scipy.linalg as LA
import matplotlib.pyplot as plt
# Basic parameters
L = 2. # length of the box in a.u.
Nmax = 20. # number of basis vectors
xsteps = 1000. # number of divisions in x
delx = L/xsteps # increment in x
V0 = np.zeros((xsteps)) # Potential inside the well if I feel like playing
V0[249:749]=80.
#V0[748:-1]=80.
x = np.linspace(0,L,xsteps) # 1D array denoting position along x
# Compose basis vectors
# phi is an array storing our basis functions in rows
# Basis functions are sin solutions to the infinite well
phi = np.zeros((Nmax,xsteps))
for i in xrange(1,int(Nmax+1)):
        for j in xrange(0,int(xsteps)):
                phi[i-1,j]=np.sin(i*np.pi*x[j]/L)
                phi[i-1,-1]=0. # boundary condition
# Normalising our values
for i in xrange(0,int(Nmax)):
        phi[i,:]=phi[i,:]/np.sqrt(np.inner(phi[i,:],phi[i,:]))
# Populate the kinetic energy and potential operators
K = np.zeros((xsteps,xsteps))
V = np.zeros((xsteps,xsteps))
for i in xrange(0,int(xsteps)):
        K[i,i] = 1./(delx**2)
        V[i,i] = V0[i]
        if i < int(xsteps-1):
                K[i,i+1] = -1./(2.*delx**2)
                K[i+1,i] = K[i,i+1]
# Construct the Hamiltonian in terms of the basis set for psi
H = np.zeros((Nmax,Nmax))
for i in xrange(0,int(Nmax)):
        for j in xrange(0,int(Nmax)):
                phi_=np.inner(K,phi[j])
                phi_[0]=0.
                phi_[-1]=0.
                H[i,j]=np.inner(phi[i,:],phi_)+np.inner(phi[i,:],np.inner(V,phi[j,:]))
                # Actually this can be shorter as it will be symmetric about the diagonal

eigenvalues, eigenvectors = LA.eig(H)
E = np.sort(eigenvalues)
eigenvectors=np.transpose(eigenvectors)
np.set_printoptions(precision=3)
print "Eigenvectors: ", eigenvectors, "\n Eigenenergies: ", eigenvalues
Edisp=E[0:5]
# The eigenvectors weight the linear combination of the basis vectors for psi
psi = np.zeros((Nmax,xsteps))
for i in xrange(0,int(Nmax)):
        for j in xrange(0,int(Nmax)):
                psi[i,:]+=eigenvectors[i,j]*phi[j,:]
psi = psi[np.argsort(eigenvalues)]
# Calculated ideal E
Eideal = np.zeros((5))
for n in xrange (0,5):
        Eideal[n]=(n+1.)**2.*np.pi**2./(2.*L**2.)
# Compare calculated energies with ideal
print "Numerical Energies: ", Edisp, "\n Ideal energies: ", Eideal
for i in xrange(0,3):
        plt.plot(x,psi[i])
plt.show()
