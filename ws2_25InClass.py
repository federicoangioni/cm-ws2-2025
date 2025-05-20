##================================================================= 
#
# AE2220-II: Computational Modelling 
# Main program for work session 2
#
# Line 92:  Definition of f for manufactured solution
# Line 110: Definition of Ke[i,j]
#
#=================================================================
# This code estimates the electric potential distribution 
# in a carbon-fibre composite part using the finite element method.
# Although individual carbon-fibre layers have strongly anisotropic 
# permittivities, we will assume that enough layers are present in 
# the part so the the electrostatic field can be determined assuming 
# a constant permittivity of 1, except in the neighbourhood 
# of the damage, where it falls rapidly to a small value.
#=================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
import TriFEMLibD_InClass

#=========================================================
# Input parameters
#=========================================================
n=1                   # Mesh refinement factor
dlx=1.0               # Delamination x location
dll=0.3               # Delamination length

#=========================================================
# Fixed parameters
#=========================================================
xm1=1.0               # Measurement location 1
xm2=4.0               # Measurement location 2



#=========================================================
# Create the mesh 
#=========================================================
mesh = TriFEMLibD_InClass.TriMesh();
mesh.loadMesh(n,dlx,dll)
print ("\nMesh: nVert=",mesh.nVert,"nElem=",mesh.nElem);
#mesh.plotMesh(); 


#=========================================================
# Create a finite-element space.
# This object maps the degrees of freedom in an element
# to the degrees of freedom of the global vector.
#=========================================================
fes = TriFEMLibD_InClass.LinTriFESpace(mesh)


#=========================================================
# Prepare the global left-hand matrix, right-hand vector
# and solution vector
#=========================================================
sysDim = fes.sysDim
LHM    = np.zeros((sysDim,sysDim));
RHV    = np.zeros(sysDim);
solVec = np.zeros(sysDim);


#=========================================================
# Assemble the global left-hand matrix and
# right-hand vector by looping over the elements
print ("Assembling system of dimension",sysDim);
#=========================================================
for elemIndex in range(mesh.nElem):

  #----------------------------------------------------------------
  # Create a FiniteElement object for 
  # the element with index elemIndex
  #----------------------------------------------------------------
  elem = TriFEMLibD_InClass.LinTriElement(mesh,elemIndex)

  #----------------------------------------------------------------
  # Initialise the element vector and matrix to zero.
  # In this case we have only one unknown varible in the PDE (u),
  # So the element vector dimension is the same as
  # the number of shape functions (psi_i)  in the element.
  #----------------------------------------------------------------
  evDim   = elem.nFun
  elemVec = np.zeros((evDim))
  elemMat = np.zeros((evDim,evDim))

  #----------------------------------------------------------------
  # Evaluate the shape function integrals in the vector and matrix 
  # by looping over integration points (integration by quadrature)
  # int A = sum_ip (ipWeight*A_ip) where A is the function to be 
  # integrated and ipWeight is the weight of an integration point
  #----------------------------------------------------------------
  for ip in range(elem.nIP):

    # Retrieve the coordinates and weight of the integration point
    xIP      = elem.ipCoords[ip,0] 
    yIP      = elem.ipCoords[ip,1] 
    ipWeight = elem.ipWeights[ip];

    # Compute the local value of the source term, f
    fIP = 0.;

    # Retrieve other values evaluated at this integration point (ip)
    # - perm is the value of permittivity at this ip
    # - psi[i] is the value of the function psi_i at this ip.
    # - gradPsi[i] is a vector contraining the x and y
    #   gradients of the function psi_i at this ip
    #   e.g.
    #     gradPsi[2][0] is the x gradient of shape 2 at point xIP,yIP
    #     gradPsi[2][1] is the y gradient of shape 2 at point xIP,yIP
    perm    = mesh.getPerm(xIP,yIP);      
    psi     = elem.getShapes(xIP,yIP)
    gradPsi = elem.getShapeGradients(xIP,yIP)

    # Add this ip's contribution to the integrals in the
    # element vector and matrix
    for i in range(evDim):
      elemVec[i] += ipWeight*psi[i]*fIP;   # Right-hand side of weak form
      for j in range(evDim):
        # ***** Change the line below for the desired left-hand side
        elemMat[i,j] -= ipWeight*perm; 

 

  #----------------------------------------------------------------
  # Add the completed element matrix and vector to the system
  #----------------------------------------------------------------
  fes.addElemMat(elemIndex, elemMat, LHM )
  fes.addElemVec(elemIndex, elemVec, RHV ) 


#=========================================================
print ("Applying boundary conditions");
#=========================================================
# Left boundary conditions
#=========================================================
for i in range(fes.nLeft):
   row = fes.leftDof[i];
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = 10.

for i in range(fes.nRight):
   row = fes.rightDof[i];
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = 0.



#=========================================================
print ("Solving the system");
#=========================================================
#fes.printMatVec(LHM,RHV,"afterConstraints")
solVec = np.linalg.solve(LHM, RHV)


#=========================================================
# Find and output potential and the difference from the
# reference values at the measurement points 
#=========================================================
um1=fes.getLowerBndSoln(solVec,xm1);
um2=fes.getLowerBndSoln(solVec,xm2);
print ("\n----------------------------------------");
print ("n=",n," dlx,dll=( ",dlx," ",dll," )");
print ("");
print ("um1 (Potential at xm1) = ",um1);
print ("um2 (Potential at xm2) = ",um2);
print ("")
print ("----------------------------------------\n");


#=========================================================
# Plot the results
#=========================================================
fig = plt.figure(figsize=(11,8))
allPerm = mesh.getAllPerm()
ax1 = plt.subplot2grid((2,5), (0,0), rowspan=1, colspan=5)
sp1 = fes.plotSoln(ax1,allPerm,"Permittivity")
ax2 = plt.subplot2grid((2,5), (1,0), rowspan=1, colspan=5)
sp2 =fes.plotSolnAndGrad(ax2,solVec,"Solution")
fig.colorbar(sp1,ax=ax1)
fig.colorbar(sp2,ax=ax2)
#plt.savefig('ndtl.png',dpi=250)
plt.show()
