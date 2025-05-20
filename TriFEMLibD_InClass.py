#================================================================= 
# AE2220-II: Computational Modelling 
# TriFEMLib: A number of classes to assist in implementing
# finite-element methods on meshes of triangles
#================================================================= 
import math
import numpy as np
import matplotlib
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import scipy.interpolate as interp


#=================================================================
# TriOMesh class definition.
# TriMesh objects create and hold data for meshes of triangles.
# Either structured algebraic or Delaunay triangulation can be used.
# To use, create an empty object, set the parameters below 
# and call "loadMesh"
#=================================================================
class TriMesh(object):

  #=========================================================
  # Public data
  #=========================================================
  x1         =  0.0;                    # Left boundary position
  x2         =  5.0;                    # Right boundary position
  y1         =  0. ;                    # lower boundary position
  y2         =  2.0;                    # Upper boundary position
  dlx        =  1.5;                    # Delamination x position
  dly        =  0.5;                    # Delamination y position
  dll        =  0.4;                    # Delamination length
  dlt        =  0.2;                    # Delamination thickness
  bnx        =  20;                     # Background elements in x
  bny        =  10;                     # Background elements in y
  permn      = 1.0;                     # Nominal value of permittivity
  permd      = 0.001;                   # Permittivity of delamination
  dx         = (x2-x1)/float(bnx-1);    # Mesh spacing in x
  dy         = (y2-y1)/float(bny-1);    # Mesh spacing in y
  minTriArea = (dx+dy)/10000.;          # Minimum triangle size
  refine     = 1;                       # Mesh refinement factor

  vertices  = None;   # Mesh Vertices as list
  vertArray = None;   # Mesh Vertices as array
  elements  = None;   # Mesh Elements
  elemArray = None;   # Mesh Elements as array
  nVert     = None;   # Number of vertArray
  nElem     = None;   # Number of elements
  dtri      = None;   # Delaunay triangulation
  mask      = None;   # Triangle mask
  leftVI    = None;   # Left boundary vertex list
  rightVI   = None;   # Right boundary vertex list
  lowerVI   = None;   # Lower boundary vertex list
  upperVI   = None;   # Upper boundary vertex list
  
  #=========================================================
  # Object constructor
  #=========================================================
  def __init__(self):  
   created = 1;

  
  #=========================================================
  # Defines triangles removed from Delaunay triangulation
  #=========================================================
  def triMask(self,triangles):

    out           = []
    self.elements = []
    for points in triangles:
      a,b,c = points
#     pt=type(points);ps=points.shape;at=type(a);print(pt);print(ps);print(at);quit()
      va = self.vertices[a]
      vb = self.vertices[b]
      vc = self.vertices[c]
      x1 = float(va[0]); y1 = float(va[1]);
      x2 = float(vb[0]); y2 = float(vb[1]);
      x3 = float(vc[0]); y3 = float(vc[1]);
      Ae = 0.5*(x2*y3 + x1*y2 + x3*y1 - x3*y2 - x1*y3 - x2*y1);
      #if (Ae<self.minTriArea):
      if (Ae==0):
         out.append(True);
      else: 
         out.append(False);
         self.elements.append(points);

    return out

  #=========================================================
  # Loads the vertices and elements (call after setting
  # the desired parameters)
  #=========================================================
  def loadMesh(self,n,dlx,dll): 

    self.bnx   *= n;
    self.bny   *= n;
    self.dlx    = dlx;
    self.dll    = dll;
    self.refine = n;

    # Load background mesh
    xb = np.zeros(self.bnx+1);
    yb = np.zeros(self.bnx+1);
    nb = np.zeros(self.bny+1);
    xb[:] = np.linspace(self.x1, self.x2, self.bnx+1)
    nb[:] = np.linspace(0., 1., self.bny+1)
    yref = 0.1*math.sin(1.5*math.pi*2/(self.x2-self.x1));
    for i in range(self.bnx+1): 
      if (xb[i]<2):
         yb[i] = 1. + 0.1*math.sin(1.5*math.pi*xb[i]/(self.x2-self.x1));
      else:
         yb[i] = 1. + yref -0.2*(xb[i]-2.)/(self.x2-2.);

    self.vertices=[];
    self.leftVI=[];
    self.rightVI=[];
    self.lowerVI=[];
    self.upperVI=[];
    vi =0
    for j in range(self.bny+1):
      for i in range(self.bnx+1):
        if (i==0):        self.leftVI.append(vi);
        if (i==self.bnx): self.rightVI.append(vi);
        if (j==0):        self.lowerVI.append(vi);
        if (j==self.bny): self.upperVI.append(vi);
        self.vertices.append( (xb[i],nb[j]*yb[i]) );
        vi +=1;


    # Enrich near delamination
#   if (dlx>1.4):
#     self.addDelam(2*n,4*n);
   
    self.nVert=len(self.vertices);
    self.vertArray = np.asarray(self.vertices);
#   self.smoothVert(2,0.01);

    # Define initial Delaunay triangulation and mask bnd-only elements
    self.dtri = tri.Triangulation(self.vertArray[:,0],self.vertArray[:,1]);
    self.mask = self.triMask(self.dtri.triangles)
    self.dtri.set_mask(self.mask);

    # Refine mesh near perm transition 0<argument<1
    self.refineDelam(1.0)
    self.vertArray = np.asarray(self.vertices);

    # Dfine final Delaunay triangulation and mask bnd-only elements
    self.dtri = tri.Triangulation(self.vertArray[:,0],self.vertArray[:,1]);
    self.mask = self.triMask(self.dtri.triangles)
    self.dtri.set_mask(self.mask);

    # Finish element definition
    self.nElem=len(self.elements);
    self.elemArray = np.asarray(self.elements);


  #=========================================================
  # Returns the coordinates of a vertex
  #=========================================================
  def getVertCoords(self, vertInd):
    return self.vertices[vertInd];


  #=========================================================
  # Returns the relative radius from the delamination centre
  #=========================================================
  def delamRR(self,x,y):

    # argument in polar crd
    dx  = x-self.dlx;
    dy  = y-self.dly;
    dt  = math.atan2(dy,dx)
    dr  = math.sqrt(dx*dx+dy*dy)

    # radius of ellipse at same angle
    ed1 = self.dlt*math.sin(dt)
    ed2 = self.dll*math.cos(dt)
    er  = self.dll*self.dlt/math.sqrt(ed1*ed1+ed2*ed2)

    # return relative radius
    return dr/er;


  #=========================================================
  # Returns the local permittivity
  #=========================================================
  def getPerm(self,x,y):

    rr=self.delamRR(x,y)
    rt=0.4
    if (rr<rt):
       perm=self.permd;
    elif (rr>1.0):
       perm=self.permn;
    else:
       delp=self.permn-self.permd
       perm=0.5*delp*(1.-math.cos((rr-rt)*2.*math.pi))

    return perm



  #=========================================================
  # Refines near the change in permitivity
  #=========================================================
  def refineDelam(self,width):

    self.nElem=len(self.elements);
    for elemIndex in range(self.nElem):

      # get element vertices
      vertIndices = self.elements[elemIndex]
      v1 = self.vertices[vertIndices[0]]
      v2 = self.vertices[vertIndices[1]]
      v3 = self.vertices[vertIndices[2]]
      x1 = v1[0]; y1 = v1[1];
      x2 = v2[0]; y2 = v2[1];
      x3 = v3[0]; y3 = v3[1];

      # find centroid and its relative radius
      xc  = (x1+x2+x3)/3.
      yc  = (y1+y2+y3)/3.
      rrc = self.delamRR(xc,yc)
      
      # Update element array
      rt = min(abs(width),0.99)
      if (rrc>1-rt and rrc<1.+rt):

          # Add the centroid as a vertex
          self.vertices.append((xc,yc))
          self.nVert=len(self.vertices); 
          
          # retrive the vertex indices
          vi1=vertIndices[0];vi2=vertIndices[1];vi3=vertIndices[2]
          vic = self.nVert-1;

          # redefine original element and append two new elements
          er1 = np.array([vic,vi3,vi2]); 
          er2 = np.array([vi1,vi3,vic]); 
          er3 = np.array([vi1,vic,vi2]); 
          self.elements[elemIndex] = er1;
          self.elements.append(er2)
          self.elements.append(er3)
          self.nElem +=2




  #=========================================================
  # Returns the local permittivity for all vertices
  #=========================================================
  def getAllPerm(self):

    allPerm=np.zeros(self.nVert)
    return perm

  #=========================================================
  # Returns the local permittivity for all vertices
  #=========================================================
  def getAllPerm(self):

    allPerm=np.zeros(self.nVert)
    for i in range(self.nVert):
      vrt = self.vertices[i]
      allPerm[i] = self.getPerm(float(vrt[0]),float(vrt[1]));

    return allPerm;


  #=========================================================
  # Returns the distribution of voltage across an electrode
  #=========================================================
  def getElectrodeDist(self,x,xe,le):
    return 0.5 + 0.5*math.cos((x-xe)*2.*math.pi/le);



  #=========================================================
  # Prints the vertices and elements to a file
  #=========================================================
  def printMesh(self,basename="mesh"):
  
    mFile = open(basename+".txt", 'w')

    mFile.write("\n\nVertices:\n")
    for i in range(self.nVert):
      mFile.write("vi=");mFile.write(str(i));
      mFile.write(" xy=");mFile.write(str(self.vertArray[i,:]));
      mFile.write("\n")

    mFile.write("\n\nElements:\n")
    for e in range(self.nElem):
      mFile.write("ei=");mFile.write(str(e));
      mFile.write(" vert=");mFile.write(str(self.elements[e]));
      mFile.write("\n")


  #=========================================================
  # Plots the mesh
  #=========================================================
  def plotMesh(self):
  
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(111)
    ax.set_title("Mesh")
    ax.set_xlabel('x',size=14,weight='bold')
    ax.set_ylabel('y',size=14,weight='bold')
    plt.axes().set_aspect('equal', 'datalim')
    xy = np.asarray(self.vertices);
    plt.triplot(xy[:,0],xy[:,1],self.elements,'bo-');
#   plt.savefig('mesh.png',dpi=250)
    plt.show()



#**********************************************************************



#================================================================= 
# LinTriFESpace class definition.
# LinTriFESpace provides the mapping from local element data
# to the global system for a mesh of linear triangles. 
# It also provides boundary condition row and coordinate 
# information. One variable per vertex is assumed for now.
#================================================================= 
class LinTriFESpace(object):


  #=========================================================
  # Public data
  #=========================================================
  nVar          = 1;      # Number of variables (=1 for now)
  mesh          = None    # Link to mesh object
  nLeft         = None    # Number of rows for inner BCs
  nRight        = None    # Number of rows for outer BCs
  nLower        = None    # Number of rows for inner BCs
  nUpper        = None    # Number of rows for outer BCs
  leftDof       = None    # Global dof for lower BC
  rightDof      = None    # Global dof for upper BC
  lowerDof      = None    # Global dof for lower BC
  upperDof      = None    # Global dof for upper BC
  leftCoords    = None    # Coordinates for left vertices
  rightCoords   = None    # Coordinates for right vertices
  lowerCoords   = None    # Coordinates for left vertices
  upperCoords   = None    # Coordinates for right vertices
  sysDim        = None    # Global system dimension


  #=========================================================
  # Object constructor
  #=========================================================
  def __init__(self,mesh): 
  
    self.mesh          = mesh
    self.nVar          = 1;
    self.sysDim        = mesh.nVert;  # x nVar
    self.leftDof       = mesh.leftVI;
    self.rightDof      = mesh.rightVI;
    self.lowerDof      = mesh.lowerVI;
    self.upperDof      = mesh.upperVI;
    self.leftCoords    = []
    self.rightCoords   = []
    self.lowerCoords   = []
    self.upperCoords   = []
    self.nLeft         = len(mesh.leftVI);
    self.nRight        = len(mesh.rightVI);
    self.nLower        = len(mesh.lowerVI);
    self.nUpper        = len(mesh.upperVI);
    for i in range(self.nLeft):
       vi = int(mesh.leftVI[i]);
       self.leftCoords.append(mesh.vertices[vi]);
    for i in range(self.nRight):
       vi = int(mesh.rightVI[i]);
       self.rightCoords.append(mesh.vertices[vi]);
    for i in range(self.nLower):
       vi = int(mesh.lowerVI[i]);
       self.lowerCoords.append(mesh.vertices[vi]);
    for i in range(self.nUpper):
       vi = int(mesh.upperVI[i]);
       self.upperCoords.append(mesh.vertices[vi]);



  #=========================================================
  # Adds the element matrix to the global matrix
  #=========================================================
  def addElemMat(self, ei, elemMat, gloMat):

    # Retrieve the element vertex indices
    vertInd  = self.mesh.elements[ei];
    nVert    = vertInd.shape[0]

    # Add element matrix to the global matrix
    # In this case nVar=1 is assumed
    for m in range(nVert):
      for n in range(nVert):
        gloMat[vertInd[m],vertInd[n]] += elemMat[m,n];


  #=========================================================
  # Adds the element vector to the global vector
  #=========================================================
  def addElemVec(self, el, elemVec, gloVec):
  
    # Retrieve the node index of element vertices
    vertInd  = self.mesh.elements[el];
    nVert    = vertInd.shape[0]

    # Add element vector to the global vector. 
    # In this case nVar=1 is assumed
    for m in range(nVert):
      gloVec[vertInd[m]] +=  elemVec[m]


  #=========================================================
  # Prints a matrix and vector to two files
  #=========================================================
  def printMatVec(self, mat, vec, basename):
    np.savetxt(basename+"_mat.txt", mat)
    np.savetxt(basename+"_vec.txt", vec)


  #=========================================================
  # Determines the solution at a lower boundary x position
  #=========================================================
  def getLowerBndSoln(self, solVec, xmeas):

     # find closest points
     d1 = 20.;d2 =20.;i1=0;i2=0;x1=0.;x2=0.;
     for i in range(self.nLower): 
       xy  = self.lowerCoords[i]
       d   = math.fabs(xy[0]-xmeas)
       if (d<d1):
         i1 = i; d1=d; x1=xy[0];
       elif (d<d2):
         i2 = i; d2=d; x2=xy[0];
    
     # Interpolate
     print ("interpolating between x=",x1," and ",x2);
     row1  = self.lowerDof[i1]
     row2  = self.lowerDof[i2]
     umeas = (solVec[row1]*d2+solVec[row2]*d1)/(d1+d2);
     return umeas
         
   

  #=========================================================
  # Plots the mesh
  #=========================================================
  def plotMesh(self, plt, title=""):

    plt.set_title("Mesh")
#   plt.set_xlabel('x',size=14,weight='bold')
#   plt.set_ylabel('y',size=14,weight='bold')
    plt.set_aspect('equal');
    plt.set_xlim(-0.1,5.1); plt.set_ylim(-0.1,1.2);
    xy = np.asarray(self.mesh.vertices);
    vals=plt.triplot(xy[:,0],xy[:,1],self.mesh.elements,'b-',linewidth=0.5);
    return vals

  #=========================================================
  # Makes a contour plot of the solution 
  #=========================================================
  def plotSoln(self, plt, solVec, title=""):

    plt.set_title(title)
#   plt.set_xlabel('x',size=14,weight='bold')
#   plt.set_ylabel('y',size=14,weight='bold')
    plt.set_aspect('equal')
    plt.set_xlim(-0.1,5.1); plt.set_ylim(-0.1,1.2);
    xy = np.asarray(self.mesh.vertices);
    if xy.size < 10000:
     plt.triplot(xy[:,0],xy[:,1],self.mesh.elements,'b-',linewidth=0.5);
    vals=plt.tricontourf(self.mesh.dtri,solVec,cmap="jet")
    return vals


  #=========================================================
  # Makes a contour/vector plot of the solution 
  #=========================================================
  def plotSolnAndGrad(self, plt, solVec, title=""):

    plt.set_title(title)
#   plt.set_xlabel('x',size=14,weight='bold')
#   plt.set_ylabel('y',size=14,weight='bold')
    plt.set_aspect('equal');
    plt.set_xlim(-0.1,5.1); plt.set_ylim(-0.1,1.2);
    xy = np.asarray(self.mesh.vertices);
    tci = tri.CubicTriInterpolator(self.mesh.dtri, solVec);
    (Ex, Ey) = tci.gradient(self.mesh.dtri.x, self.mesh.dtri.y) 
    E_norm = np.sqrt(Ex**2 + Ey**2)
    vals=plt.tricontourf(self.mesh.dtri,solVec,cmap="jet")
    plt.quiver(self.mesh.dtri.x, self.mesh.dtri.y, -Ex/E_norm, -Ey/E_norm,
           units='xy', scale=20., zorder=3, color='blue',
           width=0.002, headwidth=2., headlength=2.)
    return vals



#**********************************************************************



#================================================================= 
# LinTriElement class definition.
# LinTriElement provides the shape and shape gradient values
# for a linear 2D triangle element defined in physical coordinates.
#================================================================= 
class LinTriElement(object):

  #=========================================================
  # Public data
  #=========================================================
  nFun        = 3      # Number of shape functions in this element.
  vertIndices = None   # Vertex indices
  vertCoords  = None   # Vertex coordinates
  area        = None   # Element area
  nIP         = None   # Number of integration points
  ipCoords    = None   # Coordinates for each integration point
  ipWeights   = None   # Quadrature weight for each ip
  ipScheme    = 4      # Integration scheme

  #=========================================================
  # Object constructor
  #=========================================================
  def __init__(self, mesh, elemIndex): 

    self.vertIndices = mesh.elements[elemIndex]
    v1 = mesh.vertices[self.vertIndices[0]]
    v2 = mesh.vertices[self.vertIndices[1]]
    v3 = mesh.vertices[self.vertIndices[2]]
    self.vertCoords = np.vstack((v1,v2,v3))

    x1 = v1[0]; y1 = v1[1];
    x2 = v2[0]; y2 = v2[1];
    x3 = v3[0]; y3 = v3[1];

    Ae = 0.5*(x2*y3 + x1*y2 + x3*y1 - x3*y2 - x1*y3 - x2*y1);

    self.psi1A = (x2*y3-x3*y2)/(2.*Ae);
    self.psi1B = (y2-y3)/(2.*Ae);
    self.psi1C = (x3-x2)/(2.*Ae);

    self.psi2A = (x3*y1-x1*y3)/(2.*Ae);
    self.psi2B = (y3-y1)/(2.*Ae);
    self.psi2C = (x1-x3)/(2.*Ae);

    self.psi3A = (x1*y2-x2*y1)/(2.*Ae);
    self.psi3B = (y1-y2)/(2.*Ae);
    self.psi3C = (x2-x1)/(2.*Ae);
   
    self.area = Ae; 
    self.setQuadrature()
    
  #=========================================================
  # Set the quadrature points and the weigths
  #=========================================================
  def setQuadrature(self):

    if self.ipScheme == 1:
      # One point scheme
      self.nIP = 1
      w = np.array([1.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 1./3.
      eta[0,1] = 1./3.
      eta[0,2] = 1./3.

    elif self.ipScheme == 2:
      # 3 point scheme 1
      self.nIP = 3
      w = np.array([1./3., 1./3., 1./3.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 1./2.; eta[0,1] = 1./2.; eta[0,2] = 0.
      eta[1,0] = 1./2.; eta[1,1] = 0.   ; eta[1,2] = 1./2.
      eta[2,0] = 0.   ; eta[2,1] = 1./2.; eta[2,2] = 1./2.

    elif self.ipScheme == 3:
      # 3 point scheme 2
      self.nIP = 3
      w = np.array([1./3., 1./3., 1./3.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 2./3.; eta[0,1] = 1./6.; eta[0,2] = 1./6.
      eta[1,0] = 1./6.; eta[1,1] = 2./3.; eta[1,2] = 1./6.
      eta[2,0] = 1./6.; eta[2,1] = 1./6.; eta[2,2] = 2./3.

    else:
      # 4 point scheme 1
      self.nIP = 4
      w = np.array([-27./48., 25./48., 25./48., 25./48.])
      eta = np.zeros((self.nIP,3))
      eta[0,0] = 1./3.;   eta[0,1] = 1./3.;   eta[0,2] = 1./3.
      eta[1,0] = 11./15.; eta[1,1] = 2./15.;  eta[1,2] = 2./15.
      eta[2,0] = 2./15.;  eta[2,1] = 11./15.; eta[2,2] = 2./15.
      eta[3,0] = 2./15.;  eta[3,1] = 2./15.;  eta[3,2] = 11./15.


    # Set the weights and transform to physical coordinates
    self.ipWeights = w * self.area
    self.ipCoords=np.zeros((self.nIP,2))
    for ip in range(self.nIP):
      for d in range(2):
         self.ipCoords[ip,d] = ( eta[ip,0]*self.vertCoords[0,d] 
                                +eta[ip,1]*self.vertCoords[1,d] 
                                +eta[ip,2]*self.vertCoords[2,d])
 

  #=========================================================
  # Returns the three shape functions at the point (x,y)
  #=========================================================
  def getShapes(self, x=None, y=None):

    psi    = np.zeros(3);
    psi[0] = self.psi1A + self.psi1B*x + self.psi1C*y;
    psi[1] = self.psi2A + self.psi2B*x + self.psi2C*y;
    psi[2] = self.psi3A + self.psi3B*x + self.psi3C*y;

    return psi


  #=========================================================
  # Returns the gradients of the three shape functions 
  # a the point (x,y). gradPsi[i] points to a vector 
  # with the x and y gradients of shape function i
  #=========================================================
  def getShapeGradients(self, x=None, y=None):

    gradPsi = np.zeros((3,2));
    gradPsi[0,0] = self.psi1B; gradPsi[0,1] = self.psi1C;
    gradPsi[1,0] = self.psi2B; gradPsi[1,1] = self.psi2C;
    gradPsi[2,0] = self.psi3B; gradPsi[2,1] = self.psi3C;

    return gradPsi

