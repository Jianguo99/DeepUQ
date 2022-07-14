  #!/usr/bin/env python
import argparse 
from solver import SteadyStateHeat2DSolver
import numpy as np
import os
import GPy
import matplotlib.pyplot as plt
from fipy import *    # FVM
from scipy.interpolate import griddata
from pdb import set_trace as keyboard
import time
import joblib
#parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-N', dest = 'N', type = int, 
                    default = 10, help  = 'Number of samples of the random inputs')
parser.add_argument('-nx', dest = 'nx', type =  int, 
                    default = 100, help = 'Number of FV cells in the x direction.')
parser.add_argument('-ny', dest = 'ny', type = int, 
                    default = 100, help = 'Number of FV cells in the y direction.')
parser.add_argument('-lx', dest = 'lx', type = float, 
                    default = 0.446, help = 'Lengthscale of the random field along the x direction.')
parser.add_argument('-ly', dest = 'ly', type = float, 
                    default = 0.789, help = 'Lengthscale of the random field along the y direction.')
parser.add_argument('-var', dest = 'var', type = float, 
                    default = 1., help = 'Signal strength (variance) of the random field.')
parser.add_argument('-k', dest = 'k', type = str, 
                    default = 'exp', help = 'Type of covariance kernel (rbf, exp, mat32 or mat52)')
args = parser.parse_args()
kernels = {'rbf':GPy.kern.RBF, 'exp':GPy.kern.Exponential, 
           'mat32':GPy.kern.Matern32, 'mat52':GPy.kern.Matern52}

num_samples = args.N
nx = args.nx
ny = args.ny
ellx = args.lx
elly = args.ly
variance = args.var 
k_ = args.k
assert k_ in kernels.keys()
kern = kernels[k_]

#define a mean function
def mean(x):
    """
    Mean of the permeability field. 

    m(x) = 0. 
    """
    n = x.shape[0]
    return np.zeros((n, 1))

def q(x):
    n = x.shape[0]
    s = np.zeros((n))
    return s

#data directory
cwd = os.getcwd()
data='data'
datadir = os.path.abspath(os.path.join(cwd, data))
if not os.path.exists(datadir):
    os.makedirs(datadir)

#GPy kernel
k=kern(input_dim = 2,
       lengthscale = [ellx, elly],
       variance = variance,
       ARD = True)

##define the solver object
solver = SteadyStateHeat2DSolver(nx=nx, ny=ny)
cellcenters = solver.mesh.cellCenters.value.T
joblib.dump(cellcenters,os.path.join(datadir, 'cellcenters.npy'))
# np.save(os.path.join(datadir, 'cellcenters.npy'), cellcenters)

#get source field 
source = q(cellcenters)

#get covariance matrix and compute its Cholesky decomposition
m=mean(cellcenters)
C=k.K(cellcenters) + 1e-6*np.eye(cellcenters.shape[0])  # 生成协方差矩阵
L=np.linalg.cholesky(C)

#define matrices to save results 
inputs = np.zeros((num_samples, nx, ny))
outputs = np.zeros((num_samples, nx, ny))
diffusion_sample = np.zeros((num_samples, nx, ny))

start = time.time()
#generate samples
for i in range(num_samples):
    #display
    if (i+1)%100 == 0:
        print("Generating sample "+str(i+1))
    
    #generate a sample of the random field input
    z =np.random.randn(cellcenters.shape[0], 1)
    f = m + np.dot(L, z)   
    sample = np.exp(f[:, 0])
    #solve the PDE  
    solver.set_coeff(C=sample)   #set diffusion coefficient. 
    solver.set_source(source=source)   #set source term. 
    solver.solve()  

    #save data 
    diffusion_sample[i] = sample.reshape((nx, ny))
    inputs[i] = f.reshape((nx, ny))
    outputs[i] = solver.phi.value.reshape((nx, ny))

#end timer
finish = time.time() - start
print("Time (sec) to generate "+str(num_samples)+" samples : " +str(finish))

#save data
datafile = k_+"_lx_"+str(ellx).replace('.', '')+\
            "_ly_"+str(elly).replace('.', '')+\
            "_v_"+str(variance).replace('.', '')+".pkl"

Data = { 
        "inputs":inputs,
        "outputs":outputs,
        "diffusion_sample":diffusion_sample,
        "nx":nx,
        "ny":ny,
        "lx":ellx,
        "ly":elly,
        "var":variance
}
print(os.path.join(datadir,datafile))
joblib.dump(Data,os.path.join(datadir,datafile))




