#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:      生成训练集合
@Date     :2022/06/27 20:37:42
@Author      :Jianguo Huang
@version      :1.0
'''
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
from gen_lengthscale_pair import sample_lengthscales_2D
np.random.seed(2022)

kernels = {'rbf':GPy.kern.RBF, 'exp':GPy.kern.Exponential, 
           'mat32':GPy.kern.Matern32, 'mat52':GPy.kern.Matern52}


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


def gen_dataForARanomdField(N,nx,ny,lx,ly,var,k,solver):


    num_samples = N
    ellx = lx
    elly = ly
    variance =var 
    k_ = k
    assert k_ in kernels.keys()
    kern = kernels[k_]



    #GPy kernel
    k=kern(input_dim = 2,
        lengthscale = [ellx, elly],
        variance = variance,
        ARD = True)

    
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

    #generate samples
    for i in range(num_samples):
        #display
        # if (i+1)%100 == 0:
        #     print("Generating sample "+str(i+1))
        
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
    

    
    Data = { 
            "randomfield_sample":inputs, # 
            "outputs":outputs,
            "diffusion_sample":diffusion_sample,  # 扩散系数
            "lengthscale_pair":[lx,ly]
    }
    return Data
if __name__ == "__main__":
    #############
    #生成训练数据
    ############
    l_N = 60   # lengthscale对的数量
    num_samples = 100   # 每个随机场采样的样本数
    nx=ny=32  #区域的网格划分
    k_ ='rbf'
        #data directory
    cwd = os.getcwd()
    data='data'
    datadir = os.path.abspath(os.path.join(cwd, data))
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    #save data
    # AllDataFile = k_+"_l_N_"+str(l_N)+"_num_samples_"+str(num_samples)+\
    #                 "_nx_"+str(nx)+"_ny_"+str(ny)+".pkl"
    # cellcentersFile = "cellcenters_"+"nx_"+str(nx)+"_"+str(ny)+".pkl"

    # Length_pair_array = sample_lengthscales_2D(100,1/32)
    solver = SteadyStateHeat2DSolver(nx=nx, ny=ny)
    cellcenters = solver.mesh.cellCenters.value.T
    # joblib.dump(cellcenters,os.path.join(datadir, cellcentersFile))
    # AllData = []  # 保存所有的
    start = time.time()
    # for i in range(l_N):
        

    #     l_pari = Length_pair_array[i]
    #     Data = gen_dataForARanomdField(num_samples,nx,ny,l_pari[0],l_pari[1],1,k_,solver)
    #     AllData.append(Data)
    #     if (i+1)%10==0:
    #         finish = time.time() - start
    #         print("Time (sec) to generate "+str(i+1)+" samples : " +str(finish))
    #         start = time.time()

    # joblib.dump(AllData,os.path.join(datadir, AllDataFile))

    #############
    #验证数据
    ############
    

    x= np.linspace(1/nx,1,10)
    y= np.linspace(1/ny,1,10)
    X,Y = np.meshgrid(x,y)
    coord = np.stack((X,Y), axis=2).reshape(-1,2) 
    ValDataFile = k_+"_l_N_"+str(coord.shape[0])+"_num_samples_"+str(num_samples)+\
                    "_nx_"+str(nx)+"_ny_"+str(ny)+"_val.pkl"
    AllData = []  # 保存所有的
    print(coord.shape[0])
    for i in range(coord.shape[0]):
        l_pari = coord[i]
        Data = gen_dataForARanomdField(num_samples,nx,ny,l_pari[0],l_pari[1],1,k_,solver)
        AllData.append(Data)
        if (i+1)%10==0:
            finish = time.time() - start
            print("Time (sec) to generate "+str(i+1)+" samples : " +str(finish))
            start = time.time()
    joblib.dump(AllData,os.path.join(datadir, ValDataFile))

