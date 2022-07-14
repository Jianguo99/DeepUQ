#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/06/27 14:35:20
@Author      :Jianguo Huang
@version      :1.0
'''

import numpy as np


def sample_lengthscales_2D(M,h):
    """
    @description  :  针对二维高斯过程，对lengthscale进行采样
    ---------
    @param  : M:采样数量
    -------
    @Returns  :
    -------
    """
    Length_pair_array = np.zeros((M,2))
    c= 0
    while c<M:
        l = np.random.rand(3)
        # print(l)
        if np.exp(-(l[0]+l[1]))> l[2]:
        # if l[0]+l[1]< l[2]:
            # 将length scale 标准化到[h,1]之间
            Length_pair_array[c]= [h+l[0]*(1-h),h+l[1]*(1-h)]
            c+=1

    return Length_pair_array


def sample_lengthscales_2D_Normal(M,h):
    """
    @description  :  针对二维高斯过程，对lengthscale进行采样
    ---------
    @param  : M:采样数量
    -------
    @Returns  :
    -------
    """
    Length_pair_array = np.zeros((M,2))
    c= 0
    while c<M:
        l = np.random.randn(2)*0.4
        l = [np.abs(ele) for ele in l]
        if 0< l[0] <1 and 0< l[1] <1: 
            print(l)
            Length_pair_array[c]= [h+l[0]*(1-h),h+l[1]*(1-h)]
            c+=1

    return Length_pair_array
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    Length_pair_array = sample_lengthscales_2D(100,1/32)
    # Length_pair_array = sample_lengthscales_2D_Normal(100,1/32)
    plt.scatter(Length_pair_array[:,0],Length_pair_array[:,1],s=1)
    plt.show()
    
