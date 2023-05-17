#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data file readout in the appropriate form

@author: Ksenija Kovalenka
"""
import numpy as np

def load_data():
    npartitions = 10000
    nkx = 11
    nky = 11
    nkz = 2
    nkpt = nkx*nky*nkz
    
    
    reals = np.zeros(shape=(nkpt,npartitions))
    complexs = np.zeros(shape=(nkpt,npartitions))
    alpha = np.zeros(shape=(nkpt,npartitions))
    
    
    # file = "NN_data.dat"
    file = "NN_data_equal.dat"
    
    
    data = open(file, 'r')
    
    for a in range(0, npartitions):
        for kz in range(0, nkpt):
            reals[kz,a] = float(data.read(10))
            complexs[kz,a] = float(data.read(13))
            alpha[kz,a] = float(data.read(7))
    data.close()
    
# =============================================================================
#     hoppings_combined = np.zeros((npartitions,484))
#     phases_classification = np.zeros((npartitions,1))
#     
# =============================================================================
# =============================================================================
#     for i in range(npartitions):
#         combined = np.array([reals[:,i], complexs[:,i]]).reshape((1,484))
#         hoppings_combined[i] = combined
#         if alpha[0,i] < 0.77:
#             phases_classification[i] = 0
#         else:
#             phases_classification[i] = 1
# =============================================================================
    
    data_tensor = np.zeros((npartitions, 2*nkz, nkx, nky))
    phases_classification = np.zeros((npartitions,1))

    for i in range(npartitions):
        reals_1 = np.reshape(reals[::2,i], (11,11))
        reals_0 = np.reshape(reals[1::2,i], (11,11))
        complexs_1 = np.reshape(complexs[::2,i], (11,11))
        complexs_0 = np.reshape(complexs[1::2,i], (11,11))
        
        data_tensor[i, 0] = reals_1
        data_tensor[i, 1] = reals_0
        data_tensor[i, 2] = complexs_1
        data_tensor[i, 3] =  complexs_0
        
        
        if alpha[0,i] < 0.77:
            phases_classification[i] = 0
        else:
            phases_classification[i] = 1
    print(phases_classification[5000-5:5000+5])
    return data_tensor, phases_classification

# =============================================================================
#     train_data = []
#     test_data = []
#     for a in range(0, npartitions):
#         alpha = (a-1)/(npartitions-1)
#         
#         if (alpha >= 0.000 and alpha < 0.400 or 
#             alpha >= 0.500 and alpha <= 0.770):
#             train_data.append((np.reshape(reals[:,a], (nkpt, 1)), np.array([[1],[0]])))
#             train_data.append((np.reshape(complexs[:,a], (nkpt, 1)),
#                                                np.array([[1],[0]])))
#         elif (alpha >= 0.400 and alpha < 0.500):
#             test_data.append((np.reshape(reals[:,a], (nkpt, 1)), 0))
#             test_data.append((np.reshape(complexs[:,a], (nkpt, 1)),
#                                               0))
#     
#         elif (alpha >= 0.785 and alpha < 0.795 or
#               alpha >= 0.850 and alpha < 0.900):
#             test_data.append((np.reshape(reals[:,a], (nkpt, 1)), 1))
#             test_data.append((np.reshape(complexs[:,a], (nkpt, 1)),
#                                               1))
#         else:
#             train_data.append((np.reshape(reals[:,a], (nkpt, 1)), np.array([[1],[0]])))
#             train_data.append((np.reshape(complexs[:,a], (nkpt, 1)),
#                                                np.array([[1],[0]])))
#     
#     return (train_data, test_data)
# =============================================================================
    



