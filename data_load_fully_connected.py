
"""
Created on Tue Mar  7 17:59:53 2023

@author: ksenijakovalenka
"""
import numpy as np

def load_data_flat():
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
    
    hoppings_combined = np.zeros((npartitions,484))
    phases_classification = np.zeros((npartitions,1))
    
    for i in range(npartitions):
        combined = np.array([reals[:,i], complexs[:,i]]).reshape((1,484))
        hoppings_combined[i] = combined
        if alpha[0,i] < 0.77:
            phases_classification[i] = 0
        else:
            phases_classification[i] = 1

    return hoppings_combined, phases_classification
