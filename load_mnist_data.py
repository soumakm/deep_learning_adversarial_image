# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:53:49 2015

@author: MercifulGod
"""
#### Libraries
# Standard library
import numpy as np
import cPickle
import gzip

def digit_set(s):
    """return 10 element vector with element 
    corresponding to a digit is set to 1
    """
    t = np.zeros((10,1))
    t[s] = 1.0
    return t
def data_loader():
    """training data : (a sample with 784 features, 10 element vector with element for a give digit is set)
    example: ([...],[0, 0, 0, 0, 1, 0, 0, 0,0,0] ) here the second vector is equivalent to  digit 6
    """
    data_file = gzip.open('mnist.pkl.gz', 'rb')
    tr_data, val_data, ts_data = cPickle.load(data_file)
    data_file.close()
    input_samples = [np.reshape(x,(784,1)) for x in tr_data[0] ]
    input_targets = [digit_set(k) for k in tr_data[1]]
    train_data = zip(input_samples, input_targets)
    "validation data"
    valid_input = [np.reshape(x,(784,1)) for x in val_data[0]]
    v_data  = zip(valid_input, val_data[1])
    "test data"
    test_in = [np.reshape(x, (784,1)) for x in ts_data[0]]
    test_data = zip(test_in, ts_data[1])
    return train_data, v_data, test_data
    
if __name__ == "__main__":
    traing, validation, test = data_loader()
    

