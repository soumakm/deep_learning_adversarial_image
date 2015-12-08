# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 20:46:12 2015

@author: MercifulGod
"""
import numpy as np
import cPickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import add

def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))
    
def digit_set(s):
    """return 10 element vector with element 
    corresponding to a digit is set to 1
    """
    t = np.zeros((10,1))
    t[s] = 1.0
    return t

data_file = gzip.open('./data/mnist.pkl.gz', 'rb')
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

input_layer = 784
hidden = 15
out = 10
epochs = 5
mini_batch_size = 10
eta = 3.0
 
sizes = [input_layer, hidden, out]
num_layers = len(sizes)

biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)
             for x, y in zip(sizes[:-1], sizes[1:])]
#eta2 = np.abs([np.random.randn(sizes[0],1)])
eta2 = np.abs([ 1 for x in range(0,sizes[0])])
#p = train_data[0][0] + eta2
p = map(add,test_data[0][0],eta2)
p = np.reshape(p,(28,28))
plt.imshow(p, cmap = cm.Greys_r)
plt.show()
p = map(add,test_data[1][0],eta2)
p = np.reshape(p,(28,28))
plt.imshow(p, cmap = cm.Greys_r)
plt.show()
'''
eta2 = np.abs([np.random.randn(sizes[0],1)])
p = input_samples[0] + eta2/5
while s < 0:
    eta2 = eta2 + 0.01
    s = np.dot(eta2,weights[0])
    print(s)
for w0, w in zip(biases, weights):
    x = sigmoid(np.dot(w, x )+w0)
    '''