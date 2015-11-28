# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:27:58 2015

@author: soumak
"""
import load_mnist_data as data
import random
import numpy as np

input_layer = 784
hidden = 15
out = 10
 
epochs = 30
mini_batch_size = 10
eta = 3.0
 
 
sizes = [input_layer, hidden, out]
num_layers = len(sizes)
 
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)
             for x, y in zip(sizes[:-1], sizes[1:])]
                         
def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))
    
def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))    


def feedforward(x):
    global weights
    global biases

    for w0, w in zip(biases, weights):
        x = sigmoid(np.dot(w, x)+w0)
    return x    

def cost_derivative(output_activations, y):
    return (output_activations-y)

def evaluate(test_data):
    
    test_results = [(np.argmax(feedforward(x)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)
    
def update_mini_batch(mini_batch):
    global weights
    global biases
    global eta
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(weights, nabla_w)]
    biases = [b-(eta/len(mini_batch))*nb
                   for b, nb in zip(biases, nabla_b)]      
                   
def SGD(training_data, epochs, mini_batch_size, eta,
        test_data=None):

    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            update_mini_batch(mini_batch)
        if test_data:
            print "Epoch {0}: {1} / {2}".format(
                j, evaluate(test_data), n_test)
        else:
            print "Epoch {0} complete".format(j)        
            
def backprop(x, y):
    global weights
    global biases
    global num_layers
    
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in xrange(2, num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
 
def main():
    train_data, val_data, test_data = data.data_loader()
     
     
                     
    SGD(train_data, epochs, mini_batch_size, eta, test_data)
     
if __name__ == "__main__":
     main()
     