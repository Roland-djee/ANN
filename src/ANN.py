#!/usr/bin/py
""" ANN is a rudimentary example of artificial neural network
for a comprehensive idea of the general process based on "The Hacker's
Guide to Neural Networks" on GitHub.
@author Dr. Roland Guichard <r.guichard@ucl.ac.uk>
@todo: (always) Document code
"""

from math import *
import numpy as np
import scipy as sc
import random as rand


class axon:
    """Basic axon definition
    which carries the value and the gradient
    """
    def __init__(self, value, gradient):
        self.value    = value
        self.gradient = gradient

class neuronSum:
    """Basic neuron for the sum
    Arguments:
        a (float)
        b (float)
    Returns:
        The sum of two arguments a and b
    """
    #~ def __init__(self, a, b):
        #~ self.a   = axon(a, 0.)
        #~ self.b   = axon(b, 0.)
    def forwardSum(self, a, b):
        self.a   = a
        self.b   = b
        self.Sum = axon(self.a.value + self.b.value, 0.)
        return  self.Sum
    def backwardProp(self):
        """Chain rule the local gradient with the output gradient
        """
        self.a.gradient = self.a.gradient + 1. * self.Sum.gradient
        self.b.gradient = self.b.gradient + 1. * self.Sum.gradient
        
class neuronMul:
    """Basic neuron for the multiplication
    Arguments:
        a (float)
        b (float)
    Returns:
        The product of two arguments a and b
    """
    #~ def __init__(self, a, b):
        #~ self.a   = axon(a, 0.)
        #~ self.b   = axon(b, 0.)
    def forwardProd(self, a, b):
        self.a    = a
        self.b    = b
        self.Prod = axon(self.a.value * self.b.value, 0.)
        return self.Prod
    def backwardProp(self):
        """Chain rule the local gradient with the output gradient
        """
        self.a.gradient = self.a.gradient + self.b.value * self.Prod.gradient
        self.b.gradient = self.b.gradient + self.a.value * self.Prod.gradient

def sigmoid(x):
    """Definition of the sigmoid function
    """
    return 1. / (1. + exp(-x))

class neuronSigmoid:
    """Basic neuron for the sigmoid function
    Arguments:
        a (float)
    Returns:
        1 / (1 + exp(-x))
    """
    #~ def __init__(self, a):
        #~ self.a = axon(a, 0.)
    def forwardSig(self, a):
        self.a   = a
        self.Sig = axon(sigmoid(self.a.value), 0.)
        return self.Sig
    def backwardProp(self):
        """Chain rule the local gradient with the output gradient
        """
        s = self.Sig.value
        self.a.gradient = self.a.gradient + (s * (1. - s)) * self.Sig.gradient
    
class circuitForSVM :
    """Basic circuit composition of neurons to develop f(x) = a*x + b*y +z
    for Support Vector Machine (SVM)
    """
    def __init__(self):
        self.neuronSum1 = neuronSum()
        self.neuronSum2 = neuronSum()
        self.neuronMul1 = neuronMul()
        self.neuronMul2 = neuronMul()
    def forwardCircuit(self, x, y, a, b, c):
        self.ax = self.neuronMul1.forwardProd(a, x)
        self.by = self.neuronMul2.forwardProd(b, y)
        self.axpby = self.neuronSum1.forwardSum(self.ax, self.by)
        self.axpbypc = self.neuronSum2.forwardSum(self.axpby, self.by)
        return self.axpbypc
    def backwardProp(self, gradientFromTop):
        self.axpbypc.gradient    = gradientFromTop
        self.neuronSum2.backwardProp()
        self.neuronSum1.backwardProp()
        self.neuronMul2.backwardProp()
        self.neuronMul1.backwardProp() 
        
class SVM:
    """ Support Vector Machine based upon the circuit
    """
    
    def __init__(self):
        """ Random initial parameters
        """
        self.a = axon(1.0, 0.)
        self.b = axon(-2., 0.)
        self.c = axon(-1.0, 0.)
        self.circuit = circuitForSVM()
        #~ self.circuit.forwardCircuit()
    def forwardSVM(self, x, y):
        self.circuitOutput = self.circuit.forwardCircuit(x, y, self.a, self.b, self.c)
        return self.circuitOutput
    def backwardProp(self, label):
        self.a.gradient = 0. # Reset the gradients
        self.b.gradient = 0.
        self.c.gradient = 0.
        
        # Defines the pull
        pull = 0.
        if (label == 1 and self.circuitOutput.value < 1.):
            pull = 1
        elif (label == -1 and self.circuitOutput.value > -1.):
            pull = -1
        self.circuit.backwardProp(pull)
        
        # Regularisation pulls for parameters
        self.a.gradient = self.a.gradient - self.a.value
        self.b.gradient = self.b.gradient - self.b.value
    def parameterUpdate(self):
        step = 0.01
        self.a.value = self.a.value + step * self.a.gradient
        self.b.value = self.b.value + step * self.b.gradient
        self.c.value = self.c.value + step * self.c.gradient
    def learnFrom(self, x, y, label):
        self.forwardSVM(x, y)
        self.backwardProp(label)
        self.parameterUpdate()
        
def evalTrainingAccuracy(data, labels, svm):
    """ Determines the accuracy of the classification
    """
    numCorrect = 0.
    for i in range(len(data)):
        x = axon(data[i][0], 0.)
        y = axon(data[i][1], 0.)
        label = labels[i]
        
        # see if prediction matches the provided label
        predictedLabel = svm.forwardSVM(x, y).value
        if (predictedLabel > 0):
            predictedLabel = 1
        else:
            predictedLabel = -1
        if (predictedLabel == label):
            numCorrect = numCorrect + 1.
    return numCorrect/len(data)

def main():
    """Main method
    Arguments:
    Returns:
    """
    
    data   = [] 
    labels = []
    
    data.append([1.2, 0.7])
    labels.append(1)
    data.append([-0.3, -0.5])
    labels.append(-1)
    data.append([3., 0.1])
    labels.append(1)
    data.append([-0.1, -1.0])
    labels.append(-1)
    data.append([-1., 1.1])
    labels.append(-1)
    data.append([2.1, -3.0])
    labels.append(1)

    svm = SVM()
    
    for iter in range(500):
        i = rand.randint(0, 5)
        x = axon(data[i][0], 0.)
        y = axon(data[i][1], 0.)
        label = labels[i]
        svm.learnFrom(x, y, label)
        if (iter % 25 == 0):
            print ("training accuracy at {:} : {:}").format(iter, evalTrainingAccuracy(data, labels, svm))
    
    
    # Initial Forward pass
    #~ a    = axon(1., 0.)
    #~ b    = axon(2., 0.)
    #~ c    = axon(-3., 0.)

    #~ x    = axon(-1., 0.)
    #~ y    = axon(3., 0.)
    
    #~ nS1 = neuronSum()
    #~ nS2 = neuronSum()
    #~ nP1 = neuronMul()
    #~ nP2 = neuronMul()
    
    #~ nSig1 = neuronSigmoid()
    
    #~ ax = nP1.forwardProd(a, x)
    #~ by = nP2.forwardProd(b, y)
    
    #~ axpby  = nS1.forwardSum(ax, by)
    #~ axpbpc = nS2.forwardSum(axpby, c)     
    
    #~ s = nSig1.forwardSig(axpbpc)
    
    #~ print s.value
    
    
    # Backward propagation
    
    #~ s.gradient = 1.
    
    #~ nSig1.backwardProp()
    #~ nS2.backwardProp()
    #~ nS1.backwardProp()
    #~ nP2.backwardProp()
    #~ nP1.backwardProp()
    
    # Second forward pass
    
    #~ step = 0.01
    
    #~ print a.gradient
    #~ print b.gradient
    #~ print c.gradient
    #~ print x.gradient
    #~ print y.gradient
    
    #~ a.value = a.value + step * a.gradient
    #~ b.value = b.value + step * b.gradient
    #~ c.value = c.value + step * c.gradient
    #~ x.value = x.value + step * x.gradient
    #~ y.value = y.value + step * y.gradient
    
    
    #~ ax = nP1.forwardProd(a, x)
    #~ by = nP2.forwardProd(b, y)
    
    #~ axpby  = nS1.forwardSum(ax, by)
    #~ axpbpc = nS2.forwardSum(axpby, c)     
    
    #~ s = nSig1.forwardSig(axpbpc)
    
    #~ print s.value
    
    
    #~ print s.gradient

    #~ print nS2.a.value
    #~ print nS2.b.value
    #~ print nS2.a.gradient
    #~ print nS2.b.gradient
    
    #~ print nS2.a.gradient
    #~ print nS2.b.gradient
    
    
    #~ z    = -4
    #~ step = 0.01
    
    #~ xGradient = -4
    #~ yGradient = -4
    #~ zGradient = 3

    #~ n1 = neuronSum(x, y)
    #~ n1.forwardSum()
    #~ print n1.Sum.value
    #~ n1.a.value = 2.
    #~ n1.b.value = 2.
    #~ n1.forwardSum()
    #~ print n1.a.value
    #~ print n1.b.value
    #~ print n1.Sum.value
    
    #~ circuit1 = forwardCircuit()
    #~ print circuit1.circuit(x, y, z)
    
    # Backward pass
    #~ x = x + xGradient * step
    #~ y = y + yGradient * step
    #~ z = z + zGradient * step
    
    #~ print circuit1.circuit(x, y, z)
    
    
# Launch main method if this file is being executed directly
if __name__ == "__main__":
    main()
