#!/usr/bin/py
""" ANN is a rudimentary example of artificial neural network
for a comprehensive idea of the general process.
@author Dr. Roland Guichard <r.guichard@ucl.ac.uk>
@todo: Document code
"""

import numpy as np
import scipy as sc

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
    def __init__(self, a, b):
        self.a   = axon(a, 0.)
        self.b   = axon(b, 0.)
    def forwardSum(self):
        self.Sum = axon(self.a.value + self.b.value, 0.) 
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
    def __init__(self, a, b):
        self.a   = axon(a, 0.)
        self.b   = axon(b, 0.)
    def forwardProd(self):
        self.Sum = axon(self.a.value * self.b.value, 0.)    
    def backwardProp(self):
        """Chain rule the local gradient with the output gradient
        """
        self.a.gradient = self.a.gradient + self.b.value * self.Sum.gradient
        self.b.gradient = self.b.gradient + self.a.value * self.Sum.gradient

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
    def __init__(self, a):
        self.a = axon(a, 0.)
    def forwardSig(self):
        self.Sig = axon(sigmoid(self.a.value), 0.)
    def backwardProp(self):
        """Chain rule the local gradient with the output gradient
        """
        s = self.Sig.value
        self.a.gradient = self.a.gradient + (s * (1 - s)) * self.Sig.gradient
    
class forwardCircuit:
    """Basic circuit composition of a sum and product neuron
    """
    def __init__(self):
        self.neuronSum1 = neuronSum()
        self.neuronMul1 = neuronMul()
    def circuit(self, a, b, c):
        return self.neuronMul1.Prod(self.neuronSum1.Sum(a, b), c)

def main():
    """Main method
    Arguments:
    Returns:
    """
    
    # Initial Forward pass
    a    = axon(1., 0.)
    b    = axon(2., 0.)
    c    = axon(-3., 0.)

    x    = axon(-1., 0.)
    y    = axon(3., 0.)
    
    nS1 = 
    
    
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
