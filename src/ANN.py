#!/usr/bin/py
""" ANN is a rudimentary example of artificial neural network
for a comprehensive idea of the general process.
@author Dr. Roland Guichard <r.guichard@ucl.ac.uk>
@todo: Document code
"""

import numpy as np
import scipy as sc

class neuronSum:
    """Basic neuron for the sum
    Arguments:
        a (float)
        b (float)
    Returns:
        The sum of two arguments a and b
    """
    
    def __init__(a, b):
        self.a = a
        self.b = b
    def Sum(self):
        return self.a + self.b

class neuronMul:
    """Basic neuron for the multiplication
    Arguments:
        a (float)
        b (float)
    Returns:
        The product of two arguments a and b
    """
    
    #~ def __init__(self, a, b):
        #~ self.a = a
        #~ self.b = b
    def Prod(self, a, b):
        return a * b

def main():
    """Main method
    Arguments:
    Returns:
    """
    
    # Initial Forward pass
    a    = -2
    b    = 3
    step = 0.01
    
    aGradient = b
    bGradient = a
    
    neuronMul1 = neuronMul()
    output1    = neuronMul1.Prod(a,b)
    
    # Backward pass
    a = a + aGradient * step
    b = b + bGradient * step
    
    output2    = neuronMul1.Prod(a,b)
    print output2
    
# Launch main method if this file is being executed directly
if __name__ == "__main__":
    main()
