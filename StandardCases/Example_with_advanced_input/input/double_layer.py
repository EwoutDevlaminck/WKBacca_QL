"""
This demostrates the possibility to include more complicated
models for the fluctuations envelop.
"""

import numpy as np

amplitude = 1.e7  # The density is very low in this test case.
                  # This large value makes fluctuation visible.

center1 = 1.3     # position in rho of the first turbulence layer
width1 = 0.05      # width in Delta rho of the first layer
center2 = 1.1     # position of the second layer
width2 = 0.1      # width of the second layer

                 
def scatteringDeltaneOverne(ne, rho, theta):

    """
    A double Gaussian layer of fluctuations.
    """

    layer1 = np.exp(-(rho - center1)**2 / width1**2)
    layer2 = np.exp(-(rho - center2)**2 / width2**2)

    return amplitude * (layer1 + layer2)
