import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import h5py
from scipy.io import loadmat

def plotVessel(idata):

    vesselfile  = idata.vesselfile
    print('vesselfile: ', vesselfile)
    vesseldata  = loadmat(vesselfile)['vesselcont']

    Rv_in       = vesseldata['Rv_in'][0][0][0]
    Rv_out      = vesseldata['Rv_out'][0][0][0]
    Zv_in       = vesseldata['Zv_in'][0][0][0]
    Zv_out      = vesseldata['Zv_out'][0][0][0]
    Zt          = vesseldata['Zt'][0][0][:, 0]
    Rt          = vesseldata['Rt'][0][0][:, 0]

    # Convert to cm
    Rv_in = Rv_in * 100
    Rv_out = Rv_out * 100
    Zv_in = Zv_in * 100
    Zv_out = Zv_out * 100
    Zt = Zt * 100
    Rt = Rt * 100

    return Rv_in, Rv_out, Zv_in, Zv_out, Zt, Rt
