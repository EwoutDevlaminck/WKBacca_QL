"""
Auxiliary functions for the BeamWaist.py script.

This script contains the following functions:
- Auxiliary functions for the beam profile analysis. Do a whole range of things, but are usually easy to understand

- Main functions, used for fitting and analysis. These are the functions that are used to perform the analysis of the beam profile.
        --> profile_along_beam: Given a beam trace in both (R, phiN) and (R, Z) coordinates, this function follows the centre of the beam and
        gives the perpendicular profile of the beam at regularly spaced points in R.

        --> beam_width: Given the perpendicular profile of the beam at regularly spaced points in R, this function finds the width of the beam at each point in R,
        using multiple measures

        --> broadening_main: Given a list of locations, this function performs the analysis of the beam profile for each file (location) and saves the results in a file.

- Plotting functions, and postprocessing. These are functions that are used to plot the results of the analysis, and to perform additional checks on the data.

        --> plot_perp_profiles: Given a file containing the results of the beam profile analysis, this function plots the perpendicular profiles
          of the beams at a requested distance along the beam.

        --> check_fits: Given a file containing the results of the beam profile analysis, this function checks the fits of the beam profiles
          at a requested distance along the beam, to see if a gaussian fit is appropriate. (or Cauchy)

        --> broadening_over_distance: Given a file containing the results of the beam profile analysis, this function plots the broadening of the beam,
            as a function of the distance along the beam. Either absolute or relative broadening can be plotted.
            
        --> compare_gaussian_beams: Function to compare the (approximated) gaussian beam characteristics before and after the fluctuation layer.


"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import scipy.signal as signal
from scipy.interpolate import UnivariateSpline
from CommonModules.BiSplineDer import BiSpline
import random

"""-------------------------------------------------------------------------"""
""" Functions for the beam profile analysis, auxiliary functions"""
"""-------------------------------------------------------------------------"""


def gaussian(x, ampl, mean, stdev):
    return ampl * np.exp(-((x-mean) /stdev)**2 / 2)


def Cauchy(x, ampl, mean, gamma):
    return ampl/(np.pi*gamma*(1+((x-mean)/gamma)**2))


def FullWidthHalfMax(x_data, y_data):

    # create a spline of x and blue-np.max(blue)/2 
    spline = UnivariateSpline(x_data, y_data-np.max(y_data)/2, s=0)
    roots = np.array(spline.roots()) # find the roots

    if len(roots) == 2:
        return abs( roots[1] - roots[0])
    else:
        return 0
    

def second_moment_sqrt(x, x0, y):
    # x0 is the center of the beam
    # We calculate the standard deviation through the second moment based on an integration of the beam profile
    # We use the trapezoidal rule for the integration
    dx = np.diff(x)
    dx = np.insert(dx, 0, dx[0])
    return np.sqrt(np.sum((x-x0)**2*y*dx)/np.sum(y*dx))


def kurtosis(x, x0, y):
    # x0 is the center of the beam
    # We calculate the kurtosis based on an integration of the beam profile
    # We use the trapezoidal rule for the integration
    dx = np.diff(x)
    dx = np.insert(dx, 0, dx[0])
    return np.sum(((x-x0)/second_moment_sqrt(x, x0, y))**4*y*dx)/np.sum(y*dx)
# Get the spatial data of the beam


def Rsquared(y, yfit):
    ybar = np.mean(y)
    ssreg = np.sum((y-yfit)**2)
    sstot = np.sum((y-ybar)**2)
    return 1 - ssreg/sstot


def BeamSpline(dataFile):

    BeamTraces = dataFile['BinnedTraces'][()][:,:, 0].T

    try:
        Rmax = dataFile['Rmax'][()]
        Rmin = dataFile['Rmin'][()]
        nmbrR = dataFile['nmbrR'][()]
        R = np.linspace(Rmin, Rmax, nmbrR)
    except:
        Rmax = dataFile['Xmax'][()]
        Rmin = dataFile['Xmin'][()]
        nmbrR = dataFile['nmbrX'][()]
        R = np.linspace(Rmin, Rmax, nmbrR)  

    Zmax = dataFile['Zmax'][()]
    Zmin = dataFile['Zmin'][()]
    nmbrZ = dataFile['nmbrZ'][()]
    Z = np.linspace(Zmin, Zmax, nmbrZ)

    return Rmax, Rmin, Zmax, Zmin, nmbrZ,  BiSpline(R, Z, BeamTraces.T)

# Find the R value after wich the beam is negligible due to absorption

def find_R_absorb(R, angleGrid, threshold=5e-3):
    beamIsPresent = np.where(np.max(angleGrid, axis=0) > threshold, True, False) # 2e-2 is the threshold for the beam to be present
    return beamIsPresent

        
def butterworth(dat, n=1, Wn=1, fs=10):
    b, a = signal.butter(n, Wn, btype='low', fs=fs)
    # Apply the filter to xn.  Use lfilter_zi to choose the initial condition of the filter.
    # Wn is the cutoff frequency
    # fs is the sampling frequency
    
    return signal.filtfilt(b, a, dat)


def find_max_angle(angleGrid, phiN, beamIsPresent):
    """
    Find the maximum angle for each column in the angleGrid.

    Args:
        angleGrid (numpy.ndarray): 2D array representing the angle grid.

    Returns:
        list: A list containing the maximum angle for each column in the angleGrid.
    """
    angleMax = []
    for indR, _ in enumerate(angleGrid[0]):
        if beamIsPresent[indR]:

            SmoothSliceR = butterworth(angleGrid[:, indR])

            angleMax.append(phiN[np.argmax(SmoothSliceR)])

    return angleMax


def find_max_Z(BeamTraces, Z, beamIsPresent):
    """
    Find the maximum Z for each column in the BeamTraces.

    Args:
        BeamTraces (numpy.ndarray): 2D array representing the beam traces.

    Returns:
        list: A list containing the maximum Z for each column in the BeamTraces.
    """
    Zmax = []
    for indR, _ in enumerate(BeamTraces[0]):
        if beamIsPresent[indR]:
            SmoothSliceR = butterworth(BeamTraces[:, indR])
            Zmax.append(Z[np.argmax(SmoothSliceR)])
    
    return Zmax


def find_width_angle(angleGrid, angleMax, phiN, beamIsPresent):
    """
    Find the width angle for each column in the angleGrid.

    Args:
        angleGrid (numpy.ndarray): 2D array representing the angle grid.

    Returns:
        list: A list containing the width angle for each column in the angleGrid.
    """
    angleWidth = []
    i = 0
    for indR, _ in enumerate(angleGrid[0]):
        if beamIsPresent[indR]:
            popt, pcov = curve_fit(gaussian, phiN, angleGrid[:, indR], p0=[1, angleMax[i], 1])
            angleWidth.append(2.355*abs(popt[2]))
            i += 1
    return angleWidth


def perpendicular_array(R, Z, angle, perp_distances):
    Rper = R - np.cos(angle)* perp_distances
    Zper = Z + np.sin(angle)* perp_distances
    return Rper, Zper


def zR(w0, freq=82.7):
    # freq is in GHz, and we want the beam waist in cm
    lamb = 3e8 / (freq*1e9) # in m
    return np.pi * w0**2 / (lamb*1e2)


def beamwaist(z, z0, w0, freq=82.7):
    # freq is in GHz, and we want the beam waist in cm
    zR_val = zR(w0, freq)
    return w0 * np.sqrt(1 + ((z-z0)/zR_val)**2)


"""-------------------------------------------------------------------------"""
""" Main functions, used for fitting and analysis"""
"""-------------------------------------------------------------------------"""

def profile_along_beam(dataFile, angleFile, profilewidth=10, checks=False, convention='ITER'):
    """
    Given a beam trace in both (R, phiN) and (R, Z) coordinates, this function follows the centre of the beam and
    gives the perpendicular profile of the beam at regularly spaced points in R.

    Parameters:
    - dataFile: The file containing the beam trace data.
    - angleFile: The file containing the angular data.
    - profilewidth: The width of the perpendicular profile of the beam.
    - checks: Boolean flag indicating whether to perform additional checks and plot the results.

    Returns:
    - R: The R values after which the beam is negligible due to absorption.
    - beamAngle: The maximum angle for each column in the angleGrid.
    - beamZmax: The maximum Z for each column in the BeamTraces.
    - Per: The line of points perpendicular to the beam.
    - PerpProfiles: The perpendicular profiles of the beam at regularly spaced points in R.
    - angleWidth: The width angle for each column in the angleGrid.
    - phiN: The phiN values.
    - angleGrid: The angle grid.

    """

    ## First, import the angular data
    angleGrid = angleFile['BinnedTraces'][()][:, :, 0].T
    phiNmax = angleFile['phiNmax'][()]
    phiNmin = angleFile['phiNmin'][()]
    if convention == 'ITER': # Everything is written in ASDEX/TCV convention
        phiNmax += np.pi/2
        phiNmin += np.pi/2
    nmbrphiN = angleFile['nmbrphiN'][()]
    phiN = np.linspace(phiNmin, phiNmax, nmbrphiN)
    try:
        RmaxphiN = angleFile['Rmax'][()]
        RminphiN = angleFile['Rmin'][()]
        nmbrRphiN = angleFile['nmbrR'][()]
        Rphi = np.linspace(RminphiN, RmaxphiN, nmbrRphiN)
    except:
        RmaxphiN = angleFile['Xmax'][()]
        RminphiN = angleFile['Xmin'][()]
        nmbrRphiN = angleFile['nmbrX'][()]
        Rphi = np.linspace(RminphiN, RmaxphiN, nmbrRphiN)

    ## Then, import the beam trace

    RM, Rm, ZM, Zm, nmbrZ, BeamSpl = BeamSpline(dataFile) # This is a function that returns the beam trace as a BiSpline object
    # The limits of R in the (R, phiN) and (R,Z) grids can differ, hence this care.
    
    R = np.linspace(Rm, RM, nmbrRphiN)
    Z = np.linspace(Zm, ZM, nmbrZ)
    BeamTrace = np.array([[BeamSpl.eval(R[i], Z[j]) for i in range(len(R))] for j in range(len(Z))])
    BeamTrace = np.where(BeamTrace < 1e-5, 0, BeamTrace)

    ## Find the R value after wich the beam is negligible due to absorption

    beamIsPresent = find_R_absorb(Rphi, angleGrid, threshold=5e-3)

    if checks:
        # Check first that the butterworth filter is not too aggressive
        Smoothtry = butterworth(angleGrid[:, len(Rphi)//2])
        Rawtry = angleGrid[:, len(Rphi)//2]
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.plot(phiN, Smoothtry)
        ax.plot(phiN, Rawtry)
        ax.set_xlabel('Z (cm)')
        ax.set_ylabel('Beam intensity')
        plt.show()

    # Now, find the maximum angle for each column in the angleGrid, meaning the angle at which the beam 'centre' is, for every R
    angleMax = find_max_angle(angleGrid, phiN, beamIsPresent)
    angleWidth = find_width_angle(angleGrid, angleMax, phiN, beamIsPresent)

    if checks:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        cax = ax.contourf(Rphi, phiN, angleGrid, levels=100, cmap=cm.viridis)
        ax.scatter(Rphi[beamIsPresent], angleMax, c='r', s=5)
        ax.set_xlabel('R (cm)')
        ax.set_ylabel('phiN (rad)')
        plt.colorbar(cax)  # Add colorbar
        plt.show()
        plt.show()
    
    # Now, find the maximum Z for each column in the BeamTraces
    
    Zmax = find_max_Z(BeamTrace, Z, beamIsPresent)

    if checks:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.contourf(R, Z, BeamTrace, levels=100, cmap=cm.viridis)
        ax.scatter(R[beamIsPresent], Zmax, c='r', s=5)
        ax.set_aspect('equal')
        ax.set_xlim(min(R[beamIsPresent]), RM)
        #ax.set_ylim(-20, 20)
        ax.set_xlabel('R (cm)')
        ax.set_ylabel('Z (cm)')
        plt.show()

    
    # Now that we have the angles and positions of the beam centre, we can find the perpendicular profile of the beam at regularly spaced points in R.
        
    # First, we need to find the perpendicular profile of the beam at a given R.
    PerpProfiles = []

    # Make a line of points perpendicular to the beam
    # Defined to have a given width
    Per = np.linspace(-profilewidth, profilewidth, 200)
    
    for indR, beamR in enumerate(Rphi[beamIsPresent]):

        beamAngle = angleMax[indR]
        beamZmax = Zmax[indR]

        Rper, Zper = perpendicular_array(beamR, beamZmax, beamAngle, Per)

        # Now, we need to interpolate the beam trace to find the value of the beam at these points
        PerpProfile = [BeamSpl.eval(Rper[i], Zper[i]) for i in range(len(Rper))]
        PerpProfiles.append(PerpProfile)
    
    PerpProfiles = np.array(PerpProfiles)
    
    if checks:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.plot(Per, PerpProfiles[len(R[beamIsPresent])//2, :], label='R = {:.2f}'.format(R[beamIsPresent][len(R[beamIsPresent])//2]))
        ax.legend()
        ax.set_xlabel('Perpendicular distance (cm)')
        ax.set_ylabel('Beam intensity')
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.contourf(R, Z, BeamTrace, levels=100, cmap=cm.viridis)
        ax.scatter(R[beamIsPresent], Zmax, c='r', s=5)
        for i in range(0, len(R[beamIsPresent]), 5):
            Rper, Zper = perpendicular_array(R[beamIsPresent][i], Zmax[i], angleMax[i], Per)
            ax.scatter(Rper, Zper, c='g', s=1)
        ax.set_aspect('equal')
        ax.set_xlim(min(R[beamIsPresent]), RM)
        #ax.set_ylim(-20, 20)
        ax.set_xlabel('R (cm)')
        ax.set_ylabel('Z (cm)')
        plt.show()

    return R[beamIsPresent], beamAngle, beamZmax, Per, PerpProfiles, angleWidth, phiN, angleGrid[:, beamIsPresent]

        

def beam_width(R, beamAngle, Per, PerpProfiles, name, cols, plotfigures=True):
    """
    Given the perpendicular profile of the beam at regularly spaced points in R, this function finds the width of the beam at each point in R,
    using multiple measures.
    """

    # First, take into account that when the beam is refracted, and so to know the true distance
    # traveled by the beam, we need to take into account the angle of the beam.
    dx = abs(np.diff(R)/ np.sin(beamAngle))
    dist_traveled = np.cumsum(dx)
    dist_traveled = np.insert(dist_traveled, 0, 0)[::-1]

    # Now, we need to find the width of the beam at each point in R
    # We will use the FWHM of the beam profile and also a Gaussian fit to the beam profile

    FWHM_exp = []
    FWHM_gauss = []
    FWHM_cauchy = []
    stdev_moment = []
    kurtosis_moment = []
    
    if plotfigures:
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig_ind = 0

    for i in range(len(PerpProfiles)):

        popt, pcov = curve_fit(gaussian, Per, PerpProfiles[i], p0=[1, 0, 1])
        FWHM_gauss.append(2.355*popt[2])

        popt2, pcov2 = curve_fit(Cauchy, Per, PerpProfiles[i], p0=[1, 0, 1])
        FWHM_cauchy.append(2*popt2[2])

        stdev_moment.append(second_moment_sqrt(Per, 0, PerpProfiles[i]))

        kurtosis_moment.append(kurtosis(Per, 0, PerpProfiles[i]))

        FWHM_exp.append(FullWidthHalfMax(Per, PerpProfiles[i]))

        if plotfigures:
            fig_ind += 1
            ax1.plot(Per, PerpProfiles[i], color=cols[i])


            ax2.plot(dist_traveled[i], FWHM_exp[i], 'o', color=cols[i])
            #ax2.plot(dist_traveled[i], FWHM_gauss[i], 'x', color=cols[i])


    if plotfigures:
        ax1.set_xlim(-10, 10)
        ax1.set_xlabel('Perp_distance (cm)')
        ax2.set_xlabel('Distance along beam (cm)')
        ax2.set_ylabel('FWHM (cm)')
        ax2.set_ylim(0)
        plt.tight_layout()
        plt.show()

        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.plot(dist_traveled, FWHM_exp, label='FWHM')
        #ax.plot(dist_traveled, FWHM_gauss, label='FWHM_gauss')
        ax.set_xlabel('Distance along beam (cm)')
        ax.set_ylabel('FWHM (cm)')
        ax.set_ylim(0)
        ax.legend()
        plt.title(name, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    return dist_traveled, FWHM_exp, FWHM_gauss, FWHM_cauchy, stdev_moment, kurtosis_moment



def broadening_main(locations, filename, anglename, plotfigures=True, convention='ITER', datafilename='Broadening.h5'):

    """
    Perform broadening analysis on beam profiles.

    Args:
        locations (list): List of file locations where the beam profile data is stored.
        filename (str): Name of the file containing the beam profile data.
        anglename (str): Name of the file containing the beam angle data.
        plotfigures (bool, optional): Whether to plot the figures. Defaults to True.
        convention (str, optional): Convention used for the analysis. Defaults to 'ITER'.
        datafilename (str, optional): Name of the output data file. Defaults to 'Broadening.h5'.

    Returns:
        h5py.File: The output data file containing the results of the broadening analysis.
    """

    with h5py.File('Broadening.h5', 'w') as hf:

        for i, location in enumerate(locations):

            dataFile = h5py.File(location+'/' +filename, 'r')
            angleFile = h5py.File(location+'/' + anglename, 'r')

            Folder = location.split('/')[-1]
            
            R_used, beamAngle, beamZmax, Per, PerpProfiles, angleWidth, phiN, AngleProfiles = profile_along_beam(dataFile,
                angleFile, profilewidth=10, checks=plotfigures, convention=convention)

            cols = cm.viridis(np.linspace(0, 1, len(R_used)))
            dist, FWHM_exp, FWHM_gauss, FWHM_cauchy, stdev_moment, kurtosis_moment = beam_width(R_used, beamAngle, Per, PerpProfiles, Folder, cols, plotfigures=plotfigures)

            hf.create_group(Folder)
            hf[Folder]['Per'] = np.array(Per)
            hf[Folder]['PerpProfiles'] = np.array(PerpProfiles)
            hf[Folder]['FWHM_angle_gauss'] = np.array(angleWidth)
            hf[Folder]['phiN'] = np.array(phiN)
            hf[Folder]['AngleProfiles'] = np.array(AngleProfiles)
            hf[Folder]['dist_traveled'] = np.array(dist)
            hf[Folder]['FWHM_exp'] = np.array(FWHM_exp)
            hf[Folder]['FWHM_gauss'] = np.array(FWHM_gauss)
            hf[Folder]['FWHM_cauchy'] = np.array(FWHM_cauchy)
            hf[Folder]['stdev_moment'] = np.array(stdev_moment)
            hf[Folder]['kurtosis_moment'] = np.array(kurtosis_moment)


    return h5py.File('Broadening.h5', 'r')



"""-------------------------------------------------------------------------"""
""" Plotting functions, and postprocessing"""
"""-------------------------------------------------------------------------"""

def plot_perp_profiles(datfile, index=20):
    """
    Plot perpendicular and angular beam profiles.

    Parameters:
    - datfile (dict): A dictionary containing data for different beam profiles.
    - index (int): The index of the profile to be plotted.

    Returns:
    - int: Always returns 0.

    """

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for key in datfile.keys():
        filename = key

        distance = datfile[filename]['dist_traveled']
        perp_dimension = np.array(datfile[filename]['Per'])
        perp_profiles = datfile[filename]['PerpProfiles']
        FWHM_gauss = datfile[filename]['FWHM_gauss']

        phiN = np.array(datfile[filename]['phiN'])
        AngleProfiles = datfile[filename]['AngleProfiles']
        FWHM_angle_gauss = datfile[filename]['FWHM_angle_gauss']

    
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(perp_dimension, perp_profiles[index], label=filename, color=color)
        ax.plot(perp_dimension, gaussian(perp_dimension, perp_profiles[index].max(), 0, FWHM_gauss[index]/2.355), linestyle='dashed', color=color, alpha=.5)
        
        ax2.plot(phiN, AngleProfiles[:, index], label=filename, color=color)
        ax2.plot(phiN, gaussian(phiN, AngleProfiles[:, index].max(), phiN[np.argmax(AngleProfiles[:, index])], FWHM_angle_gauss[index]/2.355), linestyle='dashed', color=color, alpha=.5)

    ax.legend()
    ax.set_xlabel('Perpendicular distance (cm)')
    ax.set_ylabel('Beam intensity (A.U.)')
    ax2.set_xlabel('phiN (rad)')
    ax2.set_ylabel('Beam intensity (A.U.)')
    plt.tight_layout()
    ax.set_title('Beam profile after = {:.2f} cm along the beam'.format(distance[index]))
    ax2.set_title('Angular profile after = {:.2f} cm along the beam'.format(distance[index]))
    plt.show()

    return 0



def check_fits(datfile, location_index=1, distance_index=20, fit_range=1):

    
    location = list(datfile.keys())[location_index]
    profile_for_fit = datfile[location]['PerpProfiles'][distance_index][fit_range:-fit_range]
    profile = datfile[location]['PerpProfiles'][distance_index]
    Per = datfile[location]['Per'][:]

    popt, pcov = curve_fit(Cauchy, Per[fit_range:-fit_range], profile_for_fit, p0=[1, 0, 1])
    popt2, pcov2 = curve_fit(gaussian, Per[fit_range:-fit_range], profile_for_fit, p0=[1, 0, 1])


    Rsq_cauch = Rsquared(profile_for_fit, Cauchy(Per[fit_range:-fit_range], *popt))
    Rsq_gauss = Rsquared(profile_for_fit, gaussian(Per[fit_range:-fit_range], *popt2))
    


    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)

    ax.vlines([Per[fit_range], Per[-fit_range]], 0, profile.max(), linestyles='dashed', color='grey')
    plt.plot(Per, profile, label=f'Raw data, FWHM = {FullWidthHalfMax(Per, profile):.2f}', color='grey')

    color = next(ax._get_lines.prop_cycler)['color']

    ax.plot(Per, Cauchy(Per, *popt), label='Cauchy fit R²={:.4f}, FWHM = {:.2f}'.format(Rsq_cauch,  2*popt[2]), color=color)
    ax.hlines(Cauchy(Per, *popt).max()/2, Per[0], Per[-1], linestyles='dashed', color=color, alpha=.5)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(Per, gaussian(Per, *popt2), label='Gaussian fit R²={:.4f}, FWHM = {:.2f}'.format(Rsq_gauss,  2.355*popt2[2]), color=color)
    ax.hlines(gaussian(Per, *popt2).max()/2, Per[0], Per[-1], linestyles='dashed', color=color, alpha=.5)
    ax.legend(loc='upper right')
    ax.set_xlabel('Perpendicular distance (cm)')
    ax.set_ylabel('Beam intensity (A.U.)')
    #ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.00)
    plt.tight_layout()
    plt.title('Beam profile for {} after {:.2f} cm along the beam'.format(location, datfile[location]['dist_traveled'][distance_index]))
    plt.show()
    return 0



def broadening_over_distance(datfile, plotindex=1, relative=False, absorption_position=0, sorting=[]):
    """
    Calculate and plot the broadening over distance for a given datfile.

    Parameters:
    - datfile (dict): A dictionary containing the data file.
    - plotindex (int): The index of the plot to be displayed. Default is 1.
    - relative (bool): Whether to calculate the relative broadening. Default is False.

    Returns:
    - distances (list): A list of distances.
    - stdevs (list): A list of standard deviations.

    """

    distances = []
    stdevs = []
    nofluct_saved = False

    if plotindex !=0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)

    
    for key in datfile.keys():
        
        distance = datfile[key]['dist_traveled'][:]
    
        FWHM_exp = datfile[key]['FWHM_exp'][:]
        FWHM_gauss = datfile[key]['FWHM_gauss'][:]

        stdev_moments = datfile[key]['stdev_moment'][:]
        kurtosis_moments = datfile[key]['kurtosis_moment'][:]

        distances.append(distance)
        stdevs.append(stdev_moments)

        phiN = datfile[key]['phiN'][:]
        AngleProfiles = datfile[key]['AngleProfiles'][:]
        FWHM_angle = np.array([FullWidthHalfMax(phiN, AngleProfiles[:, i]) for i in range(AngleProfiles.shape[1])])
   
        FWHM_angle_gauss = datfile[key]['FWHM_angle_gauss'][:]

        if not nofluct_saved:
            nofluct_saved = True
            FWHM_exp_nofluct = FWHM_exp # For the second figure I always want the relative broadening
            FWHM_gauss_nofluct = FWHM_gauss # For the second figure I always want the relative broadening
            if relative:
                FWHM_angle_nofluct = FWHM_angle
                FWHM_angle_gauss_nofluct = FWHM_angle_gauss
                stdev_moments_nofluct = stdev_moments

                ylimits = [2, 2]
        
            else:
                FWHM_angle_nofluct = 1
                FWHM_angle_gauss_nofluct = 1
                stdev_moments_nofluct = 1

                ylimits = [.3, 3]


        if plotindex != 0:
            
            color = next(ax._get_lines.prop_cycler)['color']

            ax.set_xlabel('Distance [cm]')
            ax.set_xlim(0, distance[0])
            ax.set_ylim(0, 5)
            ax.scatter(distance, FWHM_exp , color=color, s=5, marker='x')
            ax.plot(distance, FWHM_gauss, label=key, color=color)
            ax.set_ylabel('FWHM [cm]')
            ax.set_title('Empirical and Gaussian FWHM (spatial)')
            
            if absorption_position != 0:
                index_absoprtion = np.argmin(abs(distance-absorption_position))
                ax2.vlines(distance[index_absoprtion], 0, 2, linestyles='dashed', color='grey')

                if len(FWHM_exp) < len(FWHM_exp_nofluct):
                    # Add nan points, as they are not present in the fluctuation case
                    FWHM_exp = np.append(FWHM_exp, [np.nan]*(len(FWHM_exp_nofluct)-len(FWHM_exp)))
                    FWHM_gauss = np.append(FWHM_gauss, [np.nan]*(len(FWHM_gauss_nofluct)-len(FWHM_gauss)))
                    FWHM_angle = np.append(FWHM_angle, [np.nan]*(len(FWHM_angle_nofluct)-len(FWHM_angle)))
                    FWHM_angle_gauss = np.append(FWHM_angle_gauss, [np.nan]*(len(FWHM_angle_gauss_nofluct)-len(FWHM_angle_gauss)))
                    stdev_moments = np.append(stdev_moments, [np.nan]*(len(stdev_moments_nofluct)-len(stdev_moments)))
                    kurtosis_moments = np.append(kurtosis_moments, [np.nan]*(len(stdev_moments_nofluct)-len(stdev_moments)))

                    distance = np.append(distance, np.linspace(distance[-1], distance[-1]+distance[1]-distance[0], len(FWHM_exp_nofluct)-len(FWHM_exp)))

                elif len(FWHM_exp) > len(FWHM_exp_nofluct):
                    FWHM_exp = FWHM_exp[:len(FWHM_exp_nofluct)]
                    FWHM_gauss = FWHM_gauss[:len(FWHM_gauss_nofluct)]
                    FWHM_angle = FWHM_angle[:len(FWHM_angle_nofluct)]
                    FWHM_angle_gauss = FWHM_angle_gauss[:len(FWHM_angle_gauss_nofluct)]
                    stdev_moments = stdev_moments[:len(stdev_moments_nofluct)]            
                    kurtosis_moments = kurtosis_moments[:len(stdev_moments_nofluct)]        

                    distance = distance[:len(FWHM_exp_nofluct)]                               
                                           

                FWHM_exp_rel = FWHM_exp/FWHM_exp_nofluct
                FWHM_gauss_rel = FWHM_gauss/FWHM_gauss_nofluct
                experi = 'exp' # Weird construct to get the subscript in the label
                gaussi = 'gauss'
                label = rf'$b_{{{experi}}}$ {FWHM_exp_rel[index_absoprtion]:.2f}, $b_{{{gaussi}}}$ {FWHM_gauss_rel[index_absoprtion]:.2f}'
            else:
                label = key

            ax2.scatter(distance, FWHM_exp/FWHM_exp_nofluct , color=color,  s=5, marker='x')
            ax2.plot(distance, FWHM_gauss/FWHM_gauss_nofluct, label=label, color=color)
            ax2.set_title('Empirical and Gaussian FWHM (spatial)')
            ax2.set_xlim(0, distance[0])
            ax2.set_ylim(0, 2)
            ax2.set_ylabel('Relative broadening')
            ax2.set_xlabel('Distance [cm]')

            if absorption_position != 0:
                ax2.legend(loc='lower left')
            

            ax3.set_xlabel('Distance [cm]')
            ax3.set_xlim(0, distance[0])
            ax3.set_ylim(0, ylimits[0])
            ax3.scatter(distance, FWHM_angle/FWHM_angle_nofluct, color=color,  s=5, marker='x')
            ax3.plot(distance, FWHM_angle_gauss/FWHM_angle_gauss_nofluct, label=key, color=color)
            ax3.set_title('Empirical and Gaussian FWHM (angular)')


            if absorption_position != 0:
                ax4.vlines(distance[index_absoprtion], 0, 2, linestyles='dashed', color='grey')

                stdev_moments_rel = stdev_moments/stdev_moments_nofluct
                label = f'Rel. broadening {stdev_moments_rel[index_absoprtion]:.2f}'
            else:
                label = key

            ax4.plot(distance, stdev_moments/stdev_moments_nofluct, label=label, color=color)
            ax4.set_title('Square root of second central moment')
            ax4.set_xlabel('Distance [cm]')
            ax4.set_xlim(0, distance[0])
            ax4.set_ylim(0, ylimits[1])

            if absorption_position != 0:
                ax4.legend(loc='lower left')



            ax5.plot(distance, kurtosis_moments - 3, label=key, color=color)
            ax5.set_xlim(0, distance[0])
            ax5.set_title('Relative kurtosis')
            ax5.set_xlabel('Distance [cm]')
            ax5.set_ylabel('Kurtosis - 3')

            if relative:
                ax3.set_ylabel('Relative broadening')
                ax4.set_ylabel('Relative broadening')
            else:
                ax3.set_ylabel('FWHM (rad)')
                ax4.set_ylabel('Standard deviation [cm]')

            ax.legend()
            plt.tight_layout()
    plt.show()

    return distances, stdevs
        

def compare_gaussian_beams(datfile, x_startfluct, x_endfluct, buffer=10, equalaspect=True):
    """
    Function to compare the (approximated) gaussian beam characteristics before and after the fluctuation layer.
    If we assume the beam stays gaussian in its perpendicular profile and still follows gaussian beam theory 
    (thus ignoring the diffraction purely by changing plasma density), we can define the beam after cattering through 
    an alternated gaussian beam, with different waist size and position.

    Input:
    - datfile (dict): A read out hdf5 file containing the data.
    - x_startfluct (float): The distance at which the fluctuation layer starts.
    - x_endfluct (float): The distance at which the fluctuation layer ends.
    - buffer (int): The number of points to be cut off at the start of the beam profile for proper fitting.
    - equalaspect (bool): Whether to plot the figures with equal aspect ratio.

    Returns:
    - w0_before (float): The beam waist before the fluctuation layer.
    - z0_before (float): The beam position before the fluctuation layer.
    - w0_after (float): The beam waist after the fluctuation layer.
    - z0_after (float): The beam position after the fluctuation layer.

    """

    if equalaspect:
        fig = plt.figure(figsize=(16,10))
    else:
        fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    counter = 0
    for key in datfile.keys():
        # Fit the beam waist, both before and after the fluctuation
        # We use the FWHM of the gaussian fit, and convert it to the beam waist using the factor sqrt(2*ln(2))
        # Be mindful though, that the gaussian fit is not always the best option, so cross check with the actual profile.
        width_along_beam_full = datfile[key]['FWHM_gauss'][:]/np.sqrt(2*np.log(2))
        # Cut off a few buffer points at the start (as we cant define the perpendicular profile properly there)
        # and after the widest point.
        width_along_beam = width_along_beam_full[np.argmax(width_along_beam_full):-buffer]
        dist = datfile[key]['dist_traveled'][np.argmax(width_along_beam_full):-buffer]

        startfluct = np.argmin(abs(dist-x_startfluct))
        endfluct = np.argmin(abs(dist-x_endfluct))

        popt, pcov = curve_fit(beamwaist, dist[startfluct:], width_along_beam[startfluct:], p0=[100, 3])
        z0_before, w0_before = popt

        popt, pcov = curve_fit(beamwaist, dist[:endfluct], width_along_beam[:endfluct], p0=[100, 3])
        z0_after, w0_after = popt

        z = np.linspace(0, max(dist), 1000)
        w_before = beamwaist(z, z0_before, w0_before)
        w_after = beamwaist(z, z0_after, w0_after)


        if counter==0:
            ax.plot(z, w_before, label=f'w0={w0_before:.2f}, z0={z0_before:.2f}', c='gray')
            ax.plot(z, -w_before, c='gray')
            ax.vlines(z0_before, -w0_before, w0_before, linestyles='dashed', colors='gray')
            ax.scatter(dist[startfluct:], width_along_beam[startfluct:], s=20, marker='x', alpha=0.5, c='gray')

        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(z, w_after, label=f'w0={w0_after:.2f}, z0={z0_after:.2f}', color=color)
        ax.plot(z, -w_after, color=color)
        ax.vlines(z0_after, -w0_after, w0_after, linestyles='dashed', colors=color)
        ax.scatter(dist[:endfluct], width_along_beam[:endfluct], s=20, marker='x', alpha=0.5, c=color)
        ax.scatter(dist[endfluct:startfluct], width_along_beam[endfluct:startfluct], c='black', s=5)
        counter += 1
    ax.fill_between(dist[endfluct:startfluct], -4, 4, alpha=0.2)
    ax.set_xlabel('Distance along beam (cm)')
    ax.set_ylabel('Beam cross-section(cm)')
    ax.set_ylim(-4, 4)
    ax.set_xlim(0, max(dist))
    ax.legend()
    #ax.set_xlim(10, 28)
    #ax.set_ylim(2, 3)
    if equalaspect:
        ax.set_aspect('equal')
    plt.show()

    return w0_before, z0_before, w0_after, z0_after

"""-------------------------------------------------------------------------"""
""" Example of use"""
"""-------------------------------------------------------------------------"""

if __name__ == "__main__":

    locations = ['/home/devlamin/Documents/WKBeam_related/WKBEAM_ED/Benchmark_JC_Analytical/Output_nofluct_S1.1',
                '/home/devlamin/Documents/WKBeam_related/WKBEAM_ED/Benchmark_JC_Analytical/Output_fluct_S1.1/s0.4_Lf1_Delt0.2',
                '/home/devlamin/Documents/WKBeam_related/WKBEAM_ED/Benchmark_JC_Analytical/Output_fluct_S1.1/s0.4_Lf5_Delt0.2',
                '/home/devlamin/Documents/WKBeam_related/WKBEAM_ED/Benchmark_JC_Analytical/Output_fluct_S1.1/s0.4_Lf2_Delt0.2',
                ]

    filename = 'L2_binned_XZ.hdf5'
    anglename = 'L2_binned_angular.hdf5'
    convention = 'TCV'

    datfile = broadening_main(locations,
            filename, anglename, plotfigures=False, convention=convention, datafilename='Broadening_example.h5')

    plot_perp_profiles(datfile, index=20)

    check_fits(datfile, location_index=1, distance_index=20, fit_range=1)

    broadening_over_distance(datfile, plotindex=1, relative=False)
    broadening_over_distance(datfile, plotindex=1, relative=True)

    compare_gaussian_beams(datfile, 5, 12, buffer=10, equalaspect=True)