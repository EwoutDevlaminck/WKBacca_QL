""" Script to prepare input data for use in WKBeam"""

""" Requires a .mat file that is made preferentially with the make_equil_tcv_ed.m 
function on the lac servers, or at least one that has the same outputs.


Inputs:
        - filename: string, name of the .mat file to be read
        - output_names: list of strings, names of the output files to be saved
        - plot_option: 0, 1 or 2, whether to plot the no data, all data or just the final extended profiles
        
Outputs:
        - output_files: ne.dat: psi, Ne(psi) values for an extended psi grid
                        Te.dat: psi, Te(psi) values for an extended psi grid
                        topfile: # of gridpoints in R,Z, psi_sep value
                                 R,Z values,
                                 Br, Bz, Bphi values in 2D arrays
                                 psi values in 2D array 
        - plots of the extended ne, Te profiles (if plot_option = 1)
        - plots of the psi, Br, Bz, Bphi profiles in 2D (if plot_option = 1)
        - plots of psi, Br, Bz, Bphi, ne, Te in 2D (if plot_option = 2 or 1)
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import scipy.interpolate as spl
import tkinter as tk
from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Disabled for now, spcsrv26 doesn't allow this.

from BiSplineDer import BiSpline, UniBiSpline
# This is needed for the extrapolation to a 2D grid of these profiles.
# Normally only needed for plotting, but requires the BiSplineDer.py file to be in 
# the same directory as this script

working_dir = os.getcwd()
os.chdir(working_dir)
EPFLGreen = '#007480'


"""----------------------------------"""
""""------Function definitions-------"""
"""----------------------------------"""

# Function to get a 2D numpy array of values taken from a spline
# Normally spline only allows you to take singular points at a time.

def IntSample(R, z, IntObjFunct):
    jmax = np.size(R)
    imax = np.size(z)
    s = np.empty([imax, jmax])
    s[:,:] = np.nan
    for i in range(imax):
        zloc = z[i]
        for j in range(jmax):
            Rloc = R[j]
            s[i,j] = IntObjFunct(Rloc, zloc)
    return s

# Butterworth filter function to smooth out the extended ne and Te profiles

def butterworth(dat, n=1, Wn=0.3, fs=2):
    b, a = signal.butter(n, Wn, btype='low', fs=fs)
    
    # n is the order of the filter, best is 1 for our case
    # Wn is the cutoff frequency, best is 0.3 for our case
    # fs is the sampling frequency, best is 2 for our case
    
    return signal.filtfilt(b, a, dat)


def profile_extrapolation_ed(rho, profile, rho_max, rho_vacuum, refinement=100):

    # First extend the ranges of rho and profile to include
    # the region outside the last closed flux surface
    drho = rho[1]-rho[0] #assumes rho is a regular grid

    # We impose a true vacuum profile at two further rho steps away from the last data point. 

    rho_ext = np.concatenate((rho, np.arange(rho[-1] + 2*drho, rho_max, drho)))

    # From this vacuum position, we set the profile to zero before interpolating
    profile_ext = np.concatenate((profile, np.zeros(len(rho_ext)-len(profile))))

    # Now we 'interpolate' the profile to the outside of the LCFS using a spline

    profile_spline = spl.UnivariateSpline(rho_ext, profile_ext, s=0)

    # In order to properly resolve the separatrix, we need to refine our grid. 
    # Then we can smoothen the transition.
    # A value of 100 for the refined grid is more or less a lower bound, 
    # otherwise the transition is too sharp.
    # If you do change this value, it is perhaps 
    # also best to play with the butterworth filter parameters

    rho_ext_refined = np.linspace(rho[0], rho_max, refinement)
    profile_ext_refined = profile_spline(rho_ext_refined)

    # Now we smoothen the transition between the original profile and the extrapolated profile
    # The butterworth filter is a low pass filter, so it will smoothen
    # The parameters are a bit arbitrary, but they seem to work well
    # Just the n=1 is a must
    profile_ext_refined = butterworth(profile_ext_refined, n=1, Wn=0.3, fs=2)

    #Now just a safety to make sure we don't have negative values.
    # Also values that are too small are set to zero
    # to assure the demanded vacuum profile is imposed

    indices_to_zero = np.where(profile_ext_refined < 1e-3* profile_ext_refined.max())[0][0]
    profile_ext_refined[indices_to_zero:] = 0.

    # Now we set the profile to zero as well just before the maximally allowed value we impose
    # This is to make sure the antenna is in vacuum, as required by WKBeam.
    # Additionally, it makes sure no density is present outside of an x-point.
    profile_ext_refined = np.where(rho_ext_refined < rho_vacuum, profile_ext_refined, 0.)

    return np.array([rho_ext_refined, profile_ext_refined])


def replace_nan_with_avg(arr, axis=0):
    # Find the indices where NaN values are present
    nan_indices = np.isnan(arr)

    # Replace NaN values with the average of the adjacent non-NaN values
    if axis == 0:  # along columns
        for i in range(arr.shape[1]):
            for j in range(arr.shape[0]):
                if nan_indices[j, i]:
                    left_val = arr[j-1, i] if j > 0 else np.nan
                    right_val = arr[j+1, i] if j < arr.shape[0]-1 else np.nan
                    arr[j, i] = np.nanmean([left_val, right_val])

    elif axis == 1:  # along rows
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if nan_indices[i, j]:
                    left_val = arr[i, j-1] if j > 0 else np.nan
                    right_val = arr[i, j+1] if j < arr.shape[1]-1 else np.nan
                    arr[i, j] = np.nanmean([left_val, right_val])

    return arr


def inside_contour(R, Z, field, edge_value):
    #The function produces a mask of the same side as the field, with True values where the field is inside the edge_value
    #The edge_value is the value of the contour that is considered the edge of the plasma, outside of which the density is forcefully zero

    mask = np.zeros_like(field, dtype=bool)
    contour_path = plt.contour(R, Z, field, levels=[edge_value], colors='white').collections[0].get_paths()[0]

    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            mask[i,j] = contour_path.contains_point((R[i,j], Z[i,j]))
    
    return mask


def interactive_boundary_setting(rz_R, rz_Z, rz_psi, psi_antenna):
    psi_forced_edge = 1.
    print('Set the max psi value for which we will allow a non-zero density.')
    print(f'The antenna is at psi {psi_antenna:.2f} and should not be exceeded.')
    
    def update_plot():
        nonlocal psi_forced_edge
        psi_forced_edge_try = float(entry.get())
        if psi_forced_edge_try == -1:
            print('Last value will be kept.')
            root.quit()
        else:
            ax.clear()
            psi_forced_edge = psi_forced_edge_try
            mask_psi = inside_contour(rz_R, rz_Z, rz_psi, psi_forced_edge_try)
            ax.contourf(rz_R, rz_Z, mask_psi, cmap='OrRd', alpha=0.5)
            ax.contour(rz_R, rz_Z, rz_psi, levels=20, colors='black')
            ax.set_title(f"Current value: {psi_forced_edge_try}")
            canvas.draw()

    root = tk.Tk()
    root.title("Interactive Boundary Setting")
    
    fig = Figure(figsize=(5, 7), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(f"Current value: {psi_forced_edge}")
    mask_psi = inside_contour(rz_R, rz_Z, rz_psi, psi_forced_edge)
    ax.contourf(rz_R, rz_Z, mask_psi, cmap='OrRd', alpha=0.5)
    ax.contour(rz_R, rz_Z, rz_psi, levels=20, colors='black')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    label = tk.Label(root, text=f"Enter a new forced edge value (-1 if satisfied):")
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    
    button = tk.Button(root, text="Try", command=update_plot)
    button.pack()

    tk.mainloop()

    return psi_forced_edge

"""----------------------------------"""

# Main function to extract and prepare all the data

def TCV_input_prep_ed(filename, output_names= ['ne.dat', 'Te.dat', 'topfile'], plot_option=0, correct_psi_option=1, antenna_loc=[88, 40]):

    # Load the .mat file
    # The data is stored in a dictionary, so we need to extract it
    # Peculiar way of storing due to matlab
    data = loadmat(filename)['equil'][0, 0] 

    # Unpack the data. There are two big groups, those expressed in terms
    # of psi and those expressed in terms of R and Z

    # Every variable on the psi theta grid gets a pt prefix, 
    #each variable on the R Z grid gets an rz prefix

    ### TOPFILE PREPARATION ###

    #These are the coordinates of the magnetix axis
    Rp = np.array(data['Rp'][0][0])
    Zp = np.array(data['Zp'][0][0])

    R_coarse = np.array(data['x'] + Rp) # Not a lot of values
    Z_coarse = np.array(data['y'] + Zp) # We refine for WKBeam application

    # Create a fine grid of R and Z
    nR, nZ = 100, 100
    R = np.linspace(R_coarse.min(), R_coarse.max(), nR).reshape(-1, 1)
    Z = np.linspace(Z_coarse.min(), Z_coarse.max(), nZ).reshape(-1, 1)


    # R, Z grid of R, Z, psi and the magnetic field components
    rz_R = np.tile(R, (1, nZ))
    rz_Z = np.tile(Z, (1, nR)).T

    rz_psi_coarse = np.array(data['xypsin'])
    rz_rho_coarse = np.sqrt(rz_psi_coarse)

    rz_B_R_coarse = np.array(data['Bx'])
    rz_B_Z_coarse = np.array(data['By'])
    rz_B_phi_coarse = -np.array(data['BPHI']) # Negative sign added here for the proper convention! What was stored in LUKE uses the opposite sign


    # We replace NaN values with the average of the adjacent non-NaN values
    # Can be done for all the fields
    rz_psi_coarse = replace_nan_with_avg(rz_psi_coarse, axis=0)
    rz_rho_coarse = replace_nan_with_avg(rz_rho_coarse, axis=0)
    rz_B_R_coarse = replace_nan_with_avg(rz_B_R_coarse, axis=0)
    rz_B_Z_coarse = replace_nan_with_avg(rz_B_Z_coarse, axis=0)
    rz_B_phi_coarse = replace_nan_with_avg(rz_B_phi_coarse, axis=0)
 
    #All these fields are still on the R, Z grid that is too coarse, so
    # we need to interpolate them to a finer grid

    rz_psi_spline = BiSpline(R_coarse, Z_coarse, rz_psi_coarse)
    rz_rho_coarse_spline = BiSpline(R_coarse, Z_coarse, rz_rho_coarse)
    rz_B_R_spline = BiSpline(R_coarse, Z_coarse, rz_B_R_coarse)
    rz_B_Z_spline = BiSpline(R_coarse, Z_coarse, rz_B_Z_coarse)
    rz_B_phi_spline = BiSpline(R_coarse, Z_coarse, rz_B_phi_coarse)

    rz_psi = IntSample(R, Z, rz_psi_spline.eval).T
    rz_rho = IntSample(R, Z, rz_rho_coarse_spline.eval).T
    rz_B_R = IntSample(R, Z, rz_B_R_spline.eval).T
    rz_B_Z = IntSample(R, Z, rz_B_Z_spline.eval).T
    rz_B_phi = IntSample(R, Z, rz_B_phi_spline.eval).T

    psi_antenna = rz_psi_spline.eval(antenna_loc[0], antenna_loc[1]) #psi value at antenna location. We need this to know where ne has to be zero for sure


    if correct_psi_option == 1:

        # Check iteratively where a good boundary for the plasma is, i.e. where we have not yet encountered an x point
        # Otherwise we could have articafs of density outside the plasma
        #psi_forced_edge = interactive_boundary_setting(rz_R, rz_Z, rz_psi, psi_antenna)

        # Bypass for spcsrv26, where the interactive plotting is not working at the moment
        psi_forced_edge = 1.1
        print(f'Forced edge value set to {psi_forced_edge:.2f}')

        # Now force the psi field to a constant value outside this edge, as then we can be sure the density is zero there
        # Psi_forced_edge is also used later on when extending the 1D ne and Te profiles smootly to the outside of the LCFS

        mask_psi = inside_contour(rz_R, rz_Z, rz_psi, psi_forced_edge)
        rz_psi = np.where(mask_psi, rz_psi, psi_forced_edge)
    else:
        psi_forced_edge = psi_antenna #If we don't want to correct the psi, we just set the edge to the antenna location

    topfile = open(working_dir + '/' + output_names[2], 'ab')

    # Clear the content of the file when not already empty
    topfile.seek(0); topfile.truncate()

    # Export the data
    np.savetxt(topfile, ['nrm, \t nz'], fmt='%s')
    np.savetxt(topfile, np.array([[nR, nZ, 1]], dtype=int), fmt='%i')

    # In most of the topfiles used in the WKBeam examples, there's data on the Rin and Rout, where the separatrix is. 
    # This is however, never used, so I skip it here.

    np.savetxt(topfile, np.array(['R']), fmt='%s')
    np.savetxt(topfile, R)
    np.savetxt(topfile, np.array(['Z']), fmt='%s')
    np.savetxt(topfile, Z.T)

    np.savetxt(topfile, np.array(['Br']), fmt='%s')
    np.savetxt(topfile, rz_B_R.T)
    np.savetxt(topfile, np.array(['Bt']), fmt='%s')
    np.savetxt(topfile, rz_B_phi.T)
    np.savetxt(topfile, np.array(['Bz']), fmt='%s')
    np.savetxt(topfile, rz_B_Z.T)
    np.savetxt(topfile, np.array(['psi']), fmt='%s')
    np.savetxt(topfile, rz_psi.T)

    # Close file
    topfile.close()



    ### NE, TE PREPARATION ###

    rz_rho_max = np.max(rz_rho)
    try:
        rho = np.array(data['rho_eq'][0])
    except:
        rho = np.array(data['rho_psiP'][0])
    psi = rho**2
    np_LCFS = len(rho) # Number of points inside the LCFS

    rho_full = np.array(data['rho_full'][0]) #Use for ne, Te profiles with scrapeoff layer
    p_Te = data['Te_full'][0]
    p_ne = data['ne_full'][0] * 1e-19 #Convert from m^-3 to 1e19 m^-3
    p_Te_err = data['Te_err_full'][0]
    p_ne_err = data['ne_err_full'][0] * 1e-19 #Convert from m^-3 to 1e19 m^-3

    #Next lines if analytical model and we don't have data of scrapeoff layer
    """
    rho_full = np.array(data['rho_psiP'][0]) #Use for ne, Te profiles with scrapeoff layer
    p_Te = data['pTe'][0]
    p_ne = data['pne'][0] * 1e-19 #Convert from m^-3 to 1e19 m^-3
    p_Te_err = np.zeros(len(p_Te))
    p_ne_err = np.zeros(len(p_ne))
    """

    data_Te = profile_extrapolation_ed(rho_full, p_Te, rho_max=rz_rho_max, rho_vacuum=np.sqrt(psi_forced_edge))
    data_ne = profile_extrapolation_ed(rho_full, p_ne, rho_max=rz_rho_max, rho_vacuum=np.sqrt(psi_forced_edge))
    [_, ne_ext_nocorr] = profile_extrapolation_ed(rho_full, p_ne, rho_max=rz_rho_max, rho_vacuum=2.)
    rho_ext = data_Te[0]
    Te_ext = data_Te[1]
    ne_ext = data_ne[1]

    # Now also save them in a file in the correct way.

    Ne_file = open(working_dir + '/' + output_names[0], 'ab')
    Te_file = open(working_dir + '/' + output_names[1], 'ab')

    # Clear the content of both files when not already empty
    Ne_file.seek(0); Ne_file.truncate()
    Te_file.seek(0); Te_file.truncate()

    # Export the data
    np.savetxt(Ne_file, np.array([data_ne[0].size], dtype=int), fmt='%i')
    np.savetxt(Ne_file, data_ne.T)
    #
    np.savetxt(Te_file, np.array([data_Te[0].size], dtype=int), fmt='%i')
    np.savetxt(Te_file, data_Te.T)

    # Close files
    Ne_file.close()
    Te_file.close()
    

    ### PLOT OPTION ###
    # If you want all the produced data to be plotted, we need some extra things.

    if plot_option != 0:
        # Comment out the next line if you don't have the raw data from outside the LCFS, like for analytical models
        #The raw data from outside the LCFS
        rho_scr = np.array(data['rho_scr'][0]) 
        Te_scr = np.array(data['Te_scr'][0])
        ne_scr = np.array(data['ne_scr'][0]) * 1e-19 #Convert from m^-3 to 1e19 m^-3
        Te_err_scr = np.array(data['Te_err_scr'][0])
        ne_err_scr = np.array(data['ne_err_scr'][0]) * 1e-19 #Convert from m^-3 to 1e19 m^-3

        if plot_option == 1:
            
            ne_ind_zero = np.where(ne_ext == 0)[0][0]
 
            # Extended profiles
            plt.figure(figsize=(10,5))
            plot_ne = plt.subplot(1,2,1)
            plot_ne.errorbar(rho_full[:np_LCFS]**2, p_ne[:np_LCFS], yerr=p_ne_err[:np_LCFS], label='Original Thomson', fmt='o', color=EPFLGreen, zorder=1)
            # Comment out the next line if you don't have the raw data from outside the LCFS, like for analytical models
            plot_ne.errorbar(rho_scr**2, ne_scr, yerr=ne_err_scr, label='Raw Thomson outside LCFS', fmt='x', color='grey', zorder=2)
            plot_ne.plot(rho_ext**2, ne_ext, label='Extended ne', color='r', zorder=3)
            if abs(ne_ext - ne_ext_nocorr).max() > 1e-3:
                plot_ne.plot(rho_ext**2, ne_ext_nocorr, label='Extended ne, no correction', linestyle='--', color='orange', zorder=3)
            plot_ne.set_xlabel(r'$\psi_{norm}$')
            plot_ne.set_ylabel(r'$n_e [10^{19} m^{-3}]$')
            plot_ne.set_xlim(0)
            plot_ne.set_ylim(0)
            # We also specify how for the profiles now reach!
            plot_ne.annotate(rf'$n_e = 0$ at $\psi$ ={rho_ext[ne_ind_zero]**2:.2f}',xy=(0.05, 0.15), xycoords='axes fraction', fontsize=12)
            plot_ne.annotate(rf'$\psi$ at antenna: {psi_antenna:.2f}', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12)
            plot_ne.legend()

            plot_Te = plt.subplot(1,2,2)
            plot_Te.plot(rho_ext**2, Te_ext, label='Extended Te', color='r', zorder=3)
            plot_Te.errorbar(rho_full[:np_LCFS]**2, p_Te[:np_LCFS], yerr=p_Te_err[:np_LCFS], label='Original Thomson', fmt='o', color=EPFLGreen, zorder=1)
            # Comment out the next line if you don't have the raw data from outside the LCFS, like for analytical models
            plot_Te.errorbar(rho_scr**2, Te_scr, yerr=Te_err_scr, label='Raw Thomson outside LCFS', fmt='x', color='grey', zorder=2)
            plot_Te.set_xlabel(r'$\psi_{norm}$')
            plot_Te.set_ylabel(r'$T_e [keV]$')
            plot_Te.set_xlim(0)
            plot_Te.set_ylim(0)
            plot_Te.legend()

            plt.suptitle('Extended profiles', fontsize=16)
            plt.savefig('Extended_profiles.pdf', dpi=300)

            
            # 2D plots of data for LUKE

            # For LUKE, a psi, theta grid is used. We quickly import this

            theta = np.array(data['theta'][0])
            

            npsi = len(psi)
            ntheta = len(theta)

            pt_psi = np.tile(psi, (ntheta, 1)).T
            pt_rho = np.tile(rho, (ntheta, 1)).T

            pt_R = np.array(data['ptx'] + Rp)
            pt_Z = np.array(data['pty'] + Zp)

            pt_B_R = np.array(data['ptBx'])
            pt_B_Z = np.array(data['ptBy'])
            pt_B_phi = np.array(data['ptBPHI'])

            plt.figure(figsize=(6,10))
            plt.plot(rho, pt_B_phi[:, 0])
            plt.show()
            # Now we can plot the data

            plt.figure(figsize=(9,10))
            plt.suptitle(r'LUKE input data, on $\psi, \theta$ grid', fontsize=16)

            plot_psi = plt.subplot(2,2,1)
            plot_psi.set_aspect('equal')
            psi_grid =plot_psi.scatter(pt_R, pt_Z, c=pt_psi, s=2)
            plot_psi.contour(pt_R, pt_Z, pt_psi, [0.999], colors='r')
            plt.colorbar(psi_grid)
            plot_psi.set_xlabel('R [m]')
            plot_psi.set_ylabel('Z [m]')
            plot_psi.set_title(r'$\psi\ [A.U.]$')

            plot_Bt = plt.subplot(2,2,2)
            plot_Bt.set_aspect('equal')
            Bt_grid = plot_Bt.scatter(pt_R, pt_Z, c=pt_B_phi, s=2)
            plot_Bt.contour(pt_R, pt_Z, pt_psi, [0.999], colors='r')
            plt.colorbar(Bt_grid)
            plot_Bt.set_xlabel('R [m]')
            plot_Bt.set_ylabel('Z [m]')
            plot_Bt.set_title(r'$B_t\ [T]$')

            plot_BR = plt.subplot(2,2,3)
            plot_BR.set_aspect('equal')
            Br_grid = plot_BR.scatter(pt_R, pt_Z, c=pt_B_R, s=2)
            plot_BR.contour(pt_R, pt_Z, pt_psi, [0.999], colors='r')
            plt.colorbar(Br_grid)
            plot_BR.set_xlabel('R [m]')
            plot_BR.set_ylabel('Z [m]')
            plot_BR.set_title(r'$B_R\ [T]$')

            plot_BZ = plt.subplot(2,2,4)
            plot_BZ.set_aspect('equal')
            Bz_grid = plot_BZ.scatter(pt_R, pt_Z, c=pt_B_Z, s=2)
            plot_BZ.contour(pt_R, pt_Z, pt_psi, [0.999], colors='r')
            plt.colorbar(Bz_grid)
            plot_BZ.set_xlabel('R [m]')
            plot_BZ.set_ylabel('Z [m]')
            plot_BZ.set_title(r'$B_Z\ [T]$')

            plt.savefig('LUKE_input_data.pdf', dpi=300)


            # 2D plots of data for WKBeam

            plt.figure(figsize=(6,10))
            plt.suptitle(r'WKBeam input data, on $R, Z$ grid', fontsize=16)

            plot_psi = plt.subplot(2,2,1)
            plot_psi.set_aspect('equal')
            psi_grid =plot_psi.scatter(rz_R, rz_Z, c=rz_psi, s=2)
            plot_psi.contour(rz_R, rz_Z, rz_psi, [1], colors='r')
            plt.colorbar(psi_grid)
            plot_psi.set_xlabel('R [m]')
            plot_psi.set_ylabel('Z [m]')
            plot_psi.set_title(r'$\psi\ [A.U.]$')

            plot_Bt = plt.subplot(2,2,2)
            plot_Bt.set_aspect('equal')
            Bt_grid = plot_Bt.scatter(rz_R, rz_Z, c=rz_B_phi, s=2)
            plot_Bt.contour(rz_R, rz_Z, rz_psi, [1], colors='r')
            plt.colorbar(Bt_grid)
            plot_Bt.set_xlabel('R [m]')
            plot_Bt.set_ylabel('Z [m]')
            plot_Bt.set_title(r'$B_t\ [T]$')

            plot_BR = plt.subplot(2,2,3)
            plot_BR.set_aspect('equal')
            Br_grid = plot_BR.scatter(rz_R, rz_Z, c=rz_B_R, s=2)
            plot_BR.contour(rz_R, rz_Z, rz_psi, [1], colors='r')
            plt.colorbar(Br_grid)
            plot_BR.set_xlabel('R [m]')
            plot_BR.set_ylabel('Z [m]')
            plot_BR.set_title(r'$B_R\ [T]$')

            plot_BZ = plt.subplot(2,2,4)
            plot_BZ.set_aspect('equal')
            Bz_grid = plot_BZ.scatter(rz_R, rz_Z, c=rz_B_Z, s=2)
            plot_BZ.contour(rz_R, rz_Z, rz_psi, [1], colors='r')
            plt.colorbar(Bz_grid)
            plot_BZ.set_xlabel('R [m]')
            plot_BZ.set_ylabel('Z [m]')
            plot_BZ.set_title(r'$B_Z\ [T]$')

            plt.savefig('WKBeam_input_data.pdf', dpi=300)

        # Plot with extended profiles

        # We have to make a 2D grid of the extended profiles
        # This is done with the UniBiSpline and IntSample function

        #First we make a spline of the psi values, as Te and
        # ne are expressed in terms of psi

        rz_psi_spline = BiSpline(R, Z, rz_psi)

        rz_ne_spline = UniBiSpline(rho_ext, ne_ext, psi_interpolator=rz_psi_spline)
        rz_ne = IntSample(R, Z, rz_ne_spline.eval).T
        # We set all values that are too small to zero for correct interpolation
        rz_ne = np.where(rz_ne < 1e-4* p_ne.max(), 0, rz_ne)

        rz_Te_spline = UniBiSpline(rho_ext, Te_ext, psi_interpolator=rz_psi_spline)
        rz_Te = IntSample(R, Z, rz_Te_spline.eval).T
        rz_Te = np.where(rz_Te < 1e-4* p_Te.max(), 0, rz_Te)
    


        plt.figure(figsize=(10,10))
        plt.suptitle(r'WKBeam full input fields', fontsize=16)

        plot_psi = plt.subplot(2,3,1)
        plot_psi.set_aspect('equal')
        psi_grid =plot_psi.contourf(rz_R, rz_Z, rz_psi, levels=100, cmap='OrRd')
        plot_psi.contour(rz_R, rz_Z, rz_psi, [1], colors='black')
        plot_psi.contour(rz_R, rz_Z, rz_psi, np.linspace(0, rz_rho_max**2, 20), colors='grey', linestyles='dashed', linewidths=1)
        plt.colorbar(psi_grid)
        plot_psi.set_xlabel('R [m]')
        plot_psi.set_ylabel('Z [m]')
        plot_psi.set_title(r'$\psi\ [A.U.]$')

        plot_Bt = plt.subplot(2,3,2)
        plot_Bt.set_aspect('equal')
        Bt_grid = plot_Bt.contourf(rz_R, rz_Z, rz_B_phi, levels=100, cmap='OrRd')
        plot_Bt.contour(rz_R, rz_Z, rz_psi, [1], colors='black')
        plot_Bt.contour(rz_R, rz_Z, rz_psi, np.linspace(0, rz_rho_max**2, 20), colors='grey', linestyles='dashed', linewidths=1)
        plt.colorbar(Bt_grid)
        plot_Bt.set_xlabel('R [m]')
        plot_Bt.set_ylabel('Z [m]')
        plot_Bt.set_title(r'$B_t\ [T]$')

        plot_BR = plt.subplot(2,3,3)
        plot_BR.set_aspect('equal')
        Br_grid = plot_BR.contourf(rz_R, rz_Z, rz_B_R, levels=100, cmap='OrRd')
        plot_BR.contour(rz_R, rz_Z, rz_psi, [1], colors='black')
        plot_BR.contour(rz_R, rz_Z, rz_psi, np.linspace(0, rz_rho_max**2, 20), colors='grey', linestyles='dashed', linewidths=1)
        plt.colorbar(Br_grid)
        plot_BR.set_xlabel('R [m]')
        plot_BR.set_ylabel('Z [m]')
        plot_BR.set_title(r'$B_R\ [T]$')

        plot_BZ = plt.subplot(2,3,4)
        plot_BZ.set_aspect('equal')
        Bz_grid = plot_BZ.contourf(rz_R, rz_Z, rz_B_Z, levels=100, cmap='OrRd')
        plot_BZ.contour(rz_R, rz_Z, rz_psi, [1], colors='black')
        plot_BZ.contour(rz_R, rz_Z, rz_psi, np.linspace(0, rz_rho_max**2, 20), colors='grey', linestyles='dashed', linewidths=1)
        plt.colorbar(Bz_grid)
        plot_BZ.set_xlabel('R [m]')
        plot_BZ.set_ylabel('Z [m]')
        plot_BZ.set_title(r'$B_Z\ [T]$')

        plot_ne = plt.subplot(2,3,5)
        plot_ne.set_aspect('equal')
        ne_grid = plot_ne.contourf(rz_R, rz_Z, rz_ne, levels=100, cmap='OrRd')
        plot_ne.contour(rz_R, rz_Z, rz_psi, [1], colors='black')
        plot_ne.contour(rz_R, rz_Z, rz_psi, np.linspace(0, rz_rho_max**2, 20), colors='grey', linestyles='dashed', linewidths=1)
        plt.colorbar(ne_grid)
        plot_ne.set_xlabel('R [m]')
        plot_ne.set_ylabel('Z [m]')
        plot_ne.set_title(r'$n_e\ [m^{-3}]$')

        plot_Te = plt.subplot(2,3,6)
        plot_Te.set_aspect('equal')
        Te_grid = plot_Te.contourf(rz_R, rz_Z, rz_Te, levels=100, cmap='OrRd')
        plot_Te.contour(rz_R, rz_Z, rz_psi, [1], colors='black')
        plot_Te.contour(rz_R, rz_Z, rz_psi, np.linspace(0, rz_rho_max**2, 20), colors='grey', linestyles='dashed', linewidths=1)
        plt.colorbar(Te_grid)
        plot_Te.set_xlabel('R [m]')
        plot_Te.set_ylabel('Z [m]')
        plot_Te.set_title(r'$T_e\ [eV]$')

        plt.savefig('WKBeam_input_fields.pdf', dpi=300)
        plt.show()

        return


"""----------------------------------"""
"""-----------MAIN-PROGRAM-----------"""
"""----------------------------------"""

if __name__ == '__main__':
    filename = r"/home/devlamin/WKBacca_QL/TCV_preprocess/74301/For_WKBeam_with_scrapeoff/EQUIL_TCV_74301_1.2000.mat"
    antenna_location = [122.90, -0.31] #In cm, position of the antenna in R, Z
    # For L1/L4 usually around 122, 0. For L2/L5 around 122, 50.5.
    # This is basically to check if the density goes to zero at the antenna location
    TCV_input_prep_ed(filename, plot_option=1, correct_psi_option=1, antenna_loc=antenna_location)