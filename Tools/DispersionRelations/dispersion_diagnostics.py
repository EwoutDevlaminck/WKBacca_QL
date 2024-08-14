"""
Diagnostics tool to check the dispersion relation on a single ray
from data produced by WKBeam in diagnostic mode. 

WKBeam trace o ray according to the launched-mode dispersion. Along such rays 
quantities relative to both modes (Ordinary and Extra-ordinary) are computed.
The quantity relative to the lanched mode should always have physical values. 
Not so for the other mode which might become nevanescent even if the launched 
mode is propagating. 

In order to activate the diagnostic mode in WKBeam, one jast have to launch a 
single ray (therefore using only two core, one master and one ray-tracing core)
with the input flag dispersionSurfacesOnCentralRay = 'file_name' in the 
ray tracing input file. This flag should point to the file name of an hdf5
dataset which will be created in the output directory.
"""

# Import statements
import h5py
import numpy as np
#import Tools.DispersionRelations.Dispersion_and_Hamiltonians as DH
import Dispersion_and_Hamiltonians as DH
from matplotlib import pyplot as plt


# Check the dispersion relation on a ray for WKBeam diagnostics
def check_dispersion(diagnostics_file):
    
    """
    Running the WKBeam code in ray diagnostic mode, i.e.,
    running with the central ray only, will aoutput a number of
    additional information on the central ray, which are stored
    in a user defined files for diagnostics.

    Given the full path to that output file, this function check
    if the dispersion relation is properly satisfied. Three form
    of the dispersion realtion are used: (1) Bornatici et al.
    Nuclear Fusion 23 (1983) 1153, equations (3.1.55a) and (3.1.55b);
    (2) Maj Ph.D. Thesis (2003), equation (C.18); (3) the WKBeam 
    Hamiltonian (checking conservation properties of the ray integration
    and effects of round-off near the singularities.

    USAGE:

      >>> from testH_vs_BornaticiNF import check_dispersion
      >>> check_dispersion(diagnostics_file)

    Here the argument "diagnostics_file" is a string with the path to the 
    WKBeam output file as described above.
    """

    # Load the datasets
    diag = h5py.File(diagnostics_file, 'r')

    # Extract relevant parameters along the ray
    k0 = diag.get('k0')[()]
    rayXYZ = diag.get('rayXYZ')[()]
    StixX = diag.get('StixParamX')[()]
    StixY = diag.get('StixParamY')[()]
    sigma = diag.get('Mode')[0]
    Nparal = diag.get('Nparallel')[()]
    Nperp = diag.get('Nperp')[()] # the launched mode
    Nperp_OM = diag.get('NperpOtherMode')[()] # the other mode
    
    try:
        Lperp = diag.get('Lperp')[()]
        sigma_perp = diag.get('sigma_perp')[()] # = 1. / (k0 Lperp)
        Lparallel = diag.get('Lparallel')[()]
        IScatDiag = diag.get('IntensityOfScatteringEvents')[()]
        IScatOffDiag = diag.get('IntensityOfScatteringEventsOffDiagonal')[()]
        MaxScat = max(np.max(IScatDiag), np.max(IScatOffDiag))
    except:
        pass
        
    diag.close()

    # Extract the X-coordinate of the reference ray
    rayX = rayXYZ[0,:]  

    # Squared refractive indices
    Nparal2 = Nparal**2
    Nperp2 = Nperp**2
    Nperp2_OM = Nperp_OM**2
    N2 = Nparal2 + Nperp2

    # The other mode might be evanescent where the launched mode is 
    # prapagating. In those cases Nperp is NaN and those values are masked
    Nperp2_OM = np.ma.masked_invalid(Nperp2_OM)
    N2_OM = Nparal2 + Nperp2_OM
    
    # Evaluate the cosine of the angle theta in the interval [0, pi/2]
    # (The array Nperp2 and Nparal2 are set to zero by default in WKBeam
    #  where the ray has not been traced, e.g., where there is no more energy,
    #  hence the small number 1.e-10 is added to the denominator to avoid
    #  division by zero.)
    cos_theta = np.sqrt(Nparal2 / (N2 + 1.e-10))

    # Evaluate the disperation relations for the launched mode
    N2Bornatici = DH.N2_Bornatici(StixX, StixY, sigma, cos_theta)
    N2Maj = DH.N2_Maj(StixX, StixY, sigma, cos_theta)
    HWKBeam = DH.H_WKBeam(StixX, StixY, sigma, Nperp2, Nparal2)

    # Evaluate the dispersion relation for the other mode (-sigma)
    N2Bornatici_OM = DH.N2_Bornatici(StixX, StixY, -sigma, cos_theta)
    N2Maj_OM = DH.N2_Maj(StixX, StixY, -sigma, cos_theta)
    HWKBeam_OM = DH.H_WKBeam(StixX, StixY, -sigma, Nperp2_OM, Nparal2)
    
    # Mask when evamescent
    N2Bornatici_OM = np.ma.masked_invalid(N2Bornatici_OM)
    N2Maj_OM = np.ma.masked_invalid(N2Maj_OM)
    HWKBeam_OM = np.ma.masked_invalid(HWKBeam_OM)

    # Relative errors on N^2 for the lanched mode
    maxN2Bornatici =  np.max(np.abs(N2Bornatici)) 
    maxN2Maj =  np.max(np.abs(N2Maj))
    RelError_Bornatici = (N2 - N2Bornatici) / maxN2Bornatici
    RelError_Maj = (N2 - N2Maj) / maxN2Maj

    # Relative errors on N^2 for the other mode
    maxN2Bornatici_OM = np.max(np.abs(N2Bornatici_OM)) 
    maxN2Maj_OM = np.max(np.abs(N2Maj_OM))
    RelError_Bornatici_OM =  (N2_OM - N2Bornatici_OM) / maxN2Bornatici_OM
    RelError_Maj_OM = (N2_OM - N2Maj_OM) / maxN2Maj_OM
    
    # Absolute error on the Hamiltonian for the launched and the other mode
    AbsError_H = np.abs(HWKBeam)
    AbsError_H_OM = np.abs(HWKBeam_OM)

    # Remove the part of the array that hold unphysical values
    totalR2 = rayXYZ[0,:]**2 + rayXYZ[1,:]**2 + rayXYZ[2,:]**2
    rayX = np.ma.masked_where(totalR2==0., rayX)
    RelError_Bornatici = np.ma.masked_where(totalR2==0., RelError_Bornatici)
    RelError_Maj = np.ma.masked_where(totalR2==0., RelError_Maj)
    AbsError_H = np.ma.masked_where(totalR2==0., AbsError_H)
    #
    RelError_Bornatici_OM = np.ma.masked_where(totalR2==0., RelError_Bornatici_OM)
    RelError_Maj_OM = np.ma.masked_where(totalR2==0., RelError_Maj_OM)
    AbsError_H_OM = np.ma.masked_where(totalR2==0., AbsError_H_OM)

    # Max. errors
    maxRelError_Bornatici = np.max(np.abs(RelError_Bornatici))
    maxRelError_Maj = np.max(np.abs(RelError_Maj))
    maxAbsError_H = np.max(np.abs(AbsError_H))
    #
    maxRelError_Bornatici_OM = np.max(np.abs(RelError_Bornatici_OM))
    maxRelError_Maj_OM = np.max(np.abs(RelError_Maj_OM))
    maxAbsError_H_OM = np.max(np.abs(AbsError_H_OM))
    
    # Some in-line output
    print('\n')
    print('PARAMETERS:')
    print('k0 = {}'.format(k0))
    try:
        # Exponential envelopeof the total cross-section assuming Gaussian
        # correlation of the fluctuations
        env = np.exp(-0.5 * (Nperp - Nperp_OM)**2 / sigma_perp**2)
        env *= MaxScat
        # Boundaries of likelihood (3 sigma rule)
        Nperp_upper = Nperp + 3. * sigma_perp
        Nperp_lower = Nperp - 3. * sigma_perp
        print('Lparallel = {}'.format(Lparallel))
        print('Lperp = {}'.format(Lperp))
        print('sigma_perp = {}'.format(sigma_perp))
    except:
        pass

    print('\n')
    print('FOR THE LAUNCHED MODE:')
    print('Mode = {}'.format(DH.modes[sigma]))
    print('Mode index sigma = {}'.format(sigma))
    print('Relative errors on N^2:')
    print(' comparing to Barnatici et al.: {}'.format(maxRelError_Bornatici))
    print(' comparing to Maj: {}'.format(maxRelError_Maj))
    print('\n')
    print('Absolute error on the Hamiltonian: {}'.format(maxAbsError_H))
    print('\n')
    print('FOR THE OTHER MODE:')
    print('Mode = {}'.format(DH.modes[-sigma]))
    print('Mode index sigma = {}'.format(-sigma))
    print('Relative errors on N^2:')
    print(' comparing to Barnatici et al.: {}'.format(maxRelError_Bornatici_OM))
    print(' comparing to Maj: {}'.format(maxRelError_Maj_OM))
    print('\n')
    print('Absolute error on the Hamiltonian: {}'.format(maxAbsError_H_OM))
    print('\n')

    # Plot traces for the LAUNCHED mode
    plt.figure(figsize=(10,8))
    plt.figtext(0.55, 0.95,
                'Launched mode ({})'.format(DH.modes[sigma]), fontsize=20)
    plt.subplots_adjust(top=0.85, hspace=0.7, wspace=0.4)
    ax1 = plt.subplot(221)
    ax1.plot(rayXYZ[0,:], RelError_Bornatici, 
             label=r'$N^2$ rel. error vs Bornatici')
    ax1.plot(rayXYZ[0,:], RelError_Maj, 
             label=r'$N^2$ rel. error vs Maj')
    ax1.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)
    ax1.grid('on')
    ax2 = plt.subplot(222)
    ax2.plot(rayXYZ[0,:], AbsError_H)
    ax2.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax2.set_title('Absolute error on $H$', fontsize=15)    
    ax2.grid('on')
    ax3 = plt.subplot(223)
    ax3.plot(rayX, N2, 'b', label=r'$N^2$')
    ax3.plot(rayX, Nperp2, 'r', label=r'$N_\perp^2$')
    ax3.plot(rayX, Nparal2, 'g', label=r'$N_\parallel^2$')
    ax3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    ax3.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax3.grid('on')
    ax4 = plt.subplot(224)
    ax4.plot(rayX, StixX, 'b', label=r'Stix $X$')
    ax4.plot(rayX, StixY, 'r', label=r'Stix $Y$')
    ax4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    ax4.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax4.grid('on')

    # Plot traces for the OTHER mode
    plt.figure(figsize=(10,8))
    plt.figtext(0.55, 0.95,
                'Other mode ({})'.format(DH.modes[-sigma]), fontsize=20)
    plt.subplots_adjust(top=0.85, hspace=0.7, wspace=0.4)
    ax5 = plt.subplot(221)
    ax5.plot(rayXYZ[0,:], RelError_Bornatici_OM, 
             label=r'$N^2$ rel. error vs Bornatici')
    ax5.plot(rayXYZ[0,:], RelError_Maj_OM, 
             label=r'$N^2$ rel. error vs Maj')
    ax5.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax5.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)
    ax5.grid('on')
    ax6 = plt.subplot(222)
    ax6.plot(rayXYZ[0,:], AbsError_H_OM)
    ax6.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax6.set_title('Absolute error on $H$', fontsize=15)    
    ax6.grid('on')
    ax7 = plt.subplot(223)
    ax7.plot(rayX, N2_OM, 'b', label=r'$N^2$')
    ax7.plot(rayX, Nperp2_OM, 'r', label=r'$N_\perp^2$')
    ax7.plot(rayX, Nparal2, 'g', label=r'$N_\parallel^2$')
    ax7.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    ax7.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax7.grid('on')
    ax8 = plt.subplot(224)
    ax8.plot(rayX, StixX, 'b', label=r'Stix $X$')
    ax8.plot(rayX, StixY, 'r', label=r'Stix $Y$')
    ax8.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    ax8.set_xlabel(r'$x$ [cm]', fontsize=15)
    ax8.grid('on')

    # Compare the dispersion relation of the two modes
    plt.figure(figsize=(10,8))
    ax9 = plt.subplot(111)
    ax9.plot(rayX, Nperp, 'b', linewidth=2.0, label=r'$N_\perp$ launched mode')
    ax9.plot(rayX, Nperp_OM, 'r', linewidth=2.0, label=r'$N_\perp$ other mode')
    ax9.set_xlabel(r'$x$ [cm]', fontsize=25)
    ax9.set_ylabel(r'$N_\perp$', fontsize=25)
    ax9.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    ax9bis = ax9.twinx()
    try:
        ax9.plot(rayX, Nperp_upper, 'b:', linewidth=2.0)
        ax9.plot(rayX, Nperp_lower, 'b:', linewidth=2.0)

        ax9bis.plot(rayX, IScatDiag, 'k', linewidth=2.0, 
                    label='Scatt. diagonal')
        ax9bis.plot(rayX, IScatOffDiag, 'g', linewidth=2.0,
                    label='Scatt. off diagonal')
    except:
        pass
    ax9bis.legend(loc='upper left')
    ax9bis.set_ylabel(r'a.u.', fontsize=25)
    ax9.grid('on')

    # Envelope 
    plt.figure(figsize=(10,8))
    ax10 = plt.subplot(111)
    try:
        ax10.plot(rayX, IScatDiag, 'k', linewidth=2.0, 
                  label='Scatt. diagonal')
        ax10.plot(rayX, IScatOffDiag, 'g', linewidth=2.0,
                  label='Scatt. off diagonal')
        ax10.plot(rayX, env, 'b', linewidth=2.0, 
                  label='$\propto \exp[- \Delta N_\perp^2 / (2\sigma_\perp^2) ]$')

    except:
        pass
    ax10.set_xlabel(r'$x$ [cm]', fontsize=25)
    ax10.grid('on')
    ax10.legend(loc='upper left')

    plt.show()

    # Return maximum errors
    return maxRelError_Bornatici, maxRelError_Maj, maxAbsError_H

# Testing
if __name__=='__main__':
    
    import sys

    try:
        check_dispersion(sys.argv[1])
    except IndexError:
        print("""
        Usage: From base directory
        
        $ python Tools/DispersionRelations/dispersion_diagnostics.py <diag_file>
        
        where <diag_file> is the file name (with path) of the diagnostic file
        produced by WKBeam when running with one single ray and with the flag
        dispersionSurfacesOnCentralRay = '<diag_file>'.
        """)
        
# End of file
