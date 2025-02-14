"""THIS FILE CAN BE USED TO CALL THE CYTHON BINNING CODE DIRECTLY FROM TERMINAL.
ALL RELEVANT PARAMETERS CAN BE CHOSEN BELOW. THE DATA IS LOADED AND SAVED FROM
AND TO THE FILES CHOSEN AMONG THE PARAMETERS.
"""

# Load standard modules 
import sys
import math
import h5py
import numpy as np
# Load local modules
from Binning.modules.binning import binning
from Binning.modules.binning_nonuni import binning as binning_nonuni
from Binning.modules.compute_normalisationfactor import compute_norm_factor



############################################################################
# BELOW: A FUNCTION WHICH DOES THE BINNING IS PROVIDED. 
# IT IS CALLED WHEN THIS FILE IS EXECUTED AND ALSO CAN BE CALLED
# FROM OTHER PIECES OF PYTHON CODE. AS PARAMETER, IT TAKES AN idata
# INSTANCE WITH ALL RELEVANT PARAMETERS.
############################################################################
def binning_pyinterface(idata):

    # MODIFY INPUT PARAMETERS IF NEEDED
    ############################################################################
   
    # see if outputfilename is defined. 
    # if not, just attach _binned to inputfilename
    try:
        outputfilename = idata.outputfilename
    except:
        outputfilename = idata.inputfilename + '_binned'

   
    if idata.storeVelocityField == True:
        VelocityComponentsToStore = idata.VelocityComponentsToStore
    else:
        VelocityComponentsToStore = []


    # ALLOCATE MEMORY WHERE THE RESULTS CAN BE WRITTEN AND CALL THE 
    # CYTHON BINNING FUNCTION
    ############################################################################

    # see how many directions are needed in the input file.
    if len(idata.WhatToResolve) > 4:
        print('THE MAXIMUM NUMBER OF DIMENSIONS 4 IS EXCEEDED.\n')
        raise
    try: 
        uniform_bins = idata.uniform_bins
    except:
        uniform_bins = True

    if uniform_bins == True:
    # put one bin in directions not in use
    # and the boundaries around zero (because the data_sim-values will be 0 for those dimensions)
        nmbr = np.empty([4], dtype=int)
        min = np.empty([4])
        max = np.empty([4])
        for i in range(0,4):
            if i < len(idata.nmbr):
                nmbr[i] = idata.nmbr[i]
                min[i] = idata.min[i]
                max[i] = idata.max[i]
                if min[i] > max[i]:
                    print('ERROR: lower boundary larger than the upper one for %s\n' %(idata.WhatToResolve[i]))
                    raise
            else:
                nmbr[i] = 1
                min[i] = -1.
                max[i] = +1.
    else:
        bins = np.empty([4], dtype=np.ndarray)
        nmbr = np.empty([4], dtype=int)
        for i in range(0,4):
            if i < len(idata.bins):
                bins[i] = idata.bins[i]
                nmbr[i] = len(bins[i])-1
            else:
                bins[i] = np.linspace(-1.,1.,2)
                nmbr[i] = 1

    # if only the total amplitude needs to be computed do this directly
    if idata.storeWfct == True:
        WfctUnscattered = np.zeros(np.append(nmbr,2))
    if idata.storeVelocityField == True:
        VelocityFieldUnscattered = np.zeros(np.append(nmbr,[len(VelocityComponentsToStore),2]))

    if idata.storeAbsorption == True:
        AbsorptionUnscattered = np.zeros(np.append(nmbr,2))

    # if scattered rays are also needed explicitly
    if idata.computeAmplitude == True or idata.computeScatteringEffect == True:
        if idata.storeWfct == True:
            WfctScattered = np.zeros(np.append(nmbr,2))
        if idata.storeVelocityField == True:
            VelocityFieldScattered = np.zeros(np.append(nmbr,[len(VelocityComponentsToStore),2]))
        if idata.storeAbsorption == True:
            AbsorptionScattered = np.zeros(np.append(nmbr,2))

   

    nmbrRaysScattered = 0
    nmbrRaysUnscattered = 0
       

    # READ BEAM PARAMETERS FROM THE INPUT FILE
    ############################################################################
    for i in range(0,idata.nmbrFiles):
        # choose the right filename
        filename = idata.inputdirectory + idata.inputfilename + '_file%i.hdf5' %(i)
    
        print("loading file %s ...\n" %(filename))
        sys.stdout.flush() 

        # open file
        fid = h5py.File(filename,'r')
        if idata.correctionfactor == True:
            data_sim_CorrectionFactor = fid.get('TracesCorrectionFactor')[()]
        else:
            data_sim_CorrectionFactor = np.empty([0,0])

        data_sim_XYZ = fid.get('TracesXYZ')[()]
        data_sim_Wfct = fid.get('TracesWfct')[()]
        data_sim_nmbrscattevents = fid.get('TracesNumberScattEvents')[()]

        nmbrRaysToUse = data_sim_XYZ.shape[0]
        nmbrPointsPerRay = data_sim_XYZ.shape[2]

        # see what data is needed:
        if len(set(idata.WhatToResolve).intersection(set(['Nx','Ny','Nz']))) \
                + len(set(VelocityComponentsToStore).intersection(set(['Nx','Ny','Nz']))) > 0:
            data_sim_NxNyNz = fid.get('TracesNxNyNz')[()]

        if len(set(idata.WhatToResolve).intersection(set(['Vx','Vy','Vz']))) \
                + len(set(VelocityComponentsToStore).intersection(set(['Vx','Vy','Vz']))) > 0:
            data_sim_VxVyVz = fid.get('TracesGroupVelocity')[()]
           
        if 'Psi' in idata.WhatToResolve or 'rho' in idata.WhatToResolve:
            data_sim_Psi = fid.get('TracesPsi')[()]
        
        if 'Theta' in idata.WhatToResolve:
            data_sim_Theta = fid.get('TracesTheta')[()]

        if 'Nparallel' in idata.WhatToResolve:
            data_sim_Nparallel = fid.get('TracesNparallel')[()]

        if 'phiN' in idata.WhatToResolve:
            data_sim_phiN = fid.get('TracesphiN')[()]

        if 'Nperp' in idata.WhatToResolve:
            data_sim_Nperp = fid.get('TracesNperpendicular')[()]
    
        # see if time is given.
        # if not, reconstruct it using the timestep
        try:
            data_sim_time = fid.get("TracesTime")[()]
            print('time along the rays was found and is used.\n')
        except:
            timestep = fid.get("timestep")[()]
            data_sim_time = np.empty([nmbrRaysToUse,nmbrPointsPerRay])
            for k in range(0,nmbrRaysToUse):
                data_sim_time[k,:] = np.linspace(0.,(nmbrPointsPerRay-1)*timestep,nmbrPointsPerRay)

        if i == 0:
            sigma = fid.get("Mode")[()]
            freq = fid.get("FreqGHz")[()]
            antennapolangle = fid.get("antennapolangle")[()]
            antennatorangle = fid.get("antennatorangle")[()]
            rayStartX = fid.get("rayStartX")[()]
            rayStartY = fid.get("rayStartY")[()]
            rayStartZ = fid.get("rayStartZ")[()]
            beamwidth1 = fid.get("beamwidth1")[()]
            beamwidth2 = fid.get("beamwidth2")[()]
            curvatureradius1 = fid.get("curvatureradius1")[()]
            curvatureradius2 = fid.get("curvatureradius2")[()]
            centraleta1 = fid.get("centraleta1")[()]
            centraleta2 = fid.get("centraleta2")[()]
            
            # see if the normalisation factor for energy flow is needed or not
            try:
                InputPower = idata.InputPower
                NormFactor = InputPower / compute_norm_factor(freq,
                                                              beamwidth1, beamwidth2,
                                                              curvatureradius1, curvatureradius2,
                                                              centraleta1, centraleta2)
                print("Input power is %.3fMW\n" %(InputPower))
            except:
                NormFactor = 1.
                print("Normalisation such that central electric field on antenna is 1.\n")
        else:
            if sigma != fid.get("Mode")[()] or \
                    freq != fid.get("FreqGHz")[()] or \
                    antennapolangle != fid.get("antennapolangle")[()] or \
                    antennatorangle != fid.get("antennatorangle")[()] or \
                    rayStartX != fid.get("rayStartX")[()] or \
                    rayStartY != fid.get("rayStartY")[()] or \
                    rayStartZ != fid.get("rayStartZ")[()] or \
                    beamwidth1 != fid.get("beamwidth1")[()] or \
                    beamwidth2 != fid.get("beamwidth2")[()] or \
                    curvatureradius1 != fid.get("curvatureradius1")[()] or \
                    curvatureradius2 != fid.get("curvatureradius2")[()] or \
                    centraleta1 != fid.get("centraleta1")[()] or \
                    centraleta2 != fid.get("centraleta2")[()]:
                   
                print("ATTENTION: THERE IS A PROBLEM HERE, NOT ALL INPUT FILES AGREE IN ALL PARAMETERS.\n")
                sys.stdout.flush()
                raise

        
        # close file
        fid.close()

        # and compose data for the binning routine
        data_sim = np.zeros([nmbrRaysToUse,4,nmbrPointsPerRay])
        
        for i in range(0,len(idata.WhatToResolve)):
            if idata.WhatToResolve[i] == 'X':
                data_sim[:,i,:] = data_sim_XYZ[:,0,:].copy()
            elif idata.WhatToResolve[i] == 'Y':
                data_sim[:,i,:] = data_sim_XYZ[:,1,:].copy()
            elif idata.WhatToResolve[i] == 'Z':
                data_sim[:,i,:] = data_sim_XYZ[:,2,:].copy()
            elif idata.WhatToResolve[i] == 'Nx':
                data_sim[:,i,:] = data_sim_NxNyNz[:,0,:].copy()
            elif idata.WhatToResolve[i] == 'Ny':
                data_sim[:,i,:] = data_sim_NxNyNz[:,1,:].copy()
            elif idata.WhatToResolve[i] == 'Nz':
                data_sim[:,i,:] = data_sim_NxNyNz[:,2,:].copy()
            elif idata.WhatToResolve[i] == 'Nparallel':
                data_sim[:,i,:] = data_sim_Nparallel[:,:].copy()
            elif idata.WhatToResolve[i] == 'phiN':
                data_sim[:,i,:] = data_sim_phiN[:,:].copy()
            elif idata.WhatToResolve[i] == 'Nperp':
                data_sim[:,i,:] = data_sim_Nperp[:,:].copy()
            elif idata.WhatToResolve[i] == 'Vx':
                data_sim[:,i,:] = data_sim_VxVyVz[:,0,:].copy()
            elif idata.WhatToResolve[i] == 'Vy':
                data_sim[:,i,:] = data_sim_VxVyVz[:,1,:].copy()
            elif idata.WhatToResolve[i] == 'Vz':
                data_sim[:,i,:] = data_sim_VxVyVz[:,2,:].copy()
            elif idata.WhatToResolve[i] == 'Psi':
                data_sim[:,i,:] = data_sim_Psi.copy()
            elif idata.WhatToResolve[i] == 'rho':
                data_sim[:,i,:] = np.sqrt(data_sim_Psi).copy()
            elif idata.WhatToResolve[i] == 'Theta':
                data_sim[:,i,:] = data_sim_Theta.copy()
            elif idata.WhatToResolve[i] == 'R':
                data_sim[:,i,:] = np.sqrt(data_sim_XYZ[:,0,:]**2+data_sim_XYZ[:,1,:]**2).copy() 
            else:
                print('IN THE WhatToResolve LIST IN THE INPUT FILE IS A NON-SUPPORTED ELEMENT. FIX THAT.\n')
                raise
       

        # Wfct 
        data_sim_Wfct = data_sim_Wfct * NormFactor
        # information, if rays have been scattered or not
        data_sim_scattered = data_sim_nmbrscattevents.copy()   

     
                    
        print("BINNING UNSCATTERED RAYS.\n")
        if idata.storeWfct == True:
            print("BINNING WITH WEIGHT 1.\n")
            sys.stdout.flush()
            # and do the binning 
            if uniform_bins == True:
                tmpnmbrrays = binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                    np.empty([0,0]),
                                    WfctUnscattered,
                                    min[0], max[0], min[1], max[1], min[2], max[2], min[3], max[3],
                                    nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                                    data_sim_time,
                                    1,   # binning unscattered rays
                                    data_sim_scattered,
                                    0)    # not the absorption binning scheme
            else:
                tmpnmbrrays = binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                    np.empty([0,0]),
                                    WfctUnscattered,
                                    bins[0], bins[1], bins[2], bins[3],
                                    data_sim_time,
                                    1,   # binning unscattered rays
                                    data_sim_scattered,
                                    0)

        # maybe also for the velocity field
        if idata.storeVelocityField == True:
            for k in range(0,len(VelocityComponentsToStore)):
                print("BINNING WITH WEIGHT %s.\n" %(VelocityComponentsToStore[k]))
                sys.stdout.flush()
                # choose the weight
                if VelocityComponentsToStore[k] == "Nx":
                    weight = data_sim_NxNyNz[:,0,:]
                elif VelocityComponentsToStore[k] == "Ny":
                    weight = data_sim_NxNyNz[:,1,:]
                elif VelocityComponentsToStore[k] == "Nz":
                    weight = data_sim_NxNyNz[:,2,:]
                elif VelocityComponentsToStore[k] == "Vx":
                    weight = data_sim_VxVyVz[:,0,:]
                elif VelocityComponentsToStore[k] == "Vy":
                    weight = data_sim_VxVyVz[:,1,:]
                elif VelocityComponentsToStore[k] == "Vz":
                    weight = data_sim_VxVyVz[:,2,:]
                elif VelocityComponentsToStore[k] == "Nparallel":
                    weight = data_sim_Nparallel[:,:]
                elif VelocityComponentsToStore[k] == "phiN":
                    weight = data_sim_phiN[:,:]
                elif VelocityComponentsToStore[k] == "Nperp":
                    weight = data_sim_Nperp[:,:]

                if uniform_bins == True:
                    tmpnmbrrays = binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                        weight,
                                        VelocityFieldUnscattered[:,:,:,:,k,:],
                                        min[0], max[0], min[1], max[1], min[2], max[2], min[3], max[3],
                                        nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                                        data_sim_time,
                                        1,         # binning unscattered rays
                                        data_sim_scattered,
                                        0)     # not the absorption binning scheme
                else:
                    tmpnmbrrays = binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                        weight,
                                        VelocityFieldUnscattered[:,:,:,:,k,:],
                                        bins[0], bins[1], bins[2], bins[3],
                                        data_sim_time,
                                        1,         # binning unscattered rays
                                        data_sim_scattered,
                                        0)

                

        if idata.storeAbsorption == True:
            print("BINNING WITH ABSORPTION WITH WEIGHT 1.\n")
            sys.stdout.flush()
            # and do the binning 
            if uniform_bins == True:
                tmpnmbrrays = binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                    np.empty([0,0]),
                                    AbsorptionUnscattered,
                                    min[0], max[0], min[1], max[1], min[2], max[2], min[3], max[3],
                                    nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                                    data_sim_time,
                                    1,   # binning unscattered rays
                                    data_sim_scattered,
                                    1)   # absorption scheme
            else:
                tmpnmbrrays = binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                    np.empty([0,0]),
                                    AbsorptionUnscattered,
                                    bins[0], bins[1], bins[2], bins[3],
                                    data_sim_time,
                                    1,   # binning unscattered rays
                                    data_sim_scattered,
                                    1)
    

        print('%i unscattered rays have been binned, %i available in total.\n' %(tmpnmbrrays, nmbrRaysToUse))
        nmbrRaysUnscattered += tmpnmbrrays

        # see if also the scattered rays are needed
        if idata.computeAmplitude == True or idata.computeScatteringEffect == True:
            print("BINNING SCATTERED RAYS.\n")
            
            if idata.storeWfct == True:
                print("BINNING WITH WEIGHT 1.\n")
                sys.stdout.flush()
                # and do the binning 
                if uniform_bins == True:
                    tmpnmbrrays = binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                      np.empty([0,0]),
                                      WfctScattered,
                                      min[0], max[0], min[1], max[1], min[2], max[2], min[3], max[3],
                                      nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                                      data_sim_time,
                                      2,   # binning scattered rays
                                      data_sim_scattered,
                                      0)
                else:

                    tmpnmbrrays = binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                      np.empty([0,0]),
                                      WfctScattered,
                                      bins[0], bins[1], bins[2], bins[3],
                                      data_sim_time,
                                      2,   # binning scattered rays
                                      data_sim_scattered,
                                      0)
                


            # maybe also for the velocity field
            if idata.storeVelocityField == True:
                for k in range(0,len(VelocityComponentsToStore)):
                    print("BINNING WITH WEIGHT %s.\n" %(VelocityComponentsToStore[k]))
                    sys.stdout.flush()
                    # choose the weight
                    if VelocityComponentsToStore[k] == "Nx":
                        weight = data_sim_NxNyNz[:,0,:].copy()
                    elif VelocityComponentsToStore[k] == "Ny":
                        weight = data_sim_NxNyNz[:,1,:].copy()
                    elif VelocityComponentsToStore[k] == "Nz":
                        weight = data_sim_NxNyNz[:,2,:].copy()
                    elif VelocityComponentsToStore[k] == "Vx":
                        weight = data_sim_VxVyVz[:,0,:].copy()
                    elif VelocityComponentsToStore[k] == "Vy":
                        weight = data_sim_VxVyVz[:,1,:].copy()
                    elif VelocityComponentsToStore[k] == "Vz":
                        weight = data_sim_VxVyVz[:,2,:].copy()
                    elif VelocityComponentsToStore[k] == "Nparallel":
                        weight = data_sim_Nparallel[:,:].copy()
                    elif VelocityComponentsToStore[k] == "phiN":
                        weight = data_sim_phiN[:,:].copy()
                    elif VelocityComponentsToStore[k] == "Nperp":
                        weight = data_sim_Nperp[:,:].copy()

                    if uniform_bins == True:
                        tmpnmbrrays = binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                            weight,
                                            VelocityFieldScattered[:,:,:,:,k,:],
                                            min[0], max[0], min[1], max[1], min[2], max[2], min[3], max[3],
                                            nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                                            data_sim_time,
                                            2,         # binning scattered rays
                                            data_sim_scattered,
                                            0)     # not the absorption binning scheme
                    else:
                        tmpnmbrrays = binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                            weight,
                                            VelocityFieldScattered[:,:,:,:,k,:],
                                            bins[0], bins[1], bins[2], bins[3],
                                            data_sim_time,
                                            2,         # binning scattered rays
                                            data_sim_scattered,
                                            0)

            
            if idata.storeAbsorption == True:
                print("BINNING WITH ABSORPTION WITH WEIGHT 1.\n")
                sys.stdout.flush()
                # and do the binning 
                if uniform_bins == True:

                    tmpnmbrrays = binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                        np.empty([0,0]),
                                        AbsorptionScattered,
                                        min[0], max[0], min[1], max[1], min[2], max[2], min[3], max[3],
                                        nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                                        data_sim_time,
                                        2,   # binning scattered rays
                                        data_sim_scattered,
                                        1)   # absorption scheme
                else:
                    tmpnmbrrays = binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                        np.empty([0,0]),
                                        AbsorptionScattered,
                                        bins[0], bins[1], bins[2], bins[3],
                                        data_sim_time,
                                        2,   # binning scattered rays
                                        data_sim_scattered,
                                        1)

                             
                    

            print('%i scattered rays have been binned, %i available in total.\n' %(tmpnmbrrays, nmbrRaysToUse))
            nmbrRaysScattered += tmpnmbrrays
            
            
                    
    print("COMPOSING AND NORMALISING THE FINAL QUANTITIES.\n")
    sys.stdout.flush()
    nmbrRays = nmbrRaysScattered + nmbrRaysUnscattered
    # IF CHOSEN IN INPUT FILE COMPUTE THE EFFECT OF SCATTERING
    ############################################################################
    if idata.computeScatteringEffect == True:
        
        if idata.storeWfct == True:
            # allocate memory for this
            WfctScatteringEffect = np.zeros(np.append(nmbr,2))
        
            # compute expectation value
            WfctScatteringEffect[:,:,:,:,0] = (WfctScattered[:,:,:,:,0] \
                                                   - nmbrRaysScattered/nmbrRaysUnscattered \
                                                   *WfctUnscattered[:,:,:,:,0]) / nmbrRays
            # and uncertainty if needed
            WfctScatteringEffect[:,:,:,:,1] = (WfctScattered[:,:,:,:,1] \
                                                   + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                   * WfctUnscattered[:,:,:,:,1]) / nmbrRays**2
            WfctScatteringEffect[:,:,:,:,1] = np.sqrt(WfctScatteringEffect[:,:,:,:,1])

        # and do the same for the velocity field
        if idata.storeVelocityField == True:
            VelocityFieldScatteringEffect = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore),2]))
            
            # compute expectation value
            VelocityFieldScatteringEffect[:,:,:,:,:,0] = (VelocityFieldScattered[:,:,:,:,:,0] \
                                                            - nmbrRaysScattered/nmbrRaysUnscattered \
                                                            *VelocityFieldUnscattered[:,:,:,:,:,0]) / nmbrRays
            # and uncertainty if needed
            VelocityFieldScatteringEffect[:,:,:,:,:,1] = (VelocityFieldScattered[:,:,:,:,:,1] \
                                                            + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                            *VelocityFieldUnscattered[:,:,:,:,:,1]) / nmbrRays**2
            VelocityFieldScatteringEffect[:,:,:,:,:,1] = np.sqrt(VelocityFieldScatteringEffect[:,:,:,:,:,1])

            
        if idata.storeAbsorption == True:
            # allocate memory for this
            AbsorptionScatteringEffect = np.zeros(np.append(nmbr,2))
        
            # compute expectation value
            AbsorptionScatteringEffect[:,:,:,:,0] = (AbsorptionScattered[:,:,:,:,0] \
                                                   - nmbrRaysScattered/nmbrRaysUnscattered \
                                                   *AbsorptionUnscattered[:,:,:,:,0]) / nmbrRays
          
            AbsorptionScatteringEffect[:,:,:,:,1] = (AbsorptionScattered[:,:,:,:,1] \
                                                         + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                         *AbsorptionUnscattered[:,:,:,:,1]) / nmbrRays**2
            AbsorptionScatteringEffect[:,:,:,:,1] = np.sqrt(AbsorptionScatteringEffect[:,:,:,:,1])
      

    # IF CHOSEN IN INPUT FILE ESTIMATE THE CONTRIBUTION OF SCATTERED RAYS
    ############################################################################
    if idata.computeScatteredContribution == True:
        if idata.storeWfct == True:
            WfctScatteredContribution = np.zeros(np.append(nmbr,2))
            WfctScatteredContribution[:,:,:,:,0] = WfctScattered[:,:,:,:,0] / nmbrRays  
            WfctScatteredContribution[:,:,:,:,1] = np.sqrt(WfctScattered[:,:,:,:,1]) / nmbrRays  

        if idata.storeVelocityField == True:
            VelocityFieldScatteredContribution = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore),2]))
            VelocityFieldScatteredContribution[:,:,:,:,:,0] = VelocityFieldScattered[:,:,:,:,:,0] / nmbrRays
            VelocityFieldScatteredContribution[:,:,:,:,:,1] = np.sqrt(VelocityFieldScattered[:,:,:,:,:,1]) / nmbrRays
            
        if idata.storeAbsorption == True:
            AbsorptionScatteredContribution = np.zeros(np.append(nmbr,2))
            AbsorptionScatteredContribution[:,:,:,:,0] = AbsorptionScattered[:,:,:,:,0] / nmbrRays
            AbsorptionScatteredContribution[:,:,:,:,1] = np.sqrt(AbsorptionScattered[:,:,:,:,1]) / nmbrRays

            


    # IF CHOSEN IN INPUT FILE COMPUTE THE TOTAL AMPLITUDE
    ############################################################################
    if idata.computeAmplitude:
        if idata.storeWfct == True:
            # allocate memory for this
            Wfct = np.zeros(np.append(nmbr,2))
            # compute expectation value
            Wfct[:,:,:,:,0] = (WfctScattered[:,:,:,:,0] + WfctUnscattered[:,:,:,:,0]) / nmbrRays
    
            # and uncertainty if needed
            Wfct[:,:,:,:,1] = np.sqrt(WfctScattered[:,:,:,:,1] + WfctUnscattered[:,:,:,:,1]) / nmbrRays
        
        # and do the same for the velocity field if needed
        if idata.storeVelocityField == True:
            VelocityField = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore),2]))

            VelocityField[:,:,:,:,:,0] = (VelocityFieldScattered[:,:,:,:,:,0] + VelocityFieldUnscattered[:,:,:,:,:,0]) \
                / nmbrRays

            VelocityField[:,:,:,:,:,1] = np.sqrt(VelocityFieldScattered[:,:,:,:,:,1] \
                                                   + VelocityFieldUnscattered[:,:,:,:,:,1]) / nmbrRays

        if idata.storeAbsorption == True:
            # allocate memory for this
            Absorption = np.zeros(np.append(nmbr,2))
            # compute expectation value
            Absorption[:,:,:,:,0] = (AbsorptionScattered[:,:,:,:,0] + AbsorptionUnscattered[:,:,:,:,0]) / nmbrRays
            Absorption[:,:,:,:,1] = np.sqrt(AbsorptionScattered[:,:,:,:,1] + AbsorptionUnscattered[:,:,:,:,1]) / nmbrRays 
                


    # IF CHOSEN IN INPUT FILE NORMALISE THE UNSCATTERED AMPLITUDE
    ############################################################################
    if idata.computeAmplitudeUnscattered == True:
        if idata.storeWfct == True:
            WfctUnscattered[:,:,:,:,0] = WfctUnscattered[:,:,:,:,0] / nmbrRaysUnscattered
            WfctUnscattered[:,:,:,:,1] = np.sqrt(WfctUnscattered[:,:,:,:,1]) / nmbrRaysUnscattered

        # and the same for the velocity field
        if idata.storeVelocityField == True:
            VelocityFieldUnscattered[:,:,:,:,:,0] = VelocityField[:,:,:,:,:,0] / nmbrRaysUnscattered
            VelocityFieldUnscattered[:,:,:,:,:,1] = np.sqrt(VelocityField[:,:,:,:,:,1]) / nmbrRaysUnscattered
    
        if idata.storeAbsorption == True:
            AbsorptionUnscattered[:,:,:,:,0] = AbsorptionUnscattered[:,:,:,:,0] / nmbrRaysUnscattered
            AbsorptionUnscattered[:,:,:,:,1] = np.sqrt(AbsorptionUnscattered[:,:,:,:,1]) / nmbrRaysUnscattered


    


    # REDUCE THE MATRIXES TO THE MEANINGFULL DIMENSIONS
    ############################################################################
    firstaxistosum = len(idata.WhatToResolve)
    for i in range(firstaxistosum,4):
        if idata.computeAmplitude == True:
            if idata.storeWfct == True:
                Wfct = np.sum(Wfct, axis=firstaxistosum)
            if idata.storeVelocityField == True:
                VelocityField = np.sum(VelocityField, axis=firstaxistosum)
            if idata.storeAbsorption == True:
                Absorption = np.sum(Absorption, axis=firstaxistosum)
        if idata.computeAmplitudeUnscattered == True:
            if idata.storeWfct == True:
                WfctUnscattered = np.sum(WfctUnscattered, axis=firstaxistosum)
            if idata.storeVelocityField == True:
                VelocityFieldUnscattered = np.sum(VelocityFieldUnscattered, axis=firstaxistosum)
            if idata.storeAbsorption == True:
                AbsorptionUnscattered = np.sum(AbsorptionUnscattered, axis=firstaxistosum)
        if idata.computeScatteringEffect == True:
            if idata.storeWfct == True:
                WfctScatteringEffect = np.sum(WfctScatteringEffect, axis=firstaxistosum)
            if idata.storeVelocityField == True:
                VelocityFieldScatteringEffect = np.sum(VelocityFieldScatteringEffect, axis=firstaxistosum)
            if idata.storeAbsorption == True:
                AbsorptionScatteringEffect = np.sum(AbsorptionScatteringEffect, axis=firstaxistosum)
        if idata.computeScatteredContribution == True:
            if idata.storeWfct == True:
                WfctScatteredContribution = np.sum(WfctScatteredContribution, axis=firstaxistosum)
            if idata.storeVelocityField == True:
                VelocityFieldScatteredContribution = np.sum(VelocityFieldScatteredContribution, axis=firstaxistosum)
            if idata.storeAbsorption == True:
                AbsorptionScatteredContribution = np.sum(AbsorptionScatteredContribution, axis=firstaxistosum)
            



    # WRITE THE RESULTS TO FILE outputfilename
    ############################################################################
    outputfilename = idata.outputdirectory + outputfilename + '.hdf5'
    print('write results to file %s \n' %(outputfilename))
    sys.stdout.flush()
    fid = h5py.File(outputfilename,'w') 
    # store the computed datasets
    if idata.computeAmplitude == True:
        if idata.storeWfct == True:
            fid.create_dataset("BinnedTraces", data=Wfct)
        if idata.storeVelocityField == True:
            fid.create_dataset("VelocityField", data=VelocityField)
        if idata.storeAbsorption == True:
            fid.create_dataset("Absorption", data=Absorption)
    if idata.computeAmplitudeUnscattered == True:
        if idata.storeWfct == True:
            fid.create_dataset("BinnedTracesUnscattered", data=WfctUnscattered)
        if idata.storeVelocityField == True:
            fid.create_dataset("VelocityFieldUnscattered", data=VelocityFieldUnscattered)
        if idata.storeAbsorption == True:
            fid.create_dataset("AbsorptionUnscattered", data=AbsorptionUnscattered)
    if idata.computeScatteringEffect == True:
        if idata.storeWfct == True:
            fid.create_dataset("BinnedTracesScatteringEffect", data=WfctScatteringEffect)
        if idata.storeVelocityField == True:
            fid.create_dataset("VelocityFieldScatteringEffect", data=VelocityFieldScatteringEffect)
        if idata.storeAbsorption == True:
            fid.create_dataset("AbsorptionScatteringEffect", data=AbsorptionScatteringEffect)
    if idata.computeScatteredContribution == True:
        if idata.storeWfct == True:
            fid.create_dataset("BinnedTracesScatteredContribution", data=WfctScatteredContribution)
        if idata.storeVelocityField == True:
            fid.create_dataset("VelocityFieldScattererContribution", data=VelocityFieldScatteredContribution)
        if idata.storeAbsorption == True:
            fid.create_dataset("AbsorptionScatteredContribution", data=AbsorptionScatteredContribution)

    if idata.storeVelocityField == True: 
        # compose a string where they are separated with ',':
        s = ''
        for i in range(0,len(VelocityComponentsToStore)):
            s += VelocityComponentsToStore[i]
            s += ','    
        fid.create_dataset("VelocityFieldStored", data=s)

    # compose a string where they are separated with ',':
    s = ''
    for i in range(0,len(idata.WhatToResolve)):
        s += idata.WhatToResolve[i]
        s += ','
    fid.create_dataset("WhatToResolve", data=s)

    # look what boundaries are the different directions
    if uniform_bins == True:
        for i in range(0,len(idata.WhatToResolve)):
            if idata.WhatToResolve[i] == 'X':
                fid.create_dataset("Xmin", data=idata.min[i])
                fid.create_dataset("Xmax", data=idata.max[i])
                fid.create_dataset("nmbrX", data=idata.nmbr[i])
                
            elif idata.WhatToResolve[i] == 'Y':
                fid.create_dataset("Ymin", data=idata.min[i])
                fid.create_dataset("Ymax", data=idata.max[i])
                fid.create_dataset("nmbrY", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Z':
                fid.create_dataset("Zmin", data=idata.min[i])
                fid.create_dataset("Zmax", data=idata.max[i])
                fid.create_dataset("nmbrZ", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Nx':
                fid.create_dataset("Nxmin", data=idata.min[i])
                fid.create_dataset("Nxmax", data=idata.max[i])
                fid.create_dataset("nmbrNx", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Ny':
                fid.create_dataset("Nymin", data=idata.min[i])
                fid.create_dataset("Nymax", data=idata.max[i])
                fid.create_dataset("nmbrNy", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Nz':
                fid.create_dataset("Nzmin", data=idata.min[i])
                fid.create_dataset("Nzmax", data=idata.max[i])
                fid.create_dataset("nmbrNz", data=idata.nmbr[i])
            
            elif idata.WhatToResolve[i] == 'Nparallel':
                fid.create_dataset("Nparallelmin", data=idata.min[i])
                fid.create_dataset("Nparallelmax", data=idata.max[i])
                fid.create_dataset("nmbrNparallel", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Nperp':
                fid.create_dataset("Nperpmin", data=idata.min[i])
                fid.create_dataset("Nperpmax", data=idata.max[i])
                fid.create_dataset("nmbrNperp", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'phiN':
                fid.create_dataset("phiNmin", data=idata.min[i])
                fid.create_dataset("phiNmax", data=idata.max[i])
                fid.create_dataset("nmbrphiN", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Psi':
                fid.create_dataset("Psimin", data=idata.min[i])
                fid.create_dataset("Psimax", data=idata.max[i])
                fid.create_dataset("nmbrPsi", data=idata.nmbr[i])
                
            elif idata.WhatToResolve[i] == 'rho':
                fid.create_dataset("rhomin", data=idata.min[i])
                fid.create_dataset("rhomax", data=idata.max[i])
                fid.create_dataset("nmbrrho", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'Theta':
                fid.create_dataset("Thetamin", data=idata.min[i])
                fid.create_dataset("Thetamax", data=idata.max[i])
                fid.create_dataset("nmbrTheta", data=idata.nmbr[i])

            elif idata.WhatToResolve[i] == 'R':
                fid.create_dataset("Rmin", data=idata.min[i])
                fid.create_dataset("Rmax", data=idata.max[i])
                fid.create_dataset("nmbrR", data=idata.nmbr[i])
    else:
        for i in range(0,len(idata.WhatToResolve)):
            if idata.WhatToResolve[i] == 'X':
                fid.create_dataset("Xbins", data=idata.bins[i])
                
            elif idata.WhatToResolve[i] == 'Y':
                fid.create_dataset("Ybins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Z':
                fid.create_dataset("Zbins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Nx':
                fid.create_dataset("Nxbins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Ny':
                fid.create_dataset("Nybins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Nz':
                fid.create_dataset("Nzbins", data=idata.bins[i])
            
            elif idata.WhatToResolve[i] == 'Nparallel':
                fid.create_dataset("Nparallelbins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Nperp':
                fid.create_dataset("Nperpbins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'phiN':
                fid.create_dataset("phiNbins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Psi':
                fid.create_dataset("Psibins", data=idata.bins[i])
                
            elif idata.WhatToResolve[i] == 'rho':
                fid.create_dataset("rhobins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'Theta':
                fid.create_dataset("Thetabins", data=idata.bins[i])

            elif idata.WhatToResolve[i] == 'R':
                fid.create_dataset("Rbins", data=idata.bins[i])

    fid.create_dataset("nmbrRays", data=nmbrRays)
    fid.create_dataset("nmbrRaysUnscattered", data=nmbrRaysUnscattered)
    fid.create_dataset("nmbrRaysScattered", data=nmbrRaysScattered)

    fid.create_dataset("Mode",data=sigma)
    fid.create_dataset("FreqGHz",data=freq)
    fid.create_dataset("antennapolangle",data=antennapolangle)
    fid.create_dataset("antennatorangle",data=antennatorangle)
    fid.create_dataset("rayStartX",data=rayStartX)
    fid.create_dataset("rayStartY",data=rayStartY)
    fid.create_dataset("rayStartZ",data=rayStartZ)
    fid.create_dataset("beamwidth1",data=beamwidth1)
    fid.create_dataset("beamwidth2",data=beamwidth2)
    fid.create_dataset("curvatureradius1",data=curvatureradius1)
    fid.create_dataset("curvatureradius2",data=curvatureradius2)
    fid.create_dataset("centraleta1",data=centraleta1)
    fid.create_dataset("centraleta2",data=centraleta2)
    fid.close()

# END OF FILE
