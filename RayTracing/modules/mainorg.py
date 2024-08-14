"""In this file, the main-function for the organisation cores is defined.
This function generates initial conditions for the rays, sends them to the
ray tracing cores using MPI and waits for their results via MPI.
It collects the results and writes them to a hdf5-file."""

#
# MAIN FILE FOR ORGANISATION CORES
#



###########################################################################
# IMPORT
###########################################################################

# Load standard modules
import sys
import numpy as np
import math
import time   
import h5py
from mpi4py import MPI
# Load local modules
import CommonModules.physics_constants as phys
import RayTracing.modules.random_numbers    as rn
from RayTracing.modules.rotation_matrix import rotMatrix
from RayTracing.modules.MetropolisHastingsBoundaryCond import RayInit 
from RayTracing.modules.dispersion_matrix_cfunctions import * 



###########################################################################
# MAIN ROUTINE FOR ORGANISATION CORES
###########################################################################
def mainOrg(inputfilename, idata, comm):

    #######################################################################
    # ORGANISE MPI WORLD
    #######################################################################
    rank=comm.rank      # core index
    size = comm.size    # number of cores in total
    # number of cores tracing rays in total
    nmbrCPUperGroup = idata.nmbrCPUperGroup
    nmbrTracecpusTotal = size - int((size - 1) / nmbrCPUperGroup) - 1
    # number of rays each core has to trace
    nmbrRaysPerCore = int(idata.nmbrRays / nmbrTracecpusTotal)

    # number of cores for ray tracing this organisation core is 
    # responsible for (not more than nmbrCPUperGroup-1, because then a second 
    # organisation core is used, if the total number of cores is not a multiple 
    # of nmbrCPUperGroup, the last organisation core will be responsible for 
    # less than nmbrCPUperGroup trace-cores).
    nmbrRayTracingCores = size - rank - 1
    if nmbrRayTracingCores > nmbrCPUperGroup - 1:
        nmbrRayTracingCores = nmbrCPUperGroup - 1
    
    # see for how many rays this organisation core is 
    # responsible
    nmbrRaysForThisOrgCore = nmbrRayTracingCores * nmbrRaysPerCore
    
    ############################################################################
    # INITIALISE METROPOLIS HASTINGS INITIAL CONDITIONS GENERATION OBJECT
    ############################################################################
    print('rank %i initializing the ray parameters to be traced...\n' %(rank))
    sys.stdout.flush()

    # Create a random number generator
    freeze_seed = idata.freeze_random_numbers
    InitialRay_random_state = rn.create_numpy_random_state(freeze_seed, rank)
     
    # create Metropolis-Hastings object (initialization of rays) 
    InitialRayMetrHast = RayInit(idata, InitialRay_random_state)  
        
    # initialise with the most probable values
    InitialRayMetrHast.SetInitialValue(0., 0., idata.centraleta1, idata.centraleta2)  
        
    # initialise the random number generator, therefore let it run 
    # the nmbrMetrHastBoundaryInit times as indicated in the input file
    InitialRayMetrHast.InitialiseMHAlg(idata.nmbrMetrHastBoundaryInit)
        
    # compute the normalisation constant called C in the handwritten notes 
    # between the Wigner function and the probability distribution
    normfact = InitialRayMetrHast.GenerateNormalisation()
       


    ############################################################################
    # FOR ALL CORES THIS ONE IS RESPONSIBLE FOR GENERATE RAY PARAMETERS
    ############################################################################
    for recentcore in range(0,nmbrRayTracingCores):

        # generate list where the ray parameters will be stored
        ray_parameters = []
    
        # initialise rays for this current core
        # Remark: one extra ray (i=-1) is traced and used for the cores to rougly 
        # determine the position of absorption for step size adaptation
        for i in range(-1,nmbrRaysPerCore):
    
            # If this is the first ray of the first rank and if it is chosen 
            # in the input file, do not generate the initial conditions using 
            # the MH-object, but just use the most probable ones 
            # (i.e. take  the central ray)
            # The logical condition for central ray calculation is
            #     TraceCentralRay = idata.takecentralrayfirst  
            # Then, the central ray is indexed by i = 0             
            #     TraceCentralRay = TraceCentralRay and i==0   
            # Only for the first core of the group ...
            #     TraceCentralRay = TraceCentralRay and recentcore==0
            # and only for the first group of cpus ... 
            #     TraceCentralRay = TraceCentralRay and rank==0    
            # All together one has (combining the conditions is twice as fast)
            TraceCentralRay = idata.takecentralrayfirst and \
                            i==0  and                     \
                            recentcore==0 and             \
                            rank==0   

            # Both the central ray and the extra ray are issued from the beam center
            if TraceCentralRay or i == -1:             
                Y1 = 0.
                Y2 = 0.
                eta1 = idata.centraleta1
                eta2 = idata.centraleta2

                if idata.twodim == True:  
                    eta2 = 0.

            # otherwise: use MH-object for the initial condition for the ray.
            else:
                Y1, Y2, eta1, eta2 = InitialRayMetrHast.GenerateRandom()
                      
            # calculate the third component of the wavevector such that 
            # the norm is unity (assuming free space around the antenna)
            etan = math.sqrt(1 - eta1**2 - eta2**2)
                        
            # unless the lense like medium (valley) is considered.
            # here, even around the antenna there is no free space and 
            # a different etan must be taken
            if idata.valley == True:
                etan = math.sqrt(1 - eta1**2 - eta2**2 -
                                 Y1**2/idata.linearlayervalleyL**2 - 
                                 Y2**2/idata.linearlayervalleyL**2)
                
            # see if ITER or ASDEX specification is used for the injection 
            # angles (the default is ASDEX specification)
            try:
                anglespecification = idata.anglespecification	
            except AttributeError:
                anglespecification = 'ASDEX'
            if anglespecification == 'ASDEX':
                antennapolangle = -idata.antennapoldeg/180.*math.pi
                antennatorangle = idata.antennatordeg/180.*math.pi
            elif anglespecification == 'ITER':
                alpha = -idata.antennapoldeg/180.*math.pi
                beta = idata.antennatordeg/180.*math.pi
                antennapolangle = math.asin(math.cos(beta)*math.sin(alpha))
                antennatorangle = -math.atan(math.tan(beta)/math.cos(alpha))
            else:
                # in case, something is specified, but not 'ASDEX' nor 'ITER'
                msg = """Variable "anglespecification" can be either 'ITER' 
                         or 'ASDEX'. Input variable not understood."""
                raise ValuetError(msg)	    	
		
            # use the function given above to compute the rotational matrix
            T = rotMatrix(antennapolangle, antennatorangle)
                   
            # calculate the wavevector in laboratory system
            Nx = T[0,0]*eta1 + T[0,1]*eta2 + T[0,2]*etan
            Ny = T[1,0]*eta1 + T[1,1]*eta2 + T[1,2]*etan
            Nz = T[2,0]*eta1 + T[2,1]*eta2 + T[2,2]*etan
        
            # and the starting point in laboratory system
            X = idata.rayStartX + T[0,0]*Y1 + T[0,1]*Y2
            Y = idata.rayStartY + T[1,0]*Y1 + T[1,1]*Y2
            Z = idata.rayStartZ + T[2,0]*Y1 + T[2,1]*Y2

            # set the initial mode to store it in the parameters
            sigma_start = idata.sigma

            # here is some historical confusion:
            # Wfct used to be the value of the Wigner function in the old
            # scattering scheme and probfunction the probability of 
            # tracing the ray.
            # now, instead, probfunction is meaningless and must be set to
            # 1, because it might still be used as a prefactor for the binning.
            # Wfct has the meaning of the function f normalised with the
            # probability of launching the ray. Hence, it is only the prefactor
            # which relates the Gaussian shaped Wigner function with the
            # initial probability distribution
            Wfct = normfact
            probfunction = 1.   

            # Store the initial condition of this ray as a disctionary
            InitialRay = {'X': X,
                          'Y': Y,
                          'Z': Z,
                          'Nx': Nx,
                          'Ny': Ny,
                          'Nz': Nz,
                          'Wfct': Wfct,
                          'probfunction': probfunction,
                          'initial mode index': sigma_start}

            # ... and add to ray_parameter list
            ray_parameters.append(InitialRay)

               
        # convert ray_parameters to an ndarray and roll it back
        # so that the first entry, which corresponds to i = -1, 
        # goes to the last position consistently with negative indexing
        ray_parameters = np.roll(np.array(ray_parameters), -1, axis=0)

        # send ray parameters to the appropriate core
        print('rank %i sending ray parameters to rank %i.\n' 
              %(rank,rank+recentcore+1))
        sys.stdout.flush()
        comm.send(ray_parameters, dest=rank+recentcore+1, tag=1)


    ############################################################################
    # SIGNAL THAT RAY TRACING CALCULATIONS HAVE BEEN SUCCESSFULLY INITIALIZED
    ############################################################################
    print('total number of rays this rank %i is responsible for=%i\n' 
          %(rank,nmbrRaysForThisOrgCore))
    sys.stdout.flush()


    ############################################################################
    # INITIALIZE HDF5 DATA STRUCTURES
    ############################################################################

    # open hdf5 file for this rank
    filename = "_file%i.hdf5" %(int(rank/nmbrCPUperGroup)) 
    filename = idata.output_filename + filename 
    fid = h5py.File(idata.output_dir+filename,'w') 

    # directly store parameters of the run
    fid.create_dataset("Mode", data=idata.sigma)  # initial mode only
    fid.create_dataset("FreqGHz", data=idata.freq)
    fid.create_dataset("antennapolangle", data=idata.antennapoldeg/180.*math.pi)
    fid.create_dataset("antennatorangle", data=idata.antennatordeg/180.*math.pi)
    fid.create_dataset("nmbrRays", data=nmbrRaysForThisOrgCore)
    fid.create_dataset("rayStartX", data=idata.rayStartX)
    fid.create_dataset("rayStartY", data=idata.rayStartY)
    fid.create_dataset("rayStartZ", data=idata.rayStartZ)
    fid.create_dataset("beamwidth1", data=idata.beamwidth1)
    fid.create_dataset("beamwidth2", data=idata.beamwidth2)
    fid.create_dataset("curvatureradius1", data=idata.curvatureradius1)
    fid.create_dataset("curvatureradius2", data=idata.curvatureradius2)
    fid.create_dataset("centraleta1", data=idata.centraleta1)
    fid.create_dataset("centraleta2", data=idata.centraleta2)


    # write input file and rank into output
    fid.create_dataset("fileindex", data=int(rank/nmbrCPUperGroup))
    
    f = open(inputfilename, 'r')
    lines = f.readlines()
    ifile = ''
    for i in range(0,len(lines)):
        ifile += lines[i]
    f.close()
    
    fid.create_dataset("inputfile", data=ifile)


    # shapes of the ray tracing datasets
    parameter = (nmbrRaysForThisOrgCore,)
    scalar = (nmbrRaysForThisOrgCore, idata.npt)
    vector = (nmbrRaysForThisOrgCore, 3, idata.npt)

    # create ray tracing datasets
    h5XYZ = fid.create_dataset("TracesXYZ", vector, dtype='float64')
    h5Wfct = fid.create_dataset("TracesWfct", scalar, dtype='float64')
    h5time = fid.create_dataset("TracesTime", scalar, dtype='float64')
    h5mode = fid.create_dataset("TracesMode", scalar, dtype='float64')
    h5nmbrscattevents = fid.create_dataset("TracesNumberScattEvents", parameter, dtype='int')
    if idata.storeNxNyNz:
        h5NxNyNz = fid.create_dataset("TracesNxNyNz", vector, dtype='float64')
    if idata.storeNparallelphiN:
        h5Nparallel = fid.create_dataset("TracesNparallel", scalar, dtype='float64')
        h5Nperpendicular = fid.create_dataset("TracesNperpendicular", scalar, dtype='float64')
        h5phiN = fid.create_dataset("TracesphiN", scalar, dtype='float64')
    if idata.storeGroupVelocity:
        h5VxVyVz = fid.create_dataset("TracesGroupVelocity", vector, dtype='float64')
    if idata.storeCorrectionFactor:
        h5xiFactor = fid.create_dataset("TracesCorrectionFactor", scalar, dtype='float64')
    if idata.storePsi:
        h5Psi = fid.create_dataset("TracesPsi", scalar, dtype='float64')
    if idata.storeTheta:
        h5Theta = fid.create_dataset("TracesTheta", scalar, dtype='float64')


    ############################################################################
    # AND WAIT FOR THE RESULTS TO BE COPIED INTO THE MEMORY
    ############################################################################
    print("rank %i waiting for results ...\n" %(rank))
    sys.stdout.flush()

    # initialise some variables to see when all results are received.
    raystraced = 0
    raystracedpercpu = 0
    t = time.time()
    
    # for control output: see what number of rays has to be printed
    # if nothing is defined in the input file, 100 is chosen.
    if idata.ControlOutput == True:
        try:
            PrintNmbrRays = idata.PrintNmbrRays
        except:
            PrintNmbrRays = 100

    # receive results one after the other from slave CPUs
    for j in range(0,nmbrRaysPerCore):
        for i in range(0,nmbrRayTracingCores):

            # wait for result from slave ranks
            _rank, ray, result = comm.recv(source=rank+i+1, tag=2)   
            
            # copy results to the appropriate memory cell 
            # X,Y,Z information
            h5XYZ[j+nmbrRaysPerCore*i,0:3] = result['orbit'][0:3].copy()

            # Wfct information
            h5Wfct[j+nmbrRaysPerCore*i] = result['Wfct'].copy()

            # time desired
            h5time[j+nmbrRaysPerCore*i] = result['time'].copy()

            # mode index
            h5mode[j+nmbrRaysPerCore*i] = result['mode index'].copy()

            # number of scattering events for each ray
            h5nmbrscattevents[j+nmbrRaysPerCore*i] = result['n. scatt. events']

            # Nx,Ny,Nz information if desired
            if idata.storeNxNyNz:
                h5NxNyNz[j+nmbrRaysPerCore*i,0:3] = result['orbit'][3:6].copy()

            # Nparallel, phiN information if desired
            if idata.storeNparallelphiN:
                h5Nparallel[j+nmbrRaysPerCore*i] = result['N parallel'].copy()
                h5Nperpendicular[j+nmbrRaysPerCore*i] = result['N perp'].copy()
                h5phiN[j+nmbrRaysPerCore*i] = result['phi_N'].copy()

            # group velocity information if desired
            if idata.storeGroupVelocity:
                h5VxVyVz[j+nmbrRaysPerCore*i] = result['V Group'].copy()

            # correction factor information to encounter for the group velocity if desired
            if idata.storeCorrectionFactor:
                h5xiFactor[j+nmbrRaysPerCore*i] = result['scaling'].copy()

            # Psi if desired
            if idata.storePsi:
                h5Psi[j+nmbrRaysPerCore*i] = result['Psi'].copy()

            # Theta if desired
            if idata.storeTheta:
                h5Theta[j+nmbrRaysPerCore*i] = result['Theta'].copy()

            # note, that one result has been received
            raystraced += 1
            # and print out progress if desired 
            if idata.ControlOutput:
                if raystraced % PrintNmbrRays == 0:
                    print("organisation core rank %i traced %i / %i rays in %d seconds" %(rank, raystraced, nmbrRaysForThisOrgCore, time.time()-t))
                    sys.stdout.flush()

    ############################################################################
    # END OF RAY TRACING - CLOSE THE FILE
    ############################################################################
    fid.close()

    print('rank %i has written results to file %s...\n' %(rank, idata.output_dir+filename))
    sys.stdout.flush()
    
# END OF FILE
