""" Define a function which takes the data from ray tracing and does the binning.
The results are written in an array in the parameter list.
"""

# Load standard modules
import numpy as np
import sys
cimport numpy as np

# and some c-functions
cdef extern from "math.h":
    double sqrt( double )
    double exp( double )
    double log( double )



############################################################################
# DEFINE SOME MINOR FUNCTIONS USED BELOW
############################################################################
# function which calculates the weighted average of quant0 and quant1
# where xi gives the weight. xi=0 means take purely quant0,
# xi=1 takes quant1
cdef inline linint(double xi, double quant0, double quant1):
   return xi*quant1 + (1.-xi)*quant0



def compute_midpoints(np.ndarray[double, ndim=1] x):
    cdef int n = x.shape[0]
    if n < 2:
        raise ValueError("Array must have at least two elements to compute midpoints")

    # Compute midpoints using NumPy's vectorized operations
    return x[:-1] + np.diff(x) / 2.0





# min...: min. bin value
# max...: max. bin value
# nmbr...: nmbr of bins in the corresponding direction
# nmbrRays: number of rays which were traced
# data_sim: data of the ray tracing
# data_sim_NxNyNz: if available ray tracing data on NxNyNz
# data_sim_NparallelphiN: the same for Nparallel, phiN
# Wfct: array where the results of the binning will be written, 5 dimensional:
#       index: 1st --> 1st direction
#              2nd --> 
#              3rd --> 
#              4th --> 4th direction
#              5th --> 0: data from the binning,
#                      1: uncertainty

# data_sim_weight:
# if an array gives the appropriate weight
# if size=0, weight 1 is used

# scatterparameter: 0 --> do the binning for both, scattered and unscattered rays
#                   1 --> binning only for unscattered rays
#                   2 --> binning only for scattered rays

# data_sim_scattered: 
# index: rays, contains the number of scattering kicks a ray has experienced.

# absorption == 1 --> use absorption binning strategy,
#               0 --> all normal binning

# time: time along the rays, array


# returns: number of rays which are binned
############################################################################
# DEFINE FUNCTION WHICH DOES THE BINNING
############################################################################
cpdef int binning(np.ndarray [double, ndim=3] data_sim,
                  np.ndarray [double, ndim=2] data_sim_Wfct,               
                  np.ndarray [double, ndim=2] data_sim_CorrectionFactor,
                  np.ndarray [double, ndim=2] data_sim_weight,
      	          np.ndarray [double, ndim=5] Wfct,
                  np.ndarray [double, ndim=1] bin1,
                  np.ndarray [double, ndim=1] bin2,
                  np.ndarray [double, ndim=1] bin3,
                  np.ndarray [double, ndim=1] bin4,
                  np.ndarray [double, ndim=2] time,
                  int scatterparameter, 
                  np.ndarray [np.int_t, ndim=1] data_sim_scattered,
                  int absorption):

   
   """ Crucial binning function. See source file for more information.
   """


   ############################################################################
   # check if the given parameters do make sense
   ############################################################################  
   
   cdef int CorrectionFactorGiven
   if data_sim_CorrectionFactor.size > 0:
      CorrectionFactorGiven = 1
   else:
      CorrectionFactorGiven = 0
      print('The correction factor for transforming the results to the ones corresponding to the physical Hamiltonian is not given. It is assumed to be 1.\n')
      sys.stdout.flush()

   cdef int weightGiven
   if data_sim_weight.size == 0:
      weightGiven = 0
   else:
      weightGiven = 1

   ############################################################################
   # print some information
   ############################################################################
   print('binning started ...\n')
   sys.stdout.flush()

   ############################################################################
   # estimate some more information out of the data_sim array given
   ############################################################################
   # number of traces
   cdef int nTraces = data_sim.shape[0]

   # number of points per ray
   cdef int nPoints = data_sim.shape[2]

   # get bin midpoints
   cdef np.ndarray[double, ndim=1] bin1c = compute_midpoints(bin1)
   cdef np.ndarray[double, ndim=1] bin2c = compute_midpoints(bin2)
   cdef np.ndarray[double, ndim=1] bin3c = compute_midpoints(bin3)
   cdef np.ndarray[double, ndim=1] bin4c = compute_midpoints(bin4)

   # side length of the cubes
   #cdef double Deltadir1 = (dir1max-dir1min)/nmbrdir1
   #cdef double Deltadir2 = (dir2max-dir2min)/nmbrdir2
   #cdef double Deltadir3 = (dir3max-dir3min)/nmbrdir3
   #cdef double Deltadir4 = (dir4max-dir4min)/nmbrdir4

   # define some variables which will be needed   
   # (...0: starting point, ...1: destination point)
   cdef int index
   cdef double t0, dir10, dir20, dir30, dir40, Wfct0, GrpVel0=1., CorrectionFactor0=1., weight0=1.         
   cdef double t1, dir11, dir21, dir31, dir41, Wfct1, GrpVel1=1., CorrectionFactor1=1., weight1=1.         
   cdef double WfctInt

   # define variables where the corresponding bin indices can be stored
   cdef int ndir10, ndir20, ndir30, ndir40
   cdef int ndir11, ndir21, ndir31, ndir41
 
   # define variable where the Wfct contribution of a ray in one bin is temporarily stored
   cdef double DeltaWfct

   # define variable, where the uncertainty for the recent bin is temporarily stored
   cdef double DeltaUncertainty

   # define parameter to parametrice the ray in between two points
   cdef double xidir1, xidir2, xidir3, xidir4
   cdef double dir1bound, dir2bound, dir3bound, dir4bound
    

   # define temporarily used variables
   cdef double xiUsed      # in order to store the xi which actually is in use
   cdef int whichxiUsed    # in order to store information on which xi is used
                           # 0 --> xidir1, 1 --> xidir2, 2 --> xidir3, 3 --> xidir4
    
                           # in order to temporarily store those quantities
   cdef double tempWfct, tempGrpVel=1., tempCorrectionFactor=1., tempweight=1.    

   # define variable where the binned rays are counted
   cdef int raycounter = 0

   ############################################################################
   # loop over the rays
   ############################################################################
   for i in range(0,nTraces):

      # sometimes print out some information.
      if (i*100/(int(nTraces/20)*20)) % 10 == 0:
         print('progress: binning ray %i / %i\n' %(i,nTraces))
         sys.stdout.flush()

      # see if the recent ray has to be binned or not
      if scatterparameter == 0:     # means: bin all rays
          pass
      elif scatterparameter == 1:   # means: only unscattered rays
          if data_sim_scattered[i] > 0:   # if ray is scattered: skip it
              continue
      elif scatterparameter == 2:   # means: only scattered rays
          if data_sim_scattered[i] == 0:  # if ray is not scattered: skipt it
              continue

      # if the ray is binned, increment the ray counter
      raycounter += 1


      # read the first ray point
      t0 = 0.
      index = 0
      dir10 = data_sim[i,0,index]
      dir20 = data_sim[i,1,index]
      dir30 = data_sim[i,2,index]
      dir40 = data_sim[i,3,index]
      Wfct0 = data_sim_Wfct[i,index]

      if CorrectionFactorGiven == 1:
         CorrectionFactor0 = data_sim_CorrectionFactor[i,index]
      if weightGiven == 0:
         weight0 = 1.
      else:
         weight0 = data_sim_weight[i,1]
    
      # and calculate the bin indices. the data in bin1 etc are the bin centres
      ndir10 = np.argmin(abs(dir10-bin1c))           
      ndir20 = np.argmin(abs(dir20-bin2c))           
      ndir30 = np.argmin(abs(dir30-bin3c))  
      ndir40 = np.argmin(abs(dir40-bin4c)) 
 
      # for the first bin, for the moment, the uncertainty to add is 0
      DeltaUncertainty = 0.
  
      
      ############################################################################
      # loop over the ray points
      ############################################################################
      # start with the third line, because in the first, there is the extra information
      # and the first ray point in the second line is already stored in point 0
      for j in range(1, nPoints):         
         # read the next ray point
         t1 = time[i,j] 
         dir11 = data_sim[i,0,j]
         dir21 = data_sim[i,1,j]
         dir31 = data_sim[i,2,j]
         dir41 = data_sim[i,3,j]
         Wfct1 = data_sim_Wfct[i,j]
  

         # if Wfct vanishes, this is because the ray tracing has been stopped for some reasons.
         # in this case, don't further consider the recent ray
         if Wfct1 == 0.:
             break

         if CorrectionFactorGiven == 1:
            CorrectionFactor1 = data_sim_CorrectionFactor[i,j]
         if weightGiven == 0:
            weight1 = 1.
         else:
            weight1 = data_sim_weight[i,j]

         # and calculate the bin indices
         ndir11 = np.argmin(abs(dir11-bin1c))            
         ndir21 = np.argmin(abs(dir21-bin2c))           
         ndir31 = np.argmin(abs(dir31-bin3c))      
         ndir41 = np.argmin(abs(dir41-bin4c)) 
        

         ############################################################################
         # loop over intersections with bin boundaries in between the two recent ray
         # points
         ############################################################################
         while True:
           
                  
            ############################################################################
            # if start and end point are in the same bin, the binning can directly be
            # done and one can continue with the next ray point
            ############################################################################
            # if the starting point and the end point are in the same bin, do the binning directly
            if ndir10 == ndir11 and ndir20 == ndir21 and ndir30 == ndir31 and ndir40 == ndir41:
               if ndir10 >= 0 and ndir10 < len(bin1c) \
                   and ndir20 >= 0 and ndir20 < len(bin2c) \
                   and ndir30 >= 0 and ndir30 < len(bin3c) \
                   and ndir40 >= 0 and ndir40 < len(bin4c):
                  # calculate the distance in between the two points (either the spacial distance or the time)
                  dist = (t1-t0)

                  # and add the right value to the bin. It must be properly weighted with the probability of
                  # launching the recent ray. The Wfct which is taken into account corresponds to the average value of
                  if absorption == 0:
                      # the two points in between of whom one does the binning.
                      DeltaWfct = dist * (Wfct0+Wfct1)/2. * (CorrectionFactor0+CorrectionFactor1)/2. * (weight0+weight1)/2.
                  else:   # if absorption is chosen it must be treated in a different way
                      DeltaWfct = (Wfct0 - Wfct1) * (weight0+weight1)/2.


                  Wfct[ndir10,ndir20,ndir30,ndir40,0] += DeltaWfct
                  DeltaUncertainty += DeltaWfct
                  
               # the next starting ray point is the recent end point
               # the bin indices are already the same.
               t0 = t1
               dir10 = dir11
               dir20 = dir21
               dir30 = dir31
               dir40 = dir41
               Wfct0 = Wfct1
               CorrectionFactor0 = CorrectionFactor1
               weight0 = weight1
               break

            #############################################################################
            # if start and end point are not in the same bin, do the binning step by step
            # for all the bins in between both.
            #############################################################################
            # calculate the next intersection with a bin boundary
            # and therefore calculate the relevant boundaries 
            # always the boundary which is at least to the right direction of the starting
            # point 0 is taken. Therefore, it is seen if the coordinates of point 1 are
            # superior or not.
            if dir11 >= dir10:
               dir1bound = bin1[ndir10+1]
            else:
               dir1bound = bin1[ndir10]
            if dir21 >= dir20:
               dir2bound = bin1[ndir20+1]
            else:
               dir2bound = bin1[ndir20]
            if dir31 >= dir30:
               dir3bound = bin1[ndir30+1]
            else:
               dir3bound = bin1[ndir30]
            if dir41 >= dir40:
               dir4bound = bin1[ndir40+1]
            else:
               dir4bound = bin1[ndir40]

            # the nearest boundary is taken into account. Therefore, parameters
            # xi... along the ray are calculated giving the intersection with the
            # next bin boundary in ... direction. xi... = 0 corresponds to point 0,
            # xi... = 1 gives point 1.
            # If the coordinates are the same, the xi cannot be calculated. Then  
            # it is set to 20 which is a value > 1 and thus this xi for sure is not
            # the next one (because even not in between point 0 and point 1) and 
            # ignored.
            if dir11-dir10 != 0.:
               xidir1 = (dir1bound-dir10)/(dir11-dir10)
            else:
               xidir1 = 20.   
            if dir21-dir20 != 0.:
               xidir2 = (dir2bound-dir20)/(dir21-dir20)
            else:
               xidir2 = 20.
            if dir31-dir30 != 0.:
               xidir3 = (dir3bound-dir30)/(dir31-dir30)
            else:
               xidir3 = 20.
            if dir41-dir40 != 0.:
               xidir4 = (dir4bound-dir40)/(dir41-dir40)
            else:
               xidir4 = 20.


            # do the binning for the recent bin (which means in between point 0 and 
            # an intermediate point at the next smallest reasonable xi)
            # look for the smallest reasonable xi and do the binning for the
            # corresponding part of the ray. For further explanation on the binning
            # look where the binning within one bin is done.
            if xidir1 < xidir2 and xidir1 < xidir3 and xidir1 < xidir4 and xidir1 <= 1.:
               xiUsed = xidir1
               whichxiUsed = 0
            elif xidir2 < xidir3 and xidir2 < xidir4 and xidir2 < 1.:
               xiUsed = xidir2
               whichxiUsed = 1
            elif xidir3 < xidir4 and xidir3 < 1.:
               xiUsed = xidir3
               whichxiUsed = 2
            elif xidir4 < 1.:
               xiUsed = xidir4
               whichxiUsed = 3

            # if there was no reasonable xi this means, that there was some problem due to rounding issues which occurs
            # whenever a ray is oriented exactly along a bin boundary and thus practically only for the vacuum case 
            # and the central ray.
            else:  
               # print a warning message. 
               dist = t1-t0
               print("WARNING: a problem in the binning occured. For ray number=%i, dest. point number=%i. dist=%f[gen.time unit] could not be binned.\n" %(i,j-1,dist))
               sys.stdout.flush()

               # go one with the next ray point
               t0 = t1
               dir10 = dir11
               dir20 = dir21
               dir30 = dir31
               dir40 = dir41
               Wfct0 = Wfct1
               CorrectionFactor0 = CorrectionFactor1
               weight0 = weight1
               ndir10 = ndir11
               ndir20 = ndir21
               ndir30 = ndir31
               ndir40 = ndir41
               break
           

            # compute the intermediate quantities and save in temporare variables 
            tempWfct = linint(xiUsed, Wfct0, Wfct1)
            tempCorrectionFactor = linint(xiUsed, CorrectionFactor0, CorrectionFactor1)
            tempweight = linint(xiUsed, weight0, weight1)

            # and do the binning
            if absorption == 0:
               dist = (t1-t0)*xiUsed
               DeltaWfct = dist * (Wfct0+tempWfct) / 2. * (CorrectionFactor0+tempCorrectionFactor)/2. * (weight0+tempweight)/2.
            else:   # for absorption different computation
               if Wfct1 > 0.:
                  WfctInt = Wfct0 * exp(log(Wfct1/Wfct0)*xiUsed) 
               else:
                  WfctInt = 0.
               DeltaWfct = (Wfct0 - WfctInt) * (weight0+tempweight)/2.
             
            # if no problem occured, see if indices are inside the boundaries and if yes do the binning.
            if ndir10 >= 0 and ndir10 < len(bin1c) \
                and ndir20 >= 0 and ndir20 < len(bin2c) \
                and ndir30 >= 0 and ndir30 < len(bin3c) \
                and ndir40 >= 0 and ndir40 < len(bin4c):

               Wfct[ndir10,ndir20,ndir30,ndir40,0] += DeltaWfct
               Wfct[ndir10,ndir20,ndir30,ndir40,1] += (DeltaWfct+DeltaUncertainty)**2

            DeltaUncertainty = 0.

            # the recent intermediate point is the next starting point
            t0 = linint(xiUsed, t0, t1)
            dir10 = linint(xiUsed, dir10, dir11)
            dir20 = linint(xiUsed, dir20, dir21)
            dir30 = linint(xiUsed, dir30, dir31)
            dir40 = linint(xiUsed, dir40, dir41)
            if absorption == 0:
                Wfct0 = linint(xiUsed, Wfct0, Wfct1)
            else:
                Wfct0 = WfctInt
            CorrectionFactor0 = linint(xiUsed, CorrectionFactor0, CorrectionFactor1)
            weight0 = linint(xiUsed, weight0, weight1)
          
            # and the bin index of the recent bin is moved into the right neighbour bin.
            # therefore, see which bin index has to be changed
            if whichxiUsed == 0:
               if dir11 > dir10:
                  ndir10 += 1
               else:
                  ndir10 -= 1               
            elif whichxiUsed == 1:
               if dir21 > dir20:
                  ndir20 += 1
               else:
                  ndir20 -= 1    
            elif whichxiUsed == 2:
               if dir31 > dir30:
                  ndir30 += 1
               else:
                  ndir30 -= 1    
            elif whichxiUsed == 3:
               if dir41 > dir40:
                  ndir40 += 1
               else:
                  ndir40 -= 1    

            
   # return the number of rays binned
   return raycounter


