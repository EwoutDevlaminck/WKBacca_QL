"""In this file, function related to the dispersion matrix and derived 
quantities as the Hamiltonian and so on are defined.
This is a cython file."""




##############################################################################
# IMPORT STATEMENTS
##############################################################################
from RayTracing.modules.atanrightbranch import atanRightBranch
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    double sqrt( double )
    double sin( double )
    double cos( double )
    double acos( double )
    double asin( double )
    double atan2(double, double)



##############################################################################
# DEFINE SOME CONSTANTS
##############################################################################
cdef double emass = 9.11*1e-31            # electron mass in kg
cdef double echarge = 1.602*1e-19         # electron charge in C
cdef double epsilon0 = 8.85*1e-12         # dielectric constant in V/Cm

    

##############################################################################
# SOME BASIC FUNCTIONS ARE DEFINED HERE.
##############################################################################

# fct that provides the in-plane coordinates (R,z) for a given (x,y,z)
cpdef double disROutOf(double X, double Y):
    # R must be the norm of (x,y)
    return sqrt(X**2 + Y**2)

# fct that provides the in-plane angle theta for given (R, Z)
cpdef double disThetaOutOf(double R, double Z):
    return atan2(Z, R)
    
# fct that returns the electron cyclotron frequency in rad/s
# argument: magnetic field norm in T
cpdef double disParamOmega(double B):
    return B*echarge / emass   
        
# fct that returns plasma frequency in rad/s
# argument: electron density in 1e19 m^-3
cpdef double disParamomegaP(double Ne):
    return echarge*sqrt(Ne*1e19/epsilon0/emass)

# fct that returns S
# argument: beam omega in rad/s, magnetic field norm in T, 
#           electron density in 1e19 m^-3 = 1e13 cm^-3
cpdef double disParamS(double omega, double B, double Ne):
    return 1. - disParamomegaP(Ne)**2 / (omega**2 - disParamOmega(B)**2)

# fct that returns P
# argument: beam omega in rad/s, electron density in 1e19 m^-3 = 1e13 cm^-3
cpdef double disParamP(double omega, double Ne):
    return 1. - disParamomegaP(Ne)**2 / omega**2

# fct that returns D
# argument: beam omega in rad/s, magnetic field norm in T, 
#           electron density in 1e19 m^-3 = 1e13 cm^-3
cpdef double disParamD(double omega, double B, double Ne):
    return disParamOmega(B) / (omega**2 - disParamOmega(B)**2) * disParamomegaP(Ne)**2 / omega





################################################################
# A FUNCTION WHICH PROVIDES THE ROTATIONAL EULER ANGLES
################################################################
# returns [alpha,beta]
# arguments:
# magnetic field components Bt, BR, Bz in any unit
# coordinates X,Y,Z
cpdef np.ndarray [double , ndim = 1] disrotMatrixAngles(double Bt, double BR, double Bz, double X , double Y, double Z):
      
      # first of all read out the magnetic field
      cdef double R = disROutOf(X,Y)	  
      # the magnetic field has to be turned around the z axis because the direction given in 
      # DispersionMatrix is based on the coordinate system (R,t,z) where t only
      # coincides with y in case y=0
      
      # now calculate the angle this field has to be turned by around the z axis.
      cdef double angle = atanRightBranch(Y, X)          

      # and then turn the magnetic field
      cdef double Bx = BR*cos(angle) - Bt*sin(angle)
      cdef double By = BR*sin(angle) + Bt*cos(angle)

      # and normalize it
      norm = sqrt(Bx**2 + By**2 + Bz**2)
      bx = Bx / norm
      by = By / norm
      bz = Bz / norm

      # now calculate the two Euler angles needed
      cdef double alpha = -atanRightBranch(by, bx)    
      cdef double beta = acos(bz)		        

      return np.array([alpha, beta], dtype=np.double)


################################################################
# A FUNCTION WHICH PROVIDES THE ROTATIONAL MATRIX Q
################################################################
# returns Q
# arguments: Euler angles alpha, beta
cpdef np.ndarray [double, ndim=2] disrotMatrixQ(double alpha, double beta):  
       
    """This function provides the rotational matrix Q which allows
    a transformation between coordinates relative to the magnetic
    field and the cartesian coordinates system (x,y,z)
    """
      
    # and finally define the rotational matrix Q
    cdef np.ndarray [double , ndim =2] Q = np.empty([3,3], dtype=np.double)  # first index is considered to be the row
    
    Q[0,0] =  cos(alpha)*cos(beta)
    Q[1,0] = -sin(alpha)*cos(beta)
    Q[2,0] =            -sin(beta)
    Q[0,1] =  sin(alpha)
    Q[1,1] =  cos(alpha)
    Q[2,1] =  0.
    Q[0,2] =  cos(alpha)*sin(beta)
    Q[1,2] = -sin(alpha)*sin(beta)
    Q[2,2] =             cos(beta)

    #and return this matrix
    return Q


################################################################
# FUNCTIONS TRANSFORMING REFRACTIVE INDEX COMPONENTS
# USING THE ROTATIONAL MATRIX Q
################################################################
# calculate the parallel and perpendicular wavevector components
# and the angle phiN.
# arguments: 
# Euler angles alpha, beta
# refractive index vector in cartesian coordinates Nx,Ny,Nz
cpdef np.ndarray [double, ndim=1] disNparallelNperpphiNOutOfNxNyNz(double alpha, double beta,
      		 	  	  				double Nx, double Ny, double Nz):
         
    cdef np.ndarray [double, ndim=2] Q = disrotMatrixQ(alpha,beta)
    cdef double Nparallel = Q.T[2,0]*Nx + Q.T[2,1]*Ny + Q.T[2,2]*Nz
    cdef double Nperp1 = Q.T[0,0]*Nx + Q.T[0,1]*Ny + Q.T[0,2]*Nz
    cdef double Nperp2 = Q.T[1,0]*Nx + Q.T[1,1]*Ny + Q.T[1,2]*Nz
    cdef double Nperp = sqrt(Nperp1**2 + Nperp2**2)
    cdef double phiN = atanRightBranch(Nperp2,Nperp1)

    return np.array([Nparallel, Nperp, phiN],dtype=np.double)

    
# calculate the cartesian wavevector components
# out of Nparallel, Nperp, phiN
# arguments:
# Euler angles alpha, beta
# refractive index vector components Nparallel, Nperp, phiN
cpdef np.ndarray [double, ndim=1] disNxNyNzOutOfNparallelNperpphiN(double alpha, double beta,
                                                                    double Nparallel,double Nperp,double phiN):

    cdef np.ndarray [double, ndim=2] Q = disrotMatrixQ(alpha,beta)
    Nperp1 = Nperp*cos(phiN)
    Nperp2 = Nperp*sin(phiN)

    cdef double Nx = Q[0,0]*Nperp1 + Q[0,1]*Nperp2 + Q[0,2]*Nparallel
    cdef double Ny = Q[1,0]*Nperp1 + Q[1,1]*Nperp2 + Q[1,2]*Nparallel
    cdef double Nz = Q[2,0]*Nperp1 + Q[2,1]*Nperp2 + Q[2,2]*Nparallel        
    
    return np.array([Nx, Ny, Nz], dtype=np.double)

   
# calculate the derivatives of Nparallel and Nperp
# with respect to R and z
# arguments:
# derivatives of B-components, 
# coordinates X,Y,Z
# refractive index vector Nx, Ny, Nz
cpdef np.ndarray [double, ndim=1] disNparallelNperpDerivative(double Bt, double dBt_dR, double dBt_dz,
    	  	     	      	      			       double BR, double dBR_dR, double dBR_dz,
							       double Bz, double dBz_dR, double dBz_dz,
    	  	     	      	      			       double X, double Y, double Z,
							       double Nx, double Ny, double Nz):
        

    # calculate the rotational Euler angles
    cdef double alpha
    cdef double beta
    alpha, beta = disrotMatrixAngles(Bt,BR,Bz,X,Y,Z)
   
    # calculate Nparallel and Nperp
    Nparallel, Nperp, phiN = disNparallelNperpphiNOutOfNxNyNz(alpha,beta,Nx,Ny,Nz)

    # calculate cos(beta) and sin(beta) and the same for alpha
    cosbeta = cos(beta)
    sinbeta = sin(beta)
    cosalpha = cos(alpha)
    sinalpha = sin(alpha)

    # Nparallel, Nperp do not depend on alpha, so only
    # calculate the derivatives with respect to beta
    dNparallel_dalpha = -Nx*sinalpha*sinbeta - Ny*cosalpha*sinbeta
    dNparallel_dbeta = Nx*cosalpha*cosbeta - Ny*sinalpha*cosbeta - Nz*sinbeta
    dNperp_dalpha = sinbeta/Nperp * ( sinalpha*cosalpha*sinbeta*(Nx**2-Ny**2) \
                                        + (cosalpha**2-sinalpha**2)*sinbeta*Nx*Ny \
                                        + (cosalpha*Ny+sinalpha*Nx) * cosbeta*Nz)
    dNperp_dbeta = sinbeta*cosbeta*( Nz**2 - (cosalpha*Nx-sinalpha*Ny)**2 ) / Nperp \
                     - (cosalpha*Nx-sinalpha*Ny)*(cosbeta**2-sinbeta**2)*Nz / Nperp
    
    cdef double R = disROutOf(X,Y)
    cdef double dBt_dX = X / R * dBt_dR
    cdef double dBt_dY = Y / R * dBt_dR
    
    cdef double dBR_dX = X / R * dBR_dR
    cdef double dBR_dY = Y / R * dBR_dR

    cdef double dBz_dX = X / R * dBz_dR
    cdef double dBz_dY = Y / R * dBz_dR

    # calculate the derivatives of the Euler angle beta
    cdef double dbeta_dR = -1./sqrt( BR**2+Bt**2 ) / ( 1. + Bz**2/(BR**2+Bt**2) ) \
               		   * ( dBz_dR - Bz/(BR**2+Bt**2)*( BR*dBR_dR + Bt*dBt_dR ))
    cdef double dbeta_dz = -1./sqrt( BR**2+Bt**2 ) / ( 1. + Bz**2/(BR**2+Bt**2) ) \
               		   * ( dBz_dz - Bz/(BR**2+Bt**2)*( BR*dBR_dz + Bt*dBt_dz ))
    
    cdef double dbeta_dX = X / R * dbeta_dR
    cdef double dbeta_dY = Y / R * dbeta_dR

    cdef double dalpha_dX = -(BR*dBt_dX - Bt*dBR_dX)/(BR**2+Bt**2) + Y / R**2
    cdef double dalpha_dY = -(BR*dBt_dY - Bt*dBR_dY)/(BR**2+Bt**2) - X / R**2
    cdef double dalpha_dZ = -(BR*dBt_dz - Bt*dBR_dz)/(BR**2+Bt**2) 


    # and finally compute the derivatives of Nparallel, Nperp with respect
    # to R and z
    cdef double dNparallel_dX = dNparallel_dalpha * dalpha_dX + dNparallel_dbeta * dbeta_dX 
    cdef double dNparallel_dY = dNparallel_dalpha * dalpha_dY + dNparallel_dbeta * dbeta_dY 
    cdef double dNparallel_dZ = dNparallel_dalpha * dalpha_dZ + dNparallel_dbeta * dbeta_dz 
    cdef double dNperp_dX = dNperp_dalpha * dalpha_dX + dNperp_dbeta * dbeta_dX
    cdef double dNperp_dY = dNperp_dalpha * dalpha_dY + dNperp_dbeta * dbeta_dY
    cdef double dNperp_dZ = dNperp_dalpha * dalpha_dZ + dNperp_dbeta * dbeta_dz

    return np.array([dNparallel_dX, dNparallel_dY, dNparallel_dZ, dNperp_dX, dNperp_dY, dNperp_dZ],dtype=np.double)
    




##############################################################################
# SOME FUNCTIONS CONSERNING THE DISPERSION RELATION / - MATRIX ARE DEFINED
# BELOW. 
##############################################################################

# fct that returns the Nperp-component such that the refractive index vector
# fullfills the dispersion relation.
# arguments:
# beam omega in rad/s
# magnetic field norm in T
# electron density in 1e-19 / m^3
# Nparallel component of the refractive index vector
# mode sigma
# regularisation parameter for the inversion of S
cpdef double disNperp(double omega, double Bnorm, double Ne, double Nparallel,int sigma,double epsilonRegS):
                                         
    """Calculate the perpendicular wavevector component for a given Nparallel at (R,z) for mode alpha
    such that the dispersion relation is fullfilled.
    For propagation perpendicular to the magnetic field sigma=+1 corresponds to the
    O-mode, sigma=-1 to the X-mode.
    """
    cdef S = disParamS(omega, Bnorm, Ne)
    cdef P = disParamP(omega, Ne)
    cdef D = disParamD(omega, Bnorm, Ne)
    cdef Omega = disParamOmega(Bnorm)

    B = (S+P)*Nparallel**2 \
        - S*P + D**2 - S**2
    C = (S**2-D**2)*P - 2.*S*P*Nparallel**2 + P*Nparallel**4
    SregInv = S / (S**2 + epsilonRegS**2)
    
    return sqrt((-B+sigma*D*sqrt(4.*P*Nparallel**2+Omega**2/omega**2*(Nparallel**2-1.)**2)) / 2. * SregInv)
              









##############################################################################
# DETERMINANT
##############################################################################
cpdef double disDeterminant(double omega, double Bt, double BR, double Bz, double Ne, double Nparallel, double Nperp):
    """Hamiltonian for only one specific mode is defined here.
    This is also what will be used for the ray tracing.
    """
   
    cdef double Bnorm = sqrt(Bt**2 + BR**2 + Bz**2)
    cdef double S = disParamS(omega,Bnorm,Ne)
    cdef double P = disParamP(omega, Ne)
    cdef double D = disParamD(omega,Bnorm,Ne)  
   
    return S*Nperp**4 + ((S+P)*Nparallel**2-S**2-S*P+D**2)*Nperp**2 + P*S**2 - P*D**2 - 2.*P*S*Nparallel**2 + P*Nparallel**4





##############################################################################
# FUNCTIONS CONCERNING THE DISPERSION MATRIX ARE TREATED HERE.
# AS COORDINATES NOW X, Y, Z IS CHOSEN
# THE FUNCTIONS FOR THE DISPERSION MATRIX WITH COORDINATES R, z  
# DEFINED ABOVE ARE USED.
##############################################################################
cpdef double disHamiltonian(double omega, double Bt, double BR, double Bz, double Ne, double Nparallel, double Nperp, double sigma, double epsilonRegS):
    """Hamiltonian for only one specific mode is defined here.
    This is also what will be used for the ray tracing.
    """
   
    cdef double Bnorm = sqrt(Bt**2 + BR**2 + Bz**2)
    cdef double S = disParamS(omega,Bnorm,Ne)
    cdef double P = disParamP(omega,Ne)
    cdef double D = disParamD(omega,Bnorm,Ne)  

    # define parameters A, B, C such that Nperp**2 = -B/(2A) +/- sqrt(B**2/(4A**2) - C/A)
    # according to the dispersion relation det D0 = 0
    cdef double B = (S+P)*Nparallel**2 - S*P + D**2 - S**2 
    cdef double C = S**2*P - D**2*P - 2*S*P*Nparallel**2 + P*Nparallel**4

    # define a regularised S one can divide by
    cdef double SregInv = S / (S**2 + epsilonRegS**2)

      
    return 2*Nperp**2 + B*SregInv - sigma*D*sqrt((Nparallel**2-1.)**2 + 4.*P*Nparallel**2)*SregInv
      
    





# compute the derivatives of the Hamiltonian with respect to X,Y,Z,Nx,Ny,Nz
# arguments: 
# magnetic field components and derivatives,
# electron density and derivatives
# coordinates X,Y,Z
# refractive index vector Nx, Ny, Nz
# mode sigma (+1: o-mode, -1: x-mode)
# regularisation parameter for inversion of S
cpdef np.ndarray [double, ndim=1] disHamiltonianDerivatives(double omega, 
      		 	  	  		         double Bt, double dBt_dR, double dBt_dz,
							 double BR, double dBR_dR, double dBR_dz,
							 double Bz, double dBz_dR, double dBz_dz,
							 double Ne, double dNe_dR, double dNe_dz,
							 X,Y,Z,
							 Nx,Ny,Nz,
							 sigma,
							 epsilonRegS):
          
    """This function returns all first order derivatives of
    Hamiltonian."""      
    cdef double alpha, beta
    alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X , Y, Z)
    cdef np.ndarray [double, ndim=2] rotQ = disrotMatrixQ(alpha, beta)

    cdef double Nperp1 = rotQ.T[0,0]*Nx + rotQ.T[0,1]*Ny + rotQ.T[0,2]*Nz
    cdef double Nperp2 = rotQ.T[1,0]*Nx + rotQ.T[1,1]*Ny + rotQ.T[1,2]*Nz
    cdef double Nparallel = rotQ.T[2,0]*Nx + rotQ.T[2,1]*Ny + rotQ.T[2,2]*Nz
    cdef double Nperp = sqrt(Nperp1**2 + Nperp2**2)
    cdef double Nparallelpower2 = Nparallel**2
    cdef double Nparallelpower4 = Nparallel**4

    cdef double Bnorm = sqrt(BR**2 + Bt**2 + Bz**2)
    # read the plasma parameters at the current position  
    cdef double Omega = disParamOmega(Bnorm)
    cdef double omegaP = disParamomegaP(Ne)
    cdef double S = disParamS(omega, Bnorm, Ne)
    cdef double P = disParamP(omega, Ne)
    cdef double D = disParamD(omega, Bnorm, Ne)

  
    # define the regularised inverse of S (division by S is replaced by multiplication with this quantity)
    cdef double SregInv = S / (S**2 + epsilonRegS**2)


    # define parameters A, B, C such that Nperp**2 = -B/(2A) +/- sqrt(B**2/(4A**2) - C/A)
    # according to the dispersion relation det D0 = 0
    cdef double A = S
    cdef double B = (S+P)*Nparallelpower2 - S*P + D**2 - S**2 
    cdef double C = S**2*P - D**2*P - 2.*S*P*Nparallelpower2 + P*Nparallelpower4



    # deal with the squarroot
    cdef double Delta = ((Nparallelpower2-1.)**2*Omega**2/omega**2 + 4.*P*Nparallelpower2)
    cdef double squarrtDelta = sqrt(Delta)  

    # calculate the derivatives that can be directly calculated 
    # (with respect to Nparallel,Nperp)
    cdef double dH_dNperp = 4.*Nperp
    cdef double dB_dNparallel = 2.*(S+P)*Nparallel
    cdef double dDelta_dNparallel = (8.*P + 4.*Omega**2/omega**2*(Nparallelpower2-1.))*Nparallel
    cdef double dH_dNparallel = SregInv*(dB_dNparallel - sigma*D*dDelta_dNparallel / 2. / squarrtDelta)

    # now calculate the derivatives with respect to X,Y,Z:
    # therefore, first calculate derivatives with respect to R,z
    # dH_dR and dH_dz

    # the same for the el. cyclotron frequency
    # read the magnetic field in order to not evaluate the interpolation function twisw

    cdef double dOmega_dR = echarge / emass / Bnorm \
                  * (Bt*dBt_dR \
                    +BR*dBR_dR \
                    +Bz*dBz_dR)
    cdef double dOmega_dz = echarge / emass / Bnorm \
                  * (Bt*dBt_dz \
                    +BR*dBR_dz \
                    +Bz*dBz_dz)
   
   
    # derivatives of the parameters S, P, D
    cdef double dS_dR = -1./(omega**2 - Omega**2) \
               *(echarge**2/epsilon0/emass*dNe_dR*1e19 \
                 + omegaP**2/(omega**2-Omega**2)*2.*Omega*dOmega_dR)
    cdef double dS_dz = (-1./(omega**2 - Omega**2)) \
               *(echarge**2/epsilon0/emass*dNe_dz*1e19 \
                 + omegaP**2/(omega**2-Omega**2)*2.*Omega*dOmega_dz)	       
    cdef double dP_dR = (-1./omega**2)*echarge**2/epsilon0/emass*dNe_dR*1e19 
    cdef double dP_dz = (-1./omega**2)*echarge**2/epsilon0/emass*dNe_dz*1e19 
    cdef double dD_dR = (1./omega/(omega**2-Omega**2)) \
               *(Omega*echarge**2/epsilon0/emass*dNe_dR*1e19) \
             + omegaP**2/omega/(omega**2-Omega**2)*(omega**2+Omega**2)/(omega**2-Omega**2)*dOmega_dR
    cdef double dD_dz = (1./omega/(omega**2-Omega**2)) \
               *(Omega*echarge**2/epsilon0/emass*dNe_dz*1e19) \
              + omegaP**2/omega/(omega**2-Omega**2)*(omega**2+Omega**2)/(omega**2-Omega**2)*dOmega_dz
    cdef double dB_dR = (dS_dR + dP_dR)*Nparallelpower2  \
            - S*dP_dR - P*dS_dR + 2.*D*dD_dR - 2.*S*dS_dR	    
    cdef double dB_dz = (dS_dz + dP_dz)*Nparallelpower2  \
            - S*dP_dz - P*dS_dz + 2.*D*dD_dz - 2.*S*dS_dz
           
    
    cdef double dDelta_dR = 2.*(Nparallelpower2-1.)**2*Omega*dOmega_dR / omega**2 \
                  + 4.*Nparallelpower2*dP_dR
    cdef double dDelta_dz = 2.*(Nparallelpower2-1.)**2*Omega*dOmega_dz / omega**2 \
                  + 4.*Nparallelpower2*dP_dz
    
      
    # calculate the derivatives of the parallel, perp wavevector components with respect to X,Y,Z
    cdef double dNparallel_dX, dNparallel_dY, dNparallel_dZ, dNperp_dX, dNperp_dY, dNperp_dZ
    dNparallel_dX, dNparallel_dY, dNparallel_dZ, dNperp_dX, dNperp_dY, dNperp_dZ = disNparallelNperpDerivative(Bt, dBt_dR, dBt_dz, BR, dBR_dR, dBR_dz, Bz, dBz_dR, dBz_dz,
    	  	     	      	     X, Y, Z,
				     Nx, Ny, Nz)    


    # and derivatives of the parameters defined above A, B, C (assuming Nparallel to be
    # constant, the derivatives of Nparallel are considered separately)
    cdef double dH_dR = SregInv*(dB_dR - sigma*(D*dDelta_dR / 2. / squarrtDelta \
                                      + dD_dR*squarrtDelta)) \
                             - SregInv**2*dS_dR * (B - sigma*D*squarrtDelta) 
    

    cdef double dH_dz = SregInv*(dB_dz - sigma*(D*dDelta_dz / 2. / squarrtDelta \
                                      + dD_dz*squarrtDelta)) \
                             - SregInv**2*dS_dz * (B - sigma*D*squarrtDelta)
        
    cdef double R = disROutOf(X,Y)

    cdef double dH_dX = X / R * dH_dR + dH_dNperp*dNperp_dX + dH_dNparallel*dNparallel_dX
    cdef double dH_dY = Y / R * dH_dR + dH_dNperp*dNperp_dY + dH_dNparallel*dNparallel_dY
    cdef double dH_dZ =         dH_dz + dH_dNperp*dNperp_dZ + dH_dNparallel*dNparallel_dZ

    cdef double dH_dNx = dH_dNparallel * rotQ.T[2,0] \
              + dH_dNperp*(Nperp1/Nperp*rotQ.T[0,0] + Nperp2/Nperp*rotQ.T[1,0])   
    cdef double dH_dNy = dH_dNparallel * rotQ.T[2,1] \
               + dH_dNperp*(Nperp1/Nperp*rotQ.T[0,1] + Nperp2/Nperp*rotQ.T[1,1])
    cdef double dH_dNz = dH_dNparallel * rotQ.T[2,2] \
               + dH_dNperp*(Nperp1/Nperp*rotQ.T[0,2] + Nperp2/Nperp*rotQ.T[1,2])
       

    return np.array([dH_dX, dH_dY, dH_dZ, dH_dNx, dH_dNy, dH_dNz], dtype=np.double)








# Tr Q div. by SHamiltonian of -sigma (correction factor)
# The result of this function is the correction factor 
# refered to as 1/4 xi(x,N) in the master-thesis.
# To obtain the real xi, a prefactor of 4 is needed.
cpdef double disTrDispersionMatrixDivBySHamiltonian(double omega, 
                                                    double Bnorm, double Ne, 
                                                    double Nparallel, double sigma):
    # define plasma parameters

    cdef double D = disParamD(omega,Bnorm,Ne)
    cdef double P = disParamP(omega,Ne)
    cdef double S = disParamS(omega,Bnorm,Ne)
    cdef double omegaPSq = disParamomegaP(Ne)**2
    cdef double Omega = disParamOmega(Bnorm)
    cdef double OmegaSq = Omega**2
    cdef double omegaSq = omega**2
    cdef double NparallelSq = Nparallel**2
    cdef double sqrrt = sqrt(4.*P*NparallelSq+OmegaSq/omegaSq*(NparallelSq-1.)**2)

# test -
    # if np.isnan(D):
    #     print("D is nan and Ne is ", Ne)
    # if np.isnan(P):
    #     print("P is nan")
    # if np.isnan(S):
    #     print("S is nan")
    # if np.isnan(omegaPSq):
    #     print("omegaPSq is nan")
    # if np.isnan(sqrrt):
    #     print("sqrrt is nan")
#    import sys
#    sys.stdout.flush()
# end test -




    # distinguish in between the two modes
    if sigma == -1.:
        return - ( 1./4./S**2*(Omega/omega*NparallelSq+D+sqrrt)**2*D - D/4.*OmegaSq/omegaSq + sqrrt ) / 2. / sqrrt
    elif sigma == +1.:
        return ( 1./4.*D*( 2.*(OmegaSq/omegaSq-2.)*NparallelSq - Omega/omega*(D+Omega/omega) )**2 / ( Omega/omega*NparallelSq+D+sqrrt )**2 - D/4.*OmegaSq/omegaSq-sqrrt ) / 2. / sqrrt
    else:
        raise ValueError, "sigma is neither -1 nor +1." 
 
