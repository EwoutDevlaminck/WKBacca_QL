"""This module defines the class MetropolisHastingsBound. It is able to generate sets of initial 
ray parameters using the Metropolis-Hastings-Algorithm.
The beam parameters are needed.
"""

# load standard modules
import numpy as np
import math
import scipy.integrate
# load local modules
import CommonModules.physics_constants as phys


################################################################################
# DEFINE A CLASS THAT PROVIDES THE RANDOM INITIAL RAY PARAMETERS.
################################################################################
class RayInit(object):

    """This class is able to generate the initial ray parameters
    using the Metropolis-Hastings-algorithm.
    It uses the beam parameters.
    Before using the generated parameters, set reasonable starting values
    and call the initialize routine which lets run the random number generaton
    for the indicated number of times. This is needed to let the 
    probability distribution converge to the desired one.

    """
	
    ############################################################################
    # INITIALIZATION OF THE CLASS
    # DEFINITION OF BASIC PROPERTIES AND FUNCTIONS.
    ############################################################################
    def __init__(self, idata, random_state):
        
        """Inizialization procedure. Sets the class variables
        to the initial beam parameters.
        """

        # store the random number generator
        self.random = random_state
        
        # physics constants
        self.c = phys.SpeedOfLight   
        self.omega = phys.AngularFrequency(idata.freq)
        self.k0 = phys.WaveNumber(self.omega) 

        # eigenvalues of matrix phi and S as written down in theory
        self.eigenvalueD1 = phys.EigenvalueD(self.k0, idata.beamwidth1)
        self.eigenvalueD2 = phys.EigenvalueD(self.k0, idata.beamwidth2)
        self.eigenvalueS1 = phys.EigenvalueS(idata.curvatureradius1)
        self.eigenvalueS2 = phys.EigenvalueS(idata.curvatureradius2)

        # central spectrum
        self.centraleta1 = idata.centraleta1
        self.centraleta2 = idata.centraleta2

        # see if the rays have to be launched two dimensional ore three dimensional
        self.twodim = idata.twodim
        
        # different probability distribution for lense like medium (valley shaped potential)
        if idata.equilibrium == 'Model' and idata.valley == True:
            self.LenseLikeMedium = True
            self.linearlayervalleyL = idata.linearlayervalleyL
        else:
            self.LenseLikeMedium = False
        
        # return from constructor	
        return

        
    #############################################################################
    # SET THE SET OF RECENT RAY PARAMETERS TO AN INITIAL VALUE
    #############################################################################
    def SetInitialValue(self, y1, y2, eta1, eta2):

        """Set the ray parameters to an initial value.
        For the two dimensional case, only y1 and eta1 are needed.
        """
        # set the variables in the Metropolis-Hastings object.
        self.y1 = y1
        self.y2 = y2
        self.eta1 = eta1
        self.eta2 = eta2

        # if only two dimensions are traced, set the other variables to zero.
        if self.twodim == True:
            self.y2 = 0.
            self.eta2 = 0.

        return 


    ################################################################################
    # DO THE INITIALISATION AND THEREFORE LET RUN THE RANDOM NUMBER GENERATOR
    # FOR THE INDICATED NUMBER OF TIMES
    ################################################################################
    def InitialiseMHAlg(self, nmbrinit):

        """Do nmbrinit steps using the Metropolis-Hastings algorithm.
        Needed to let the distribution converge to the desired one.
        """

        for i in range(0,nmbrinit):
            self.GenerateRandom()

        return 


    ################################################################################
    # DEFINE THE NORMALISATION FACTOR SUCH THAT THE NORMALISED PROBABILITY 
    # FUNCTION MULTIPLIED WITH THIS FACTOR AND INTEGRATED GIVES THE SAME AS
    # THE WIGNER FUNCTION INTEGRATED (SEE THEORY SHEET FullMCProcess FOR MORE
    # EXPLANATION)
    ################################################################################
    def GenerateNormalisation(self):
        # for the lense like medium an other normalisation factor is needed
        if self.LenseLikeMedium == True:
            if self.twodim == False:
                print("THE LENSE LIKE MEDIUM ONLY CAN BE TREATED IN TWO DIMENSIONS.\n")
                raise
            
            # define 2 * Nx * Wfct to integrate
            normfacttemp = lambda eta, y: 2.*math.sqrt(self.k0/self.eigenvalueD1/math.pi) \
                * math.sqrt(1.-eta**2-y**2/self.linearlayervalleyL**2) \
                * math.exp(-self.k0*self.eigenvalueD1*y**2 \
                                -self.k0/self.eigenvalueD1*(eta+self.eigenvalueS1*y)**2)
          
       
            # and perform the integration using numerical quadrature.
            normfact = scipy.integrate.dblquad(
                normfacttemp,                                               #fct to integrate
                -self.linearlayervalleyL,                                   # boundaries for Y
                +self.linearlayervalleyL,      
                lambda x: -math.sqrt(1.-x**2/self.linearlayervalleyL**2),   # boundaries for eta 
                lambda x: +math.sqrt(1.-x**2/self.linearlayervalleyL**2),             
                args=())[0]

        # for all the other cases the antenna is considered to be in free space, 
        # therefore the free space calculation is sufficient
        else:
            # for the two dimensional case
            if self.twodim == True:
                # again, define 2 * Nx * Wfct, but already integrated analytically over y
                normfacttemp = lambda eta1: 2.*math.sqrt(1.-eta1**2) \
                    / math.sqrt(self.eigenvalueD1**2+self.eigenvalueS1**2) \
                    * math.exp(-self.k0*self.eigenvalueD1 \
                                    /(self.eigenvalueD1**2+self.eigenvalueS1**2) \
                                    *(eta1-self.centraleta1)**2)

                # and perform the remaining eta-integration.
                normfact = scipy.integrate.quad(normfacttemp,-1.,+1., limit=1000, args=())[0]

            # and for the three dimensional case
            else:
                # define 2 * Nx * Wfct, already integrated analytically over y, z
                normfacttemp = lambda eta2, eta1: 2.*math.sqrt(1.-eta1**2-eta2**2) \
                    / math.sqrt(self.eigenvalueD1**2+self.eigenvalueS1**2) \
                    / math.sqrt(self.eigenvalueD2**2+self.eigenvalueS2**2) \
                    * math.exp(-self.k0*self.eigenvalueD1 \
                                    /(self.eigenvalueD1**2+self.eigenvalueS1**2) \
                                    *(eta1-self.centraleta1)**2) \
                    * math.exp(-self.k0*self.eigenvalueD2 \
                                    /(self.eigenvalueD2**2+self.eigenvalueS2**2) \
                                    *(eta2-self.centraleta2)**2)
                
                # and integrate over eta1, eta2
                normfact = scipy.integrate.dblquad(
                    normfacttemp,                      #fct to integrate
                    -1.,                               # boundaries for eta1
                    +1.,      
                    lambda x: -math.sqrt(1.-x**2),     # boundaries for eta2 
                    lambda x: +math.sqrt(1.-x**2),             
                    args=())[0]

                
            
        # return the normalisation factor
        return normfact


    ################################################################################
    # GENERATE AND RETURN A NEW SET OF RAY PARAMETERS
    ################################################################################
    def GenerateRandom(self):

        """Generate a new set of ray parameters and return them.
        No random numbers with norm(eta) > 1 will be generated.
        """

        # if there is no chance to fullfill the dispersion relation with 
        # a chosen set of coordinates, choose an other one
        notyetsucceeded = True
        

        while notyetsucceeded:
            Y1 = self.random.normal(0.,1./math.sqrt(2.*self.k0*self.eigenvalueD1))
            Y2 = self.random.normal(0.,1./math.sqrt(2.*self.k0*self.eigenvalueD2))
            _eta1 = self.random.normal(0.,math.sqrt(self.eigenvalueD1/2./self.k0))
            _eta2 = self.random.normal(0.,math.sqrt(self.eigenvalueD2/2./self.k0))
                
            eta1 = _eta1 - self.eigenvalueS1*Y1 + self.centraleta1
            eta2 = _eta2 - self.eigenvalueS2*Y2 + self.centraleta2

            if self.twodim == True:
                eta2 = 0.
                _eta2 = 0.
                Y2 = 0.

            # and now see if this new set of parameters is accepted or not 
            # use different probability for lense like medium
            # if the calculation fails it is because the square root cannot be
            # computed. In this case an other set of coordinates must be chosen.
            try:
                if self.LenseLikeMedium == False:
                    probaccept = math.sqrt((1.-eta1**2-eta2**2)/(1.-self.eta1**2-self.eta2**2))
                else:
                    probaccept = math.sqrt((1.-eta1**2-Y1**2/self.linearlayervalleyL**2) \
                                               / (1.-self.eta1**2-self.y1**2/self.linearlayervalleyL**2))
                notyetsucceeded = False
            except:
                notyetsucceeded = True


        # if probaccept >= 1, accept in any case
        # otherwise, accept with probability probaccept
        # generate uniform random number in between (0,1)
        acceptuniform = self.random.uniform(0.,1.)
        if probaccept > acceptuniform:
            self.eta1 = eta1
            self.eta2 = eta2
            self.y1 = Y1
            self.y2 = Y2
    
        return self.y1, self.y2, self.eta1, self.eta2

    ################################################################################
    # RETURN CURRENT SET OF RAY PARAMETERS
    ################################################################################
    def ReturnRandom(self):

        """Return the current ray parameters.
        """
 
        return self.y1, self.y2, self.eta1, self.eta2

    

  

#
# End of class MetropolisHastingsBound
