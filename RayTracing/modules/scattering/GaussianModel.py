"""This module defines a class providing functions required for the scattering 
strategy in the ray tracing.
"""

# Load standard modules
import math
import numpy as np
import scipy.integrate
# Load local modules
import CommonModules.physics_constants as phys
import RayTracing.modules.random_numbers as rn
import RayTracing.modules.scattering.gaussianmodel_cfunctions as gaussianMod

#################################################################################
# DEFINE A BASE CLASS FROM WHICH CLASSES ARE DERIVED,
# THAT PROVIDES THE RANDOM INITIAL RAY PARAMETERS.
#################################################################################
class GaussianModel_base(object):

    """This class is able to generate new refractive index vectors for 
    scattering using the Metropolis-Hastings-algorithm.

    """
	
    #############################################################################
    # INITIALIZATION OF THE CLASS
    # DEFINITION OF BASIC PROPERTIES AND FUNCTIONS.
    #############################################################################
    def __init__(self, idata, rank):
        
        """Inizialization procedure. Sets the class variables
        to the initial beam parameters.
        """
		
        self.c = phys.SpeedOfLight    
        self.omega = phys.AngularFrequency(idata.freq)   
        self.k0 = phys.WaveNumber(self.omega)
        self.epsilonRegS = idata.epsilonRegS  # regularisation parameter for S^-1

        self.scatteringLengthPerp = idata.scatteringLengthPerp
        self.scatteringLengthParallel = idata.scatteringLengthParallel
        
        # Create a local random number generator
        freeze_seed = idata.freeze_random_numbers
        self.random_state = rn.create_numpy_random_state(freeze_seed, rank)

## test - evaluate locally the parameter xiPerp and xiParallel        
        # # xis used in theory
        # self.scatteringxiPerp = 1./self.scatteringLengthPerp**2
        # self.scatteringxiParallel = 1./self.scatteringLengthParallel**2
## end test -        
        
        # factor for the integration boundaries in use
        self.scatteringintegrationboundaryfactor = idata.scatteringintegrationboundaryfactor

        self.scatteringDeltaneOverne = idata.scatteringDeltaneOverne

        # factor for the max. estimation of the probability guess factor
        self.scatteringMaxProbGuessFactor = idata.scatteringMaxProbGuessFactor
  
        # number of generations for the Metropolis-Hastings algorithm
        self.nmbrinitialisationMetropolisHastingsScattering = idata.nmbrinitialisationMetropolisHastingsScattering

        self.nmbrRays = idata.nmbrRays

## test - xiPerp and xiParallel no longer global variables        
        # also initialise c-functions module
        gaussianMod.scatteringinitialiseGlobalVariables(self.k0,
                                                        self.omega,
                                                        self.epsilonRegS,
                                                        self.nmbrinitialisationMetropolisHastingsScattering,
                                                        freeze_seed, rank)
        ### old version
        # gaussianMod.scatteringinitialiseGlobalVariables(self.k0,
        #                                                 self.omega,
        #                                                 self.epsilonRegS,
        #                                                 self.scatteringxiParallel,
        #                                                 self.scatteringxiPerp,
        #                                                 self.nmbrinitialisationMetropolisHastingsScattering,
        #                                                 freeze_seed, rank)
## end test -        
        

        # return from constructor	
        return


    ################################################################################
    # FUNCTION WHICH GIVES THE NORMALISATION FOR THE EIGENVECTORS
    # (SQUARED AND INVERSED)
    ################################################################################
    def NormalisationEigenvectors(self,
                                  Bnorm, Ne,             # plasma parameters
                                  Nparallel, Nperp):   # refractive index components
                                  
        """Computes the normalisation constant squared for the eigenvectors."""

        return gaussianMod.NormalisationEigenvectors(Bnorm, Ne, Nparallel, Nperp)


## test - This function seems to be redundant: It is never called    
    # ################################################################################
    # # SCATTERING PROBABILITY DISTRIBUTION
    # ################################################################################
    # def ScatteringProbabilityDistributionWithoutShape(self,
    #                                                   Bnorm, Ne,
    #                                                   rho, theta,          # tokamak coordinates
    #                                                   Nparallel, Nperp, phiN, 
    #                                                   Nparallelprime, phiNprime, sigma):   

    #     """Computes the probability distribution for the new refractive index 
    #     components.
    #     See theory sheets 'analytical model for density fluctuations in a tokamak' 
    #     on page 5 for details on computations."""


    #     return gaussianMod.ScatteringProbabilityDistributionWithoutShape(phiNprime, 
    #                                                                      Nparallelprime,
    #                                                                      Bnorm, Ne,
    #                                                                      rho, theta, 
    #                                                                      Nparallel, Nperp,
    #                                                                      phiN, sigma)
## end test -    
        
        
    # put +1 in ToTheSameMode, if the normal scattering is considered,
    # instead, put -1 to consider mode-to-mode scattering
    ################################################################################
    # SCATTERING TOTAL PROBABILITY
    ################################################################################
    def ScatteringProbability(self, Bnorm, Ne, Te, 
                              rho,theta,
                              Nparallel, Nperp, phiN, sigma, ToTheSameMode):

        """Computes the total scattering probability for given refractive index 
        vector as an integral over the probability density given above.
        """

        # evaluate the parallel and perpendicular correlation length and
        # the derived parameters xiParallel and xiPerp
        xiPerp = 1./self.scatteringLengthPerp(rho,theta,Ne,Te,Bnorm)**2
        xiParallel = 1./self.scatteringLengthParallel(rho,theta,Ne,Te,Bnorm)**2        

        # compute boundaries for the numerical integration
        Nparallelmin = Nparallel - math.sqrt(xiParallel)/self.k0 * self.scatteringintegrationboundaryfactor
        if Nparallelmin < -1:
            Nparallelmin = -1.
        Nparallelmax = Nparallel + math.sqrt(xiParallel)/self.k0 * self.scatteringintegrationboundaryfactor
        if Nparallelmax > +1.:
            Nparallelmax = +1.

        try:
            phiNmin = phiN - math.acos(1.-xiPerp/2./self.k0**2/Nperp**2)*self.scatteringintegrationboundaryfactor
            phiNmax = phiN + math.acos(1.-xiPerp/2./self.k0**2/Nperp**2)*self.scatteringintegrationboundaryfactor
        except:
            phiNmin = 0.
            phiNmax = 2.*math.pi

        if ToTheSameMode == +1:
            probWithoutShape = scipy.integrate.dblquad( \
                gaussianMod.ScatteringProbabilityDistributionWithoutShape,
                Nparallelmin, Nparallelmax,
                lambda x: phiNmin, lambda x: phiNmax,
                args=(Bnorm, Ne, rho, theta, Nparallel, Nperp, phiN, sigma, xiParallel, xiPerp) )[0]
        elif ToTheSameMode == -1:
            probWithoutShape = scipy.integrate.dblquad( \
                gaussianMod.ScatteringProbabilityDistributionToDifferentModeWithoutShape,
                Nparallelmin, Nparallelmax,
                lambda x: phiNmin, lambda x: phiNmax,
                args=(Bnorm, Ne, rho, theta, Nparallel, Nperp, phiN, sigma, xiParallel, xiPerp) )[0]
        else:
            print("ToTheSameMode should be +1 or -1.\n")
            raise


        return probWithoutShape * self.scatteringDeltaneOverne(Ne,rho,theta)**2


    ################################################################################
    # ESTIMATE MEAN NUMBER OF SCATTERING EVENTS FOR DIAGNOSTICS PURPOSES
    ################################################################################
    def EstimateMeanNumberOfScatteringEvents(self, 
                                             Bnorm, Ne, Te, 
                                             rho,theta,
                                             f, 
                                             Nparallel, Nperp, phiN, sigma):
                     

## test - Non longer needed
        # echarge = 1.6e-19
        # epsilon0 = 8.85*1e-12 
        # emass = 9.11*1e-31
## end test -        

        # compute probability of scattering
        probscattering = self.ScatteringProbability(Bnorm, Ne, Te, 
                                                    rho,theta,
                                                    Nparallel, Nperp, phiN, sigma, +1)*f*self.timestep
       
        return probscattering


    ################################################################################
    # ESTIMATE MEAN NUMBER OF SCATTERING EVENTS TO THE OTHER MODE
    # FOR DIAGNOSTICS PURPOSES
    ################################################################################
    def EstimateMeanNumberOfModeToModeScatteringEvents(self, 
                                                       Bnorm, Ne, Te, 
                                                       rho,theta,
                                                       f, 
                                                       Nparallel, Nperp, phiN, sigma):
                     

## test - No longer needed
        # echarge = 1.6e-19
        # epsilon0 = 8.85*1e-12 
        # emass = 9.11*1e-31
## end test -        
       
        # compute probability of scattering
        probscattering = self.ScatteringProbability(Bnorm, Ne, Te, 
                                                    rho,theta,
                                                    Nparallel, Nperp, phiN, sigma, -1)*f*self.timestep
       
        return probscattering


#
# End of class




#################################################################################
# DEFINE A DERIVED CLASS FOR SINGLE MODE SCATTERING.
#################################################################################
class GaussianModel_SingleMode(GaussianModel_base):

    """This class performs scattering in the case of a single mode
    """
	

    #############################################################################
    # FUNCTION WHICH DECIDES WHEATER A SCATTERING EVENT OCCURS
    # takes the recent ray phase-space position as an argument
    #############################################################################
    def DecideScattering(self, 
                         Bnorm, Ne, Te, 
                         rho,theta,
                         f, 
                         Nparallel, Nperp, phiN, sigma):
                     

        # scattering correlation lengths and related parameters
        xiPerp = 1./self.scatteringLengthPerp(rho,theta,Ne,Te,Bnorm)**2
        xiParallel = 1./self.scatteringLengthParallel(rho,theta,Ne,Te,Bnorm)**2        
        
        # initialise counter counting the scattering kicks
        counter = 0

        # copy the initial refractive index vector
        oldNparallel = Nparallel
        oldNperp = Nperp
        oldphiN = phiN

        echarge = 1.6e-19
        epsilon0 = 8.85*1e-12 
        emass = 9.11*1e-31

        dP_dne = -1./(self.omega**2)*echarge**2/epsilon0/emass*1e19

        # first see if an event has happend
        # compute estimation for scattering probability
        probevent = self.k0**2 * Ne**2 / 4. * math.sqrt(2.*math.pi) / Nperp \
            / math.sqrt(xiPerp) * (dP_dne)**2 * f**2 * self.timestep * self.scatteringMaxProbGuessFactor * self.scatteringDeltaneOverne(Ne,rho,theta)**2

        # the estimate given above is not appropriate in case abs(Nperp) << 1
        # an upper boundary in general is given by
        probeventUpperBound = math.pi / 2. * self.k0**3 * Ne**2 * (dP_dne)**2 / xiPerp \
            * f**2 * self.timestep * self.scatteringMaxProbGuessFactor * self.scatteringDeltaneOverne(Ne,rho,theta)**2
        if probevent > probeventUpperBound:
            probevent = probeventUpperBound
        
        # generate number of events using a Poisson distribution
        NumberOfEvents = self.random_state.poisson(probevent)

        # if only two rays are traced, print information on scattering events.
        # SIGMA is the \Sigma in the theory.
        if self.nmbrRays == 2 and probevent > 0.01:
            print('rho=%f: param. of the Poisson process is %f (factor: %f), %i events generated.\n' %(rho,probevent,self.scatteringMaxProbGuessFactor,NumberOfEvents))

        # for each event, see if scattering occurs.
        for i in range(0,NumberOfEvents):
            
            # if an event might happen compute the correct probability for scattering
            # NB in this modified version "probscattering" takes into account also
            # the mode to mode scattering.
            SIGMA_in_mode = self.ScatteringProbability(Bnorm, Ne, Te,
                                                       rho,theta,
                                                       Nparallel, Nperp, phiN, sigma, +1)*f*self.timestep

            total_cross_section = SIGMA_in_mode

            # normalise probability with the one for gerenation of an event
            probscattering = total_cross_section / probevent

            if self.nmbrRays == 2:
                print('--> Event %i\n'%(i))
                print('    total cross section to the same mode times dt: %f\n'
                      %(total_cross_section))
                print('    estimated total cross section to the same mode times dt: %f\n'
                      %(probevent))
                print('    estimated upper bound for total cross section to the same mode times dt: %f\n'
                      %(probeventUpperBound))
                print('    prob. for scattering to the same mode: %f\n'
                      %(probscattering))

            if probscattering > 1.:
                msg = """WARNING: PROBABILITY OF SCATTERING EXCEEDS ONE. 
                PLEASE TAKE CARE WHEN CHOOSING THE PARAMETERS. 
                THE SCATTERING PROBABILITY WILL BE UNDERESTIMATED.\n"""
                print(msg)
                print('    Current radial position rho = %f\n'%(rho))
                print('    Total cross section times dt: %f\n'
                      %(total_cross_section))
                print('    Estimated total cross section to the same mode times dt: %f\n'
                      %(probevent))
                print('    Estimated upper bound for total cross section times dt: %f\n'
                      %(probeventUpperBound))
                print('    Prob. for scattering to the same mode: %f\n'
                      %(SIGMA_in_mode))

            # see, if event is also a scattering kick
            if probscattering > self.random_state.uniform(0.,1.):
                                    
                # if yes choose new refractive index
                target_sigma = sigma # target mode = same mode
                Nparallel, Nperp, phiN = gaussianMod.ScatteringChooseRefractiveIndex(
                    Nparallel, Nperp, phiN,
                    Bnorm, Ne,
                    rho, theta, sigma, target_sigma,
                    xiParallel, xiPerp)
                self.newNparallel = Nparallel
                self.newNperp = Nperp
                self.newphiN = phiN
                self.newMode = target_sigma
                counter +=1
    
    
        # if scattering has occured
        if counter > 0:
            # if only two rays are traced, print information on scattering events
            if self.nmbrRays == 2:
                print('%i scattering kicks: Nparallel: %f -> %f, Nperp: %f -> %f, phiN: %f -> %f\n' %(counter,
                                                                                                      oldNparallel,
                                                                                                      Nparallel,
                                                                                                      oldNperp,
                                                                                                      Nperp,
                                                                                                      oldphiN,
                                                                                                      phiN))
                print('max. guess without factor: %f, correct scattering probability: %f\n' %(probevent / self.scatteringMaxProbGuessFactor,
                                                                                              probscattering * probevent))

            # and return True to indicate that scattering has occured
            return True   
        else:
            # if no scattering has occured, return False
            return False

#
# End of class





#################################################################################
# DEFINE A DERIVED CLASS FOR CROSS POLARIZATION SCATTERING.
#################################################################################
class GaussianModel_MultiMode(GaussianModel_base):

    """This class takes into account cross-polarization scattering
    """

    #############################################################################
    # FUNCTION WHICH DECIDES WHEATER A SCATTERING EVENT OCCURS
    # takes the recent ray phase-space position as an argument
    #############################################################################
    def DecideScattering(self, 
                         Bnorm, Ne, Te, 
                         rho,theta,
                         f, 
                         Nparallel, Nperp, phiN, sigma):
                     

        # scattering correlation lengths and related parameters
        xiPerp = 1./self.scatteringLengthPerp(rho,theta,Ne,Te,Bnorm)**2
        xiParallel = 1./self.scatteringLengthParallel(rho,theta,Ne,Te,Bnorm)**2
        
        # initialise counter counting the scattering kicks
        counter = 0

        # copy the initial refractive index vector
        oldNparallel = Nparallel
        oldNperp = Nperp
        oldphiN = phiN
        oldMode = sigma

        echarge = 1.6e-19
        epsilon0 = 8.85*1e-12 
        emass = 9.11*1e-31

        dP_dne = -1./(self.omega**2)*echarge**2/epsilon0/emass*1e19

        # first see if an event has happend
        # compute estimation for scattering probability
        probevent = self.k0**2 * Ne**2 / 4. * math.sqrt(2.*math.pi) / Nperp \
            / math.sqrt(xiPerp) * (dP_dne)**2 * f**2 * self.timestep * self.scatteringMaxProbGuessFactor * self.scatteringDeltaneOverne(Ne,rho,theta)**2

        # the estimate given above is not appropriate in case abs(Nperp) << 1
        # an upper boundary in general is given by
        probeventUpperBound = math.pi / 2. * self.k0**3 * Ne**2 * (dP_dne)**2 / xiPerp \
            * f**2 * self.timestep * self.scatteringMaxProbGuessFactor * self.scatteringDeltaneOverne(Ne,rho,theta)**2
        if probevent > probeventUpperBound:
            probevent = probeventUpperBound
        
        # generate number of events using a Poisson distribution
        NumberOfEvents = self.random_state.poisson(probevent)
        

        # if only two rays are traced, print information on scattering events
        if self.nmbrRays == 2 and probevent > 0.01:
            print('rho=%f: param. of the Poisson process is %f (factor: %f), %i events generated.\n' %(rho,probevent,self.scatteringMaxProbGuessFactor,NumberOfEvents))

        # for each event, see if scattering occurs.
        for i in range(0,NumberOfEvents):
            
            # if an event might happen compute the correct probability for scattering
            # NB in this modified version "probscattering" takes into account also
            # the mode to mode scattering.
            SIGMA_in_mode = self.ScatteringProbability(Bnorm, Ne, Te, 
                                                       rho,theta,
                                                       Nparallel, Nperp, phiN, sigma, +1)*f*self.timestep
            SIGMA_mode_to_mode = self.ScatteringProbability(Bnorm, Ne, Te, 
                                                            rho,theta,
                                                            Nparallel, Nperp, phiN, sigma, -1)*f*self.timestep                                            
            total_cross_section = SIGMA_in_mode + SIGMA_mode_to_mode

            # normalise probability with the one for gerenation of an event
            probscattering = total_cross_section / probevent

            if self.nmbrRays == 2:
                print('--> Event %i\n'%(i))
                print('    total cross section times dt: %f\n'
                      %(total_cross_section))
                print('    estimated total cross section to the same mode times dt: %f\n'
                      %(probevent))
                print('    estimated upper bound for total cross section times dt: %f\n'
                      %(probeventUpperBound))
                print('    prob. for scattering to the same mode: %f\n'
                      %(SIGMA_in_mode))
                print('    prob. for scattering to the other mode: %f\n'
                      %(SIGMA_mode_to_mode))

            if probscattering > 1.:
                msg = """WARNING: PROBABILITY OF SCATTERING EXCEEDS ONE. 
                PLEASE TAKE CARE WHEN CHOOSING THE PARAMETERS. 
                THE SCATTERING PROBABILITY WILL BE UNDERESTIMATED.\n"""
                print(msg)
                print('    Current radial position rho = %f\n'%(rho))
                print('    Total cross section times dt: %f\n'
                      %(total_cross_section))
                print('    Estimated total cross section to the same mode times dt: %f\n'
                      %(probevent))
                print('    Estimated upper bound for total cross section times dt: %f\n'
                      %(probeventUpperBound))
                print('    Prob. for scattering to the same mode: %f\n'
                      %(SIGMA_in_mode))
                print('    Prob. for scattering to the other mode: %f\n'
                      %(SIGMA_mode_to_mode))

            # see, if event is also a scattering kick
            if probscattering > self.random_state.uniform(0.,1.):

                counter += 1
                
                # probability that the mode doesn't change
                prob_in_mode = SIGMA_in_mode / total_cross_section

                # if scattered into the same mode...
                if prob_in_mode > self.random_state.uniform(0.,1.):
                    # if yes choose new refractive index
                    target_sigma = sigma # target mode = same mode
                    Nparallel, Nperp, phiN = gaussianMod.ScatteringChooseRefractiveIndex(
                        Nparallel, Nperp, phiN,
                        Bnorm, Ne,
                        rho, theta, sigma, target_sigma,
                        xiParallel, xiPerp)
                    self.newNparallel = Nparallel
                    self.newNperp = Nperp
                    self.newphiN = phiN
                    self.newMode = target_sigma
                else:
                    # ...
                    target_sigma = -sigma # target mode = different mode
                    Nparallel, Nperp, phiN = gaussianMod.ScatteringChooseRefractiveIndex(
                        Nparallel, Nperp, phiN,
                        Bnorm, Ne,
                        rho, theta, sigma, target_sigma,
                        xiParallel, xiPerp)
                    self.newNparallel = Nparallel
                    self.newNperp = Nperp
                    self.newphiN = phiN
                    self.newMode = target_sigma

    
        # if scattering has occured
        if counter > 0:
            # if only two rays are traced, print information on scattering events
            if self.nmbrRays == 2:
                print('%i scattering kicks: Nparallel: %f -> %f, Nperp: %f -> %f, phiN: %f -> %f, sigma: %f -> %f\n' %(counter,
                                                                                                                       oldNparallel,
                                                                                                                       Nparallel,
                                                                                                                       oldNperp,
                                                                                                                       Nperp,
                                                                                                                       oldphiN,
                                                                                                                       phiN,
                                                                                                                       oldMode,
                                                                                                                       self.newMode))
                print('max. guess without factor: %f, correct scattering probability: %f\n' %(probevent / self.scatteringMaxProbGuessFactor,
                                                                                              probscattering * probevent))

            # and return True to indicate that scattering has occured
            return True   
        else:
            # if no scattering has occured, return False
            return False


#
# End of class


