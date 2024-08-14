"""
Define objects representing the standard Gaussian beam solution.
This is meant for testing and post pocessing of results.
"""

# Import statements
import numpy as np


# Base class, essentially defining only common parameters
class GaussianBeam_base(object):
    
    """
    Base class for standard Gaussian beams.
    Two-dimensional and three-dimensional beams are derived from this 
    common parent object.
    """
    
    # Constructor method
    def __init__(self, k0, w0, a0):
        
        # Just store the parameter of the beam
        self.k0 = k0   # wave number in free space
        self.w0 = w0   # beam width at the waist
        self.a0 = a0   # amplitude at the waist
        
        # Derived parameters
        self.zR = 0.5 * k0 * w0**2  #Rayleigh distance

        return
        
    # Evaluate the beam width as a function of position along beam axis
    def w(self, z):
        
        """
        Standard formula for the width of a Gaussian beam:
        
          w(z) = w0 * sqrt(1. + (z/zR)**2)
        
        where z is the position along the beam axis and zR is the Rayleigh
        distance.
        """
        
        return self.w0 * np.sqrt(1. + (z / self.zR)**2)
        
    # Evaluate the curvature of the phase-front 
    # (Not the radius of curvature due the the well known singularity at z=0)
    def K(self, z):
        
        """
        Evaluate the curvature of the phase fronts as a function of the 
        position z along the beam axis. We have
        
          K(z) = 1. / R(z),
        
        where the radius of curvature is
        
          R(z) = z + zR**2 / z.
        
        Computing directly the curvatire avoid the singularity at z=0.
        Here zR is the Rayleigh distance.
        """

        return z / (z**2 + self.zR**2)
        
# End of class



# Standard Gaussian beam in two dimensions (x,z) propagating along z
class GaussianBeam2D(GaussianBeam_base):
    
    """
    Standard Gaussian beam in two dimensions. In the (x,z) plane,
    the propagation direction is along the z-axis and the waist of the
    beam is located in the origin (0,0).
    """
    
    # Evaluate the complex wave field
    def field(self, x, z):
        
        """
        Evaluate the complex wave field of the beam at position (x,z).
        """
        
        ii = complex(0.0, 1.0)
        w = self.w(z)
        K = self.K(z)
        a = self.a0 * np.sqrt(self.w0 / w) * np.exp(-x**2/w**2)
        Gouy_shift = np.arctan(z / self.zR)
        phase = self.k0*z + 0.5 * self.k0 * K * x**2 + Gouy_shift

        return a * np.exp(ii*phase)
        
    # Evaluate the energy density of the beam (squared modulus of the field)
    # (It seems more efficient to build the energy density directly rather 
    #  then calling self.field(x,z) and taking the squared modulus.)
    def energy(self, x, z):
        
        """
        Evaluate the sqrared modulus of the wave field at position (x,z).
        """
        
        w = self.w(z)
        energy = self.a0**2 * (self.w0 / w) * np.exp(-2.0*x**2/w**2)

        return energy
        
    # Evaluate the energy flux vector of the beam
    def flux(self, x, z):
        
        """
        Evaluate the wave energy flux
        
         F = [fx, fz], 
        
        where
        
         Fx = (x * k0 / R) * energy
         Fx = k0 * energy
        
        and, in particular, K = 1/R is the curvature of the phase front.
        """
        
        k0u2 = self.k0 * self.energy(x, z)
        K = self.K(z)

        F = np.array([x*K*k0u2, k0u2])
    
        return F 

# End of class



# Standard Gaussian beam in three dimensions (x,y,z) propagating along z
class GaussianBeam3D(GaussianBeam_base):
    
    """
    The same as GaussianBeam2D but with all physical dimensions.
    This essentially changes the scaling of the amplitude with the 
    beam width.
    """
    
    # Evaluate the complex wave field
    def field(self, x, y, z):
        
        """
        Evaluate the complex wave field of the beam at position (x,z).
        """
        
        ii = complex(0.0, 1.0)
        r2 = x**2 + y**2

        w = self.w(z)
        K = self.K(z)
        a = self.a0 * (self.w0 / w) * np.exp(-r2/w**2)
        Gouy_shift = np.arctan(z / self.zR)
        phase = self.k0*z + 0.5 * self.k0 * K * x**2 + Gouy_shift

        return a * np.exp(ii*phase)
        
    # Evaluate the energy density of the beam (squared modulus of the field)
    # (It seems more efficient to build the energy density directly rather 
    #  then calling self.field(x,z) and taking the squared modulus.)
    def energy(self, x, y, z):
        
        """
        Evaluate the sqrared modulus of the wave field at position (x,z).
        """
        r2 = x**2 + y**2
        w = self.w(z)
        energy = self.a0**2 * (self.w0 / w)**2 * np.exp(-2.0*r2/w**2)

        return energy
        
    # Evaluate the energy flux vector of the beam
    def flux(self, x, y, z):
        
        """
        Evaluate the wave energy flux
        
         F = [fx, fz], 
        
        where
        
         Fx = (x * k0 / R) * energy
         Fx = k0 * energy
        
        and, in particular, K = 1/R is the curvature of the phase front.
        """
        
        k0u2 = self.k0 * self.energy(x, y, z)
        K = self.K(z)

        F = np.array([x*K*k0u2, y*K*k0u2, k0u2])
    
        return F 

# End of class


# Testing the beams
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    # Common parameters and construction of beam objects
    k0 = 10.0
    w0 = 0.3
    a0 = 1.0
    beam2d = GaussianBeam2D(k0=k0, w0=w0, a0=a0)
    beam3d = GaussianBeam3D(k0=k0, w0=w0, a0=a0)
    
    # Field of a two-dimensional beam
    x = np.linspace(-2.0, +2.0, 200)
    z = np.linspace(-2.5, +2.5, 500)
    X, Z = np.meshgrid(x, z)
    u2d = beam2d.field(X, Z).real
    w = beam2d.w(Z[:,0])

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, aspect='equal')
    mesh1 = ax1.pcolormesh(X, Z, u2d, cmap='coolwarm')
    plt.colorbar(mesh1)
    ax1.plot(w, Z[:,0], 'k')
    ax1.plot(-w, Z[:,0], 'k')
    ax1.set_xlabel('$x$', fontsize=20)
    ax1.set_ylabel('$z$', fontsize=20)

    plt.show()

# End of file
