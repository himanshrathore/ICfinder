# Third-party dependencies
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import scipy.linalg as la
from scipy.special import erf
from scipy.signal import argrelextrema

# Gala
import gala.potential as gp
from gala.units import galactic
import gala.dynamics as gd
import gala.integrate as gi
import gala.coordinates as gc



def mwlmc_ics():
    wMW =  gd.PhaseSpacePosition(pos=[0,0,0]*u.kpc, vel=[0,0,0]*u.km/u.s)
    wLMC = gd.PhaseSpacePosition(pos=[-1.1, -41.1, -27.9]*u.kpc,
                                 vel=[-57., -226., 221.]*u.km/u.s)
    
    return wMW, wLMC

def host_ln_Lambda(df_params, r):
    """
    DF experienced by the satellite. 
    
    For low-mass satllite's the Hashimoto+03 DF is used
    (See Eq 5 in https://ui.adsabs.harvard.edu/abs/2003ApJ...582..196H/abstract) 
    For massive satellites the van der Marel+2012 DF is used (See Eq A1 and
    appendix A in https://ui.adsabs.harvard.edu/abs/2012ApJ...753....9V/abstract 
    for details).
    
    For equal mass mergers L=0.02, C=0.17, and alpha=0.15 is a good
    approximation
    For 1:10 mergers L=0, C=1.22, and alpha=1.0 

    Note that when L=0, alpha=1, and C=1.4 the VdM DF reduces to the Hashimoto

    Parameters:
    -----------
    df_params: list
        L, C, a, alpha, low_mass
        L : float >= 0
            Floor to prevent the dynamical friction acceleration to become
            unphysical for smaller separations (r < C*a)
        C : float > 0
        a : Satellite scale length (As described in VdM it is for the Hernquist profile)
        alpha : float >= 0.0
    DF.
    r: float
        position of the satellite
    
    
    """
    
    L, C, a, alpha, CoulombL = df_params

    assert L>=0, 'Floor value has to be larger than 0'
    assert C>0, ''

    if CoulombL == 'Hashimoto':
        return alpha*np.log(r/(1.4*a))
    elif CoulombL == 'VdM':
        return np.max([L, np.log(r/(C*a))**alpha])


def host_sigma(host_vmax, host_rs, r):
    """
    Computes the velocity dispersion (sigma) of the host galaxy at a given radius.
    Following   Zetner & Bolluck 2003
    (https://ui.adsabs.harvard.edu/abs/2003ApJ...598...49Z/abstract). See Eq. 6
    Which assumes and NFW halo and an isotropic velocity dispersion. 

    """ 
    x = r / host_rs

    return host_vmax * 1.4393 * (x)**(0.354) / (1. + 1.1756*(x)**(0.725))


def df_acceleration(w, G_gal, **kwargs):
    """
    Following methods from Patel+17a (Section 3), and Patel+20 that uses the DF as 
    defined in Equation 8.7 in Binney & Tremaine 2008, note that this assumes 
    a Maxwellian distribution for the velocities of the host halo with
    dispersion sigma.

    TODO: get gravitational constant from gala and make an input to match the
    simulations.

    Parameters:
    ----------
    w : phase space coordinates of the host and the satelites
    G_gal : Gravitational constant. 
    
    Returns:
    -------

    """
    # read in the phase space
    w1 = w[:,1:2] # sat
    w2 = w[:,0:1] # host 

    #compute relative position and velocity
    w_sat = w1[:3]-w2[:3]
    wv_sat = w1[3:]-w2[3:]

    x = np.ascontiguousarray(w_sat.T)
    v = wv_sat
    
    host_potential = kwargs['host_potential']
    host_vmax = kwargs['host_vmax']
    host_rs = kwargs['host_rs']
    Msat = kwargs['Msat']
    ln_Lambda_params = kwargs['host_Lambda_params']
    
    
    dens = host_potential.density(x[0], t=np.array([0.]))[0]

    v_norm = np.sqrt(np.sum(v**2, axis=0))

    r = la.norm(x)
    
    v_disp = host_sigma(host_vmax, host_rs, r)
            
    X = v_norm / (np.sqrt(2) * v_disp)

    fac = erf(X) - 2*X/np.sqrt(np.pi) * np.exp(-X**2)

    ln_Lambda = host_ln_Lambda(ln_Lambda_params, r)

    dv_dynfric = (- 4*np.pi * G_gal**2 * Msat * dens * ln_Lambda  * fac * v)/ v_norm**3
    
    return dv_dynfric.value

class Orbit:
    def __init__(self, host_potential, sat_potential, host_IC, sat_IC, dt, N, G_gal=4.498502151469554e-12):
        """
        Read parameters for integration 


        """
        
        self.host_potential = host_potential
        self.sat_pot = sat_potential
        self.dt = dt
        self.N = N
        self.whost = host_IC
        self.wsat = sat_IC
        

        print('Integrating orbit for satellite with: \n')
        print('Host ICs are: \n')
        print(host_IC)
        print('Satellites ICs are: \n')        
        print(sat_IC)


        self.w0s = gd.combine((self.whost, self.wsat))

        self.G_gal =  G_gal # 4.498502151469554e-12  #kpc^3/(Msun Myr**2)

        self.host_vmax = np.max(self.host_potential.circular_velocity(np.array([np.linspace(0.1, 300), np.zeros(50), np.zeros(50)])))
        
        self.host_rs = self.host_potential.parameters['halo']['r_s']
        
    def sat_orbit(self, df_params):
        def F_MW(t, raw_w, nbody, chandra_kwargs):
            """
            Computes DF at every timestep
            """
            
            w = gd.PhaseSpacePosition.from_w(raw_w, units=nbody.units)
            nbody.w0 = w

            wdot = np.zeros((2 * w.ndim, w.shape[0]))

            # Compute the mutual N-body acceleration:
            wdot[3:] = nbody._nbody_acceleration()
            #print(wdot)
            # compute DF 
            chandmw = df_acceleration(raw_w, self.G_gal, **chandra_kwargs)
            #print(chandmw)
            wdot[3:, 1:] += chandmw
            wdot[:3] = w.v_xyz.decompose(nbody.units).value

            return wdot


        joint_pot = gd.DirectNBody(self.w0s, particle_potentials=[self.host_potential, self.sat_pot], units=galactic)

        chandra_kwargs = {
        'host_potential': self.host_potential, 
        'host_vmax': self.host_vmax.decompose(galactic).value,
        'host_rs': self.host_rs.value,
        'Msat': self.sat_pot.mass_enclosed([10000, 0, 0]),
        'host_Lambda_params': df_params}

        integrator = gi.LeapfrogIntegrator(
        F_MW, func_args=(joint_pot, chandra_kwargs), 
        func_units=joint_pot.units, 
        progress=False)

        orbit_MWDF = integrator.run(self.w0s, dt=self.dt*u.Gyr, n_steps=self.N)

        return orbit_MWDF
