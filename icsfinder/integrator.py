# Third-party dependencies
import numpy as np
import astropy.units as u
from astropy.constants import G
import scipy.linalg as la
from scipy.special import erf
from scipy.signal import argrelextrema

# Gala
import gala.potential as gp
from gala.units import galactic
import gala.dynamics as gd
import gala.integrate as gi
import gala.coordinates as gc


def mwlmc_ics(pos_mw = [0, 0, 0], vel_mw = [0, 0, 0], pos_lmc = [-1.06, -41.05, -27.83], vel_lmc = [-57.60, -225.96, 221.16]):
    """Initialize phase-space positions for Milky Way (MW) and Large Magellanic Cloud (LMC).
    Args:
        Note: for these do not give units, I will internally assign units.
        -> pos_mw: list of 3 elements describing the initial cartesian position vector of the MW (assumed to be in kpc)
        -> vel_mw: list of 3 elements describing the initial cartesian velocity vector of the MW (assumed to be in km/s)
        -> pos_lmc: list of 3 elements describing the initial cartesian position vector of the LMC (assumed to be in kpc)
        -> vel_lmc: list of 3 elements describing the initial cartesian velocity vector of the LMC (assumed to be in kpc)
    Returns:
        tuple: A tuple containing two PhaseSpacePosition objects:
            - wMW: Phase-space position of the Milky Way (at origin with zero velocity). This is an astropy qty with unit of kpc.
            - wLMC: Phase-space position of the LMC with observed position and velocity. This is an astropy qty with unit of km/s.
    """
    wMW = gd.PhaseSpacePosition(pos=pos_mw * u.kpc,
                                vel=vel_mw * u.km / u.s)
    wLMC = gd.PhaseSpacePosition(pos=pos_lmc * u.kpc,
                                vel=vel_lmc * u.km / u.s)
    
    return wMW, wLMC


def host_ln_Lambda(df_params, r):
    """Compute the Coulomb logarithm for dynamical friction calculations.
    
    Implements different dynamical friction (DF) formulations based on satellite mass:
    - For low-mass satellites: Hashimoto et al. 2003 DF (Eq. 5)
    - For massive satellites: van der Marel et al. 2012 DF (Eq. A1)
    
    When L=0, α=1, and C=1.4, the van der Marel DF reduces to the Hashimoto DF.
    
    Typical parameter values:
    - Equal mass mergers: L=0.02, C=0.17, α=0.15
    - 1:10 mass mergers: L=0, C=1.22, α=1.0

    Args:
        df_params (list): Parameters for DF calculation [L, C, a, alpha, CoulombL]
            L (float): Floor value (≥0) to prevent unphysical DF at small separations (r < C*a). This is dimensionless.
            C (float): Scaling factor (>0). This is dimensionless.
            a (float): Satellite halo scale radius (Hernquist profile). This should be an astropy qty with unit of kpc.
            alpha (float): Power-law index (≥0). This is dimensionless.
            CoulombL (str): DF formulation ('Hashimoto' or 'VdM')
        r (float): Current radial position of the satellite [kpc]. This should be an astropy qty with unit of kpc.

    Returns:
        float: Coulomb logarithm value (dimensionless)

    Raises:
        AssertionError: If L < 0 or C ≤ 0

    References:
        - Hashimoto et al. 2003, ApJ, 582, 196
        - van der Marel et al. 2012, ApJ, 753, 9
    """
    L, C, a, alpha, CoulombL = df_params

    assert L >= 0, 'Floor value has to be larger than 0'
    assert C > 0, 'Scaling factor must be positive'
    assert alpha >= 0, 'power index must be positive'
    assert a >= 0, 'halo scale radius must be positive'

    if CoulombL == 'Hashimoto':
        return alpha * np.log(r / (1.4 * a))
    elif CoulombL == 'VdM':
        return np.max([L, np.log((r / (C * a))**alpha)])


def host_sigma(host_mh, host_rh, r):
    """Compute velocity dispersion profile for a Hernquist profile with isotropic velocities. Taken from Hayden Foote.
    Equation 10 of Hernquist 1990.
    Args:
        host_mh (float): Host halo mass [Msun]. This should be an astropy qty with unit of Msun.
        host_rh (float): Scale radius of the host halo [kpc]. This should be an astropy qty with unit of kpc.
        r (float): Radial position where to evaluate dispersion [kpc]. This should be an astropy qty with unit of kpc.

    Returns:
        float: Velocity dispersion at radius r [km/s]. This is an astropy qty with unit of km/s.

    References:
        Hernquist 1990, ApJ, 356:359-364
    """

    a = host_rh
    B = (G*host_mh)/(12*a)
    C = (12*r*((r + a)**3)/(a**4))*np.log((r + a)/r)
    D = (r/(r + a))*(25 + 52*(r/a) + 42*((r/a)**2) + 12*((r/a)**3))

    if(C < D):
        print("Found !")
        print(r, host_mh, a)
        
    return np.sqrt(B*(C - D)).to(u.km/u.s)

def df_acceleration(w, G_gal, **kwargs):
    """Compute dynamical friction acceleration on a satellite galaxy.
    
    Follows the formulation from Patel et al. 2017a (Section 3) and Patel et al. 2020,
    which uses the DF expression from Binney & Tremaine 2008 (Eq. 8.7). Assumes
    a Maxwellian velocity distribution for the host halo particles with dispersion σ.

    Args:
        w (PhaseSpacePosition): Combined phase-space coordinates of host and satellite.
        G_gal (float): Gravitational constant in appropriate units [kpc^3/(Msun Myr^2)]
        **kwargs: Additional parameters required for calculation:
            host_potential (Potential): Potential of the host galaxy
            host_mh (float): Host DM halo mass [Msun]
            host_rh (float): Scale radius of host halo [kpc]
            Msat (float): Satellite mass [Msun]
            host_Lambda_params (list): Parameters for Coulomb logarithm calculation

    Returns:
        ndarray: Dynamical friction acceleration vector [kpc/Myr^2]

    References:
        - Binney & Tremaine 2008, Galactic Dynamics (2nd edition)
        - Patel et al. 2017a
        - Patel et al. 2020
    """
    # read in the phase space
    print('w with units: ', w, '\n')
    w1 = w[:, 1:2]  # satellite
    w2 = w[:, 0:1]  # host 

    # compute relative position and velocity
    w_sat = w1[:3] - w2[:3]
    wv_sat = w1[3:] - w2[3:]

    x = np.ascontiguousarray(w_sat.T)
    v = wv_sat
    
    host_potential = kwargs['host_potential']
    host_mh = kwargs['host_mh']
    host_rh = kwargs['host_rh']
    Msat = kwargs['Msat']
    ln_Lambda_params = kwargs['host_Lambda_params']
    
    dens = host_potential.density(x[0], t=np.array([0.]))[0]
    v_norm = np.sqrt(np.sum(v**2, axis=0))
    r = la.norm(x)
    
    v_disp = host_sigma(host_mh, host_rh, r)
    X = v_norm / (np.sqrt(2) * v_disp)
    fac = erf(X) - 2 * X / np.sqrt(np.pi) * np.exp(-X**2)
    ln_Lambda = host_ln_Lambda(ln_Lambda_params, r)

    dv_dynfric = (-4 * np.pi * G_gal**2 * Msat * dens *
                  ln_Lambda * fac * v) / v_norm**3
    print('Velocity with units: ', v, '\n')
    print('Dynamical friction with units: ', dv_dynfric, '\n')
    return dv_dynfric.value


class Orbit:
    """Class for computing orbits with dynamical friction. Integrates the orbit of a satellite galaxy in a host potential, including dynamical friction effects using a direct N-body approach.
    """

    def __init__(self, host_potential, sat_potential, host_IC, sat_IC, host_mh, host_rh, dt, N, G_gal = G):
        """Initialize orbit integration parameters.
        
        Args:
            host_potential (Potential): Potential of the host galaxy. This is a gala object.
            sat_potential (Potential): Potential of the satellite galaxy. This is a gala object.
            host_IC (PhaseSpacePosition): Initial conditions for host. This is an astropy qty. Units involved are kpc and km/s.
            sat_IC (PhaseSpacePosition): Initial conditions for sat. This is an astropy qty. Units involved are kpc and km/s.
            host_mh (float): Host halo mass [Msun]. This is an astropy qty.
            host_rh (float): Scale radius of host halo [kpc]. This is an astropy qty.
            dt (float): Time step for integration [Gyr]. This is NOT a astropy qty.
            N (int): Number of integration steps
            G_gal (float, optional): Gravitational constant. This is an astropy qty. Give in units of kpc^3/(Msun Myr^2).
        """
        self.host_potential = host_potential #gala object
        self.sat_pot = sat_potential #astropy qty
        self.dt = dt #dimensionless, not a qty
        self.N = N
        self.whost = host_IC #astropy qty, units involved are kpc and km/s
        self.wsat = sat_IC #astropy qty, units involved are kpc and km/s
        self.host_mh = host_mh #astropy qty, unit of Msun
        self.host_rh = host_rh #astropy qty, unit of Msun
        
        print('Integrating orbit for satellite with: \n')
        print('Host ICs are: \n')
        print(host_IC)
        print('Satellites ICs are: \n')        
        print(sat_IC)

        self.w0s = gd.combine((self.whost, self.wsat))
        print('w0s are: ', self.w0s, '\n')
        self.G_gal = G_gal
        
    def sat_orbit(self, df_params):
        """Integrate satellite orbit with dynamical friction.
        
        Args:
            df_params (list): Parameters for dynamical friction calculation [L, C, a, alpha, CoulombL]
                See host_ln_Lambda() for parameter descriptions.

        Returns:
            Orbit: Integrated orbit including dynamical friction effects
        """
        
        def F_MW(t, raw_w, nbody, chandra_kwargs):
            """Compute accelerations including dynamical friction at each timestep.
            
            Args:
                t (float): Current time
                raw_w (ndarray): Current phase-space coordinates
                nbody (DirectNBody): N-body system
                chandra_kwargs (dict): Parameters for DF calculation

            Returns:
                ndarray: Time derivatives of phase-space coordinates
            """
            w = gd.PhaseSpacePosition.from_w(raw_w, units=nbody.units)
            nbody.w0 = w

            wdot = np.zeros((2 * w.ndim, w.shape[0]))
            wdot[3:] = nbody._nbody_acceleration()  # Mutual N-body acceleration
            #print('Just the potential: ', wdot[3:], '\n')
            chandmw = df_acceleration(raw_w, self.G_gal, **chandra_kwargs)
            #print('Dynamical friction: ', chandmw, '\n')
            wdot[3:, 1:] += np.sign(self.dt)*chandmw  # Add DF to satellite
            wdot[:3] = w.v_xyz.decompose(nbody.units).value

            return wdot

        joint_pot = gd.DirectNBody(
            self.w0s,
            particle_potentials=[self.host_potential, self.sat_pot],
            units=galactic)

        chandra_kwargs = {
            'host_potential': self.host_potential,
            'host_mh': self.host_mh,
            'host_rh': self.host_rh,
            'Msat': self.sat_pot.mass_enclosed([10000, 0, 0]),
            'host_Lambda_params': df_params
        }

        integrator = gi.LeapfrogIntegrator(
            F_MW,
            func_args=(joint_pot, chandra_kwargs),
            func_units=joint_pot.units,
            progress=False)

        orbit_MWDF = integrator.run(self.w0s, dt=self.dt * u.Gyr,
                                   n_steps=self.N)

        return orbit_MWDF
