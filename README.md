# ICfinder

ICfinder is designed to find ICs for N-body simulations. The code integrates the backwards orbits of satellites 

### Dynamical Friction Implementation:
- Hashimoto et al. (2003) for low-mass satellites (Eq. 5)
- van der Marel et al. (2012) for massive satellites (Appendix A)

- Velocity dispersion modeled via Zentner & Bullock (2003) NFW profile (Eq. 6)

### Orbit Integration
  - Direct N-body simulation using Gala's leapfrog integrator
  - Handles mutual gravitational interactions for 2 halos

### Flexible Parameter System
  - Tunable Coulomb logarithm parameters
