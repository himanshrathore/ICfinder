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

## Installation 
```
git clone https://github.com/jngaravitoc/ICfinder.git
cd ICfinder
python -m pip install . 
```
## Requirements
```
python>=3.9
numpy>=1.20
scipy>=1.8
astropy>=5.0
gala>=1.7

```

## Virtual environment: 

A virtual environment can be set up as:  

```
python -m venv icsfinder
source cranes-env/bin/activate
cd ~/icsfinder/
python -m pip install .
```
The `icsfind' kernel can be installed in jupyter (after activating the virtual environment) via:

```
pip install ipykernel
ipython kernel install --user --name=icsfinder
```
