import numpy as np
from decimal import Decimal as D        # getcontext().prec = 28
from numpy import linalg as LA

# global variables:

# column_dens=np.load('column_dens.npy')
# column_dens = 1e23

# natural units
GeV = 1.0e9
MeV = 1.0e6
keV = 1.0e3

# note: comes from setting hbar*c = 1 = ... m eV
meter = 5.06773093741e6      # = [1/eV] / [m]
cm = 1e-2 * meter            # = [1/eV] / [cm]

# kg = 5.62e35
# gr = 1e-3*kg
Na = 6.0221415e+23

# parsec = 3.0857e16*meter
# kpc = 1.0e3*parsec

# ================ Cross-Sections ===================
# Enu_i: Initial Energy before scattering
# Enu_f: Final Energy after scattering
# theta: 1-x

def theta(Enu_i, Enu_f, mx):
    return mx*(1./Enu_f-1/Enu_i)

def dthetadE(Enu_f, mx):
    return mx/Enu_f**2

def dxsdE_f_scalar(Enu_i, Enu_f, g, mphi, mx):
    th = theta(Enu_i, Enu_f, mx)
    return dthetadE(Enu_f, mx)*g**2/(16*np.pi)*(th*Enu_i**2*mx)/((th*Enu_i+mx)*\
                                                              (th*Enu_i*mphi**2+\
                                                               mx*(mphi**2+2*th*Enu_i**2))**2) #corrected dsign mistake here

def SSHeavyMediator(Enu, g, mphi, mx): #Scalar DM, Scalar mediator
    ## THIS FORMULA IS IN THE LAB FRAME (eV)
    Es=D(Enu)
    dm=D(mx)
    p=D(mphi)
    g=D(g)
    E2=Es**D(2.0)
    m2=dm**D(2.0)
    p2=p**D(2.0)
    g2=g**D(2.0)

    t1=p2*(D(2.0)*Es+dm)
    t2=D(2.0)*Es*p2+D(4.0)*E2*dm+p2*dm
    logs=t1.ln()-t2.ln()

    num=-g2*(D(4.0)*E2*dm+t2*logs)
    den=D(64.0)*D(np.pi)*E2*m2*t2
    sig=num/den

    return float(sig)

# ================ Cascade Equation for Attenuation ===================
def get_RHS_matrices(g, mphi, mx, interaction, energy_nodes):
    NumNodes = energy_nodes.shape[0]
    # auxiliary functions
    if interaction == 'scalar':
        sigma = lambda E: SSHeavyMediator(E, g, mphi, mx)
        DiffXS = lambda Ei,Ef: dxsdE_f_scalar(Ei, Ef, g, mphi, mx) if (Ei > Ef) else 0

    sigma_array = np.array(list(map(sigma,energy_nodes))) # this is for python 3

    # matrices and arrays
    dsigmadE = np.array([[DiffXS(Ei,Ef) for Ei in energy_nodes] for Ef in energy_nodes])
    DeltaE = np.diff(np.log(energy_nodes))
    RHSMatrix = np.zeros((NumNodes,NumNodes))

    # fill in diagonal terms
    for i in range(NumNodes):
        for j in range(i+1,NumNodes):
             # Comparing with NuFate paper: multiply by E_j (= E_in) to account
                # for log scale, then by E_i^2/E_j^2 to account for variable change phi -> E^2*phi

                # DeltaE[j-1] = logE[j] - logE[j-1] = d(logE[j])
            RHSMatrix[i][j] = DeltaE[j-1] * dsigmadE[i,j] * energy_nodes[j]**-1 * energy_nodes[i]**2

    return RHSMatrix, sigma_array

def get_eigs(g, mphi, mx, interaction, energy_nodes, gamma):
    """ Returns the eigenvalues and vectors of matrix M in eqn 6

        Args:
            parameters of DM-nu scenario: the coupling (g), mediator mass (mphi) in eV, and DM mass (mx) in eV
            interaction: interaction between DM and nu
            gamma: power law index of isotropic flux E^-gamma
            logemin: min nu energy log of GeV
            logemax: max nu energy log of GeV

        Returns:
            w: eigenvalues of M matrix in eV
            v: eigenvectors of M matrix in eV
            ci:coefficients in eqn 7
            energy_nodes: neutrino energy in eV
            phi0: initial neutrino flux
        """
    # Note that the solution is scaled by E^2; if you want to modify the incoming spectrum a lot, 
    # you'll need to change this here, as well as in the definition of RHS.
    RHSMatrix, sigma_array = get_RHS_matrices(g, mphi, mx, interaction, energy_nodes)

    # note: this is E^2 * phi_0
    phi_0 = energy_nodes**(2 - gamma)

    w,v = LA.eig(-np.diag(sigma_array)+RHSMatrix)
    ci = LA.lstsq(v,phi_0,rcond=None)[0]
    return w, v, ci


# ================ Attenuated Flux ===================

def get_att_value_theta(w, v, ci, energy_nodes, t):
    
    # w = np.tile(w,[len(energy_nodes),1])
    # phisol = np.inner(v,ci*np.exp(w.T*t).T).T * energy_nodes**(2-gamma) #attenuated flux

    # K: we want to return a vector phi_sol of length = len(energy_nodes).
    # following eq. (7) in https://arxiv.org/pdf/1706.09895.pdf
    # also, the code should match https://github.com/aaronvincent/nuFATE/blob/master/src/python/example.py, lines 22-34

    # phi_j(x) = sum_i c_i * phi_hat_i * exp(lambda_i * x)
    
    # where x = t is the column density
    #       phi_hat_i = the ith eigenvector of M = v[:,i]
    #       lambda_i = the corresponding ith eigenvalue
    #       c_i = the solution to v * x = phi_0, where v is the eigenvector matrix

    # in other words, 
    #   phisol = sum_i( v[:,i] * ci[i] * exp(w[i] * t) ) 

    # note! we do not need an additional multiplication by the flux;
    # we could optionally divide out the initial flux to get the attenuation ratio, phi_sol / phi_0

    phisol = np.dot(v, (ci * np.exp(w * t)))
    return phisol

# return the (interpolated) flux function, flux(E) where E is in GeV. 
# Args: 
#   DM parameters (g, mphi, mx) + interaction type
#   flux parameters (gamma)
#   energy range (in GeV) defined by (logemin, logemax)
#   the column density t 
def attenuated_flux(g, mphi, mx, gamma=3.2, column_dens=1e23, interaction='scalar', logemin=3, logemax=7, NumNodes=120):

    # NumNodes = 5 #120
    # energy_nodes = np.logspace(logemin, logemax, NumNodes)*GeV # in eV

    logE_nodes = np.linspace(logemin, logemax, NumNodes)        # log(E / GeV)
    energy_nodes = np.power(10, logE_nodes)*GeV                         # in eV

    w, v, ci = get_eigs(g, mphi, mx, interaction, energy_nodes, gamma)

    t = column_dens # one value, set globally (top of file)
    flux_astro = get_att_value_theta(w, v, ci, energy_nodes, t)  # in eV

    # interpolate in log-space, to use linear point spacing
    # also, divide out the E^2 (note flux is calc in eV, so need to fix units)
    f_flux = (lambda E : np.interp(np.log10(E), logE_nodes, flux_astro) * 1/E**2 * (GeV)**(gamma-2) )
    return f_flux