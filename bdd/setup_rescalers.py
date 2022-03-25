'''Setup: re-scaling and weighting of dispersions.'''

## External modules.
import numpy as np


###############################################################################


## Barron class re-scaling.

_cushion_barron = 0.05 # a cushion for OCE case to ensure minimum exists.
_sigma_thres_barron = 1.0 # threshold for sigma where we switch re-scaling.

def eta_setter_barron(sigma, alpha, interpolate=True, oce_flag=False):
    '''
    Wrapper function for automatically setting the eta value,
    with the Barron class of dispersion functions in mind.
    Here "alpha" is the shape parameter of this class.
    '''
    
    if alpha == 2.0:
        ## In quadratic case, no special settings.
        return 1.0
    elif alpha > 0.0:
        ## For all positive alpha less than 2.0, we can
        ## modulate between x**2 and x**alpha behaviour.
        return eta_setter_barron_modulated(sigma=sigma, alpha=alpha,
                                           interpolate=interpolate,
                                           oce_flag=oce_flag)
    else:
        ## For all non-positive alpha, no special settings.
        return 1.0


def eta_setter_barron_modulated(sigma, alpha, interpolate, oce_flag):
    '''
    A general-purpose function for setting the eta parameter
    in an automatic and principled fashion, based on the scale
    parameter sigma and the type of behaviour desired.
    '''

    ## Condition for existence of minimum.
    if alpha < 1.0 and oce_flag:
        raise ValueError("OCE type is only valid for alpha >= 1.0.")

    ## Condition for interpolation.
    if alpha <= 0.0 and interpolate:
        raise ValueError("Interpolate option only for positive alpha.")

    ## Set the eta value appropriately based on sigma as desired.
    if sigma <= 0.0 or sigma == np.inf:
        raise ValueError("We only handle positive, finite sigma here.")
    
    elif sigma < _sigma_thres_barron:
        
        ## The "small sigma" case.
        
        if interpolate:
            ## If choose to interpolate, the eta value will be set in
            ## such a way that we interpolate between x**2 and x**alpha.
            adiff = np.absolute(alpha-2.0)
            ahalf = alpha/2.0
            eta = alpha * adiff**(ahalf-1.0) * sigma**alpha
        else:
            eta_factor = (1.0+_cushion_barron) if oce_flag else 1.0
            eta = eta_factor * sigma
    else:
        ## The "large sigma" case; tending towards a quadratic.
        eta = sigma**2

    return eta


_cushion_holland = 0.05 # a cushion for OCE case to ensure minimum exists.
_sigma_thres_holland = 1.0 # threshold for sigma where we switch re-scaling.

def eta_setter_holland(sigma, interpolate=True, oce_flag=False):
    '''
    Wrapper function for automatically setting the eta value,
    here with the function used by Holland (2021).
    '''
    if sigma < 0:
        raise ValueError("Only non-negative sigma values allowed.")
    elif sigma == 0:
        eta = (1.0+_cushion_holland) if oce_flag else 1.0
    elif sigma < _sigma_thres_holland:
        if interpolate:
            eta = sigma / (np.pi/2.0) # denominator is np.arctan(np.inf).
        else:
            eta_factor = (1.0+_cushion_holland) if oce_flag else 1.0
            eta = eta_factor*sigma
    elif sigma < np.inf:
        eta = 2.0*sigma**2
    else:
        eta = 1.0
    return eta


## Parsing function for sigma values.
_sigma_lower = 0.0001
_sigma_upper = 1e4
def parse_sigma(sigma):
    '''
    A function which sets a lower/upper thresholds for sigma
    values. Anything below/above these thresholds is automatically
    set to 0.0/np.inf respectively.
    '''
    if sigma is None:
        return None
    elif sigma <= _sigma_lower:
        return 0.0
    elif sigma >= _sigma_upper:
        return np.inf
    else:
        return sigma


###############################################################################
