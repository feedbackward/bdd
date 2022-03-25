'''Setup: dispersion functions.'''

## External modules.
import numpy as np

## Internal modules.
from setup_rescalers import eta_setter_barron, eta_setter_holland


###############################################################################


## Dispersion function definitions.

def dispersion_barron(x, alpha):
    '''
    This is the Barron-type function rho(x;alpha).
    '''

    if alpha == 2.0:
        return x**2/2.0
        
    elif alpha == 0.0:
        return np.log1p(x**2/2.0)
    
    elif alpha == np.NINF:
        return 1.0 - np.exp(-x**2/2.0)
        
    else:
        adiff = np.absolute(alpha-2.0)
        ahalf = alpha/2.0
        return (adiff/alpha) * ((1.0+x**2/adiff)**ahalf - 1.0)


def d1_barron(x, alpha):
    '''
    Returns the first derivative of dispersion_barron,
    taken with respect to the first argument.
    '''

    if alpha == 2.0:
        return x
        
    elif alpha == 0.0:
        return x / (1.0 + x**2/2.0)
    
    elif alpha == np.NINF:
        return np.exp(-x**2/2.0) * x
        
    else:
        adiff = np.absolute(alpha-2.0)
        ahalf = alpha/2.0
        return x * (1.0+x**2/adiff)**(ahalf-1.0)


def d2_barron(x, alpha):
    '''
    Returns the second derivative of dispersion_barron,
    taken with respect to the first argument.
    '''

    if alpha == 2.0:
        return np.ones_like(a=x)
        
    elif alpha == 0.0:
        return 1.0/(1.0+x**2/2.0) - (x/(1.0+x**2/2.0))**2
    
    elif alpha == np.NINF:
        return np.exp(-x**2/2.0) * (1.0 - x**2)
        
    else:
        adiff = np.absolute(alpha-2.0)
        ahalf = alpha/2.0
        innards = 1.0+x**2/adiff
        term1 = innards**(ahalf-1.0)
        term2 = (x**2) * (ahalf-1.0) * innards**(ahalf-2.0) * (2/adiff)
        return term1 + term2


def dispersion_huber(x, beta):
    '''
    This is the penalized Huber function rho(x;beta).
    '''
    return beta + (np.sqrt(1.0+x**2)-1.0)/beta


def d1_huber(x, beta):
    '''
    Returns the first derivative of dispersion_huber,
    taken with respect to the first argument.
    '''
    return x / (beta * np.sqrt(1.0+x**2))


def d2_huber(x, beta):
    '''
    Returns the second derivative of dispersion_huber,
    taken with respect to the first argument.
    '''
    return 1.0 / (beta * (1.0+x**2)**(1.5))


def dispersion_holland(x):
    '''
    This is the dispersion function rho(x) used by Holland (2021).
    '''
    return x * np.arctan(x) - np.log1p(x**2)/2.0


def d1_holland(x):
    '''
    Returns the first derivative of dispersion_holland.
    '''
    return np.arctan(x)


def d2_holland(x):
    '''
    Returns the second derivative of dispersion_holland.
    '''
    return 1.0 / (1.0+x**2)


## Dispersion "autoset" functions, which handle re-scaling.    

def dispersion_barron_autoset(x, alpha, sigma, interpolate=False,
                              oce_flag=False, eta_custom=None):
    '''
    Barron-type dispersion with automatic eta settings.
    '''

    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_barron(sigma=sigma,
                                alpha=alpha,
                                interpolate=interpolate,
                                oce_flag=oce_flag)
    else:
        eta = eta_custom
    
    ## Compute the re-scaled dispersion values.
    if sigma <= 0.0 or sigma == np.inf:
        raise ValueError("Only finite positive sigma are allowed.")
    else:
        return eta * dispersion_barron(x=x/sigma, alpha=alpha)
        
'''
    ##### TEST CODE (if we want to capture zero/inf sigma cases) #####
    ## Depending on the sigma value, the final computations change.
    if sigma == 0.0:
        alpha_condition = alpha >= 1.0 and alpha < 2.0
        error_msg = "Given zero or infinite sigma, only alpha in [1,2) works."
        if alpha_condition:
            return eta * np.absolute(x)**alpha
        else:
            raise ValueError(error_msg)
    
    elif sigma == np.inf:
        if alpha >= 1.0 and alpha < 2.0:
            return eta * x**2
        else:
            raise ValueError(error_msg)
''' 

def d1_barron_autoset(x, alpha, sigma, interpolate=False,
                      oce_flag=False, eta_custom=None):
    '''
    Barron-type dispersion with automatic eta settings.
    '''

    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_barron(sigma=sigma,
                                alpha=alpha,
                                interpolate=interpolate,
                                oce_flag=oce_flag)
    else:
        eta = eta_custom
    
    ## Compute the re-scaled dispersion values.
    if sigma <= 0.0 or sigma == np.inf:
        raise ValueError("Only finite positive sigma are allowed.")
    else:
        return eta * d1_barron(x=x/sigma, alpha=alpha) / sigma


def dispersion_holland_autoset(x, sigma, interpolate=False,
                               oce_flag=False, eta_custom=None):
    '''
    Holland-type dispersion with automatic eta settings.
    '''
    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_holland(sigma=sigma,
                                 interpolate=interpolate,
                                 oce_flag=oce_flag)
    else:
        eta = eta_custom

    ## Compute the re-scaled dispersion values.
    if sigma < 0:
        raise ValueError("Only non-negative sigma values are allowed.")
    elif sigma == 0.0:
        return eta * np.absolute(x=x)
    elif sigma == np.inf:
        return eta * x**2
    else:
        return eta * dispersion_holland(x=x/sigma)


def d1_holland_autoset(x, sigma, interpolate=False,
                       oce_flag=False, eta_custom=None):
    '''
    First derivative of dispersion_holland_autoset.
    '''
    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_holland(sigma=sigma,
                                 interpolate=interpolate,
                                 oce_flag=oce_flag)
    else:
        eta = eta_custom
    
    ## Compute the re-scaled dispersion values.
    if sigma < 0:
        raise ValueError("Only non-negative sigma values are allowed.")
    elif sigma == 0.0:
        return eta * np.sign(x=x)
    elif sigma == np.inf:
        return eta * 2 * x
    else:
        return eta * d1_holland(x=x/sigma) / sigma


## Parser for dispersion functions (with derivatives).

def get_dispersion(name, **kwargs):
    '''
    Simplest parser, returns a dispersion function
    and its derivative, no fancy auto-setting.
    Note that the derivatives are computed *before*
    scaling using sigma, thus there is no 1/sigma
    factor here.
    '''
    if name == "barron":
        dispersion = lambda x, sigma: dispersion_barron(
            x=x/sigma, alpha=kwargs["alpha"]
        )
        dispersion_d1 = lambda x, sigma: d1_barron(
            x=x/sigma, alpha=kwargs["alpha"]
        )
    elif name == "barron1way":
        dispersion = lambda x, sigma: dispersion_barron(
            x=np.where(x>0.0, x/sigma, 0.0), alpha=kwargs["alpha"]
        )
        dispersion_d1 = lambda x, sigma: np.where(
            x>0.0,
            d1_barron(x=np.where(x>0.0, x/sigma, 0.0),
                      alpha=kwargs["alpha"]),
            0.0
        )
    else:
        raise ValueError("Please provide a valid dispersion name.")
    
    return (dispersion, dispersion_d1)


def get_dispersion_autoset(name, **kwargs):
    '''
    A parser that returns a dispersion function
    and its derivative, with appropriate scaling
    and weighting.
    '''
    if name == "barron":
        dispersion = lambda x, sigma, eta: dispersion_barron_autoset(
            x=x, alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
        dispersion_d1 = lambda x, sigma, eta: d1_barron_autoset(
            x=x, alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
    elif name == "barron1way":
        dispersion = lambda x, sigma, eta: dispersion_barron_autoset(
            x=np.where(x>0.0, x, 0.0),
            alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
        dispersion_d1 = lambda x, sigma, eta: d1_barron_autoset(
            x=np.where(x>0.0, x, 0.0),
            alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        ) * np.where(x>0.0, 1.0, 0.0)
    elif name == "huber":
        # the penalized pseudo-Huber function case.
        #beta = kwargs["beta"]
        raise NotImplementedError
    elif name == "holland":
        dispersion = lambda x, sigma, eta: dispersion_holland_autoset(
            x=x, sigma=sigma, interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
        dispersion_d1 = lambda x, sigma, eta: d1_holland_autoset(
            x=x, sigma=sigma, interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
    else:
        raise ValueError("Please provide a valid dispersion name.")

    return (dispersion, dispersion_d1)


###############################################################################
