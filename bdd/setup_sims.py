'''Setup: all sorts of helper functions used mainly in simulations.'''

## External modules.
import numpy as np
from scipy.special import gamma as gamma_fn

## Internal modules.
from setup_dispersions import dispersion_barron


###############################################################################


## Objective functions for each (relevant) risk class.

def obfn_rrisk(theta, x, paras):
    '''
    For our regularized risk.
    '''
    eta = paras["eta"]
    dispersion = get_disp_barron(alpha=paras["alpha"])
    sigma = paras["sigma"]
    return np.mean(x) + eta * np.mean(dispersion(x=(x-theta)/sigma))


def obfn_mloc(theta, x, paras):
    '''
    For M-location.
    '''
    dispersion = get_disp_barron(alpha=paras["alpha"])
    sigma = paras["sigma"]
    return np.mean(dispersion(x=(x-theta)/sigma))


def obfn_trisk(theta, x, paras):
    '''
    For our threshold risk class.
    '''
    etatilde = paras["etatilde"]
    dispersion = get_disp_barron(alpha=paras["alpha"])
    sigma = paras["sigma"]
    return np.mean(dispersion(x=(x-theta)/sigma)) + etatilde * theta


def obfn_trisk1way(theta, x, paras):
    '''
    For our threshold risk class (one-directional).
    '''
    etatilde = paras["etatilde"]
    dispersion = get_disp_barron(alpha=paras["alpha"], oneway=True)
    sigma = paras["sigma"]
    return np.mean(dispersion(x=(x-theta)/sigma)) + etatilde * theta


def obfn_cvar(theta, x, paras):
    '''
    For CVaR risk.
    '''
    prob = paras["prob"]
    return theta + np.mean(np.where(x >= theta, x-theta, 0.0)) / (1.0-prob)


def obfn_entropic(theta, x, paras):
    '''
    For entropic risk (assuming positive gamma).
    '''
    gamma = paras["gamma"]
    if gamma <= 0.0:
        raise ValueError("This objective only works with positive gamma.")
    else:
        return theta + (np.mean(np.exp(gamma*(x-theta)))-1.0)/gamma


def obfn_dro(theta, x, paras):
    '''
    For the DRO risk we consider.
    '''
    shape = 2.0
    bound = 0.5*(1.0/(1.0-paras["atilde"])-1.0)**2
    sstar = shape / (shape-1.0) # shape-star.
    factor = (1.0 + shape*(shape-1.0)*bound)**(1.0/shape)
    return theta + factor * np.mean(np.where(x >= theta, x-theta, 0.0)**sstar)**(1.0/sstar)


## A simple wrapper for parsing "obfn" functions.

def get_obfn(name):
    if name == "rrisk":
        return obfn_rrisk
    elif name == "mloc":
        return obfn_mloc
    elif name == "trisk":
        return obfn_trisk
    elif name == "triskminus":
        return obfn_trisk
    elif name == "trisk1way":
        return obfn_trisk1way
    elif name == "cvar":
        return obfn_cvar
    elif name == "entropic":
        return obfn_entropic
    elif name == "dro":
        return obfn_dro
    else:
        raise ValueError("Please provide a proper obfn name.")


## Function for computing the entropic risk directly.

def get_entropic(x, gamma):
    '''
    A direct computation of the entropic risk.
    '''
    return np.log(np.mean(np.exp(gamma*x))) / gamma


## A simple wrapper for specifying Barron-type dispersion functions.

def get_disp_barron(alpha, oneway=False):
    '''
    Take shape parameter alpha, and return the desired function.
    '''
    if oneway:
        return lambda x: dispersion_barron(x=np.where(x>0.0, x, 0.0),
                                           alpha=alpha)
    else:
        return lambda x: dispersion_barron(x=x, alpha=alpha)


## A function for preparing brackets.

def bracket_prep(x, paras, obfn_name, verbose):
    
    x_init = np.mean(x)
    x_low = np.amin(x)
    x_high = np.amax(x)
    
    ## Prepare the relevant objective function.
    if obfn_name == "rrisk":
        obfn = lambda theta: obfn_rrisk(theta=theta, x=x, paras=paras)
    elif obfn_name == "mloc":
        obfn = lambda theta: obfn_mloc(theta=theta, x=x, paras=paras)
    elif obfn_name == "trisk":
        obfn = lambda theta: obfn_trisk(theta=theta, x=x, paras=paras)
    elif obfn_name == "triskminus":
        obfn = lambda theta: obfn_trisk(theta=theta, x=x, paras=paras)
    elif obfn_name == "trisk1way":
        obfn = lambda theta: obfn_trisk1way(theta=theta, x=x, paras=paras)
    elif obfn_name == "cvar":
        obfn = lambda theta: obfn_cvar(theta=theta, x=x, paras=paras)
    elif obfn_name == "entropic":
        obfn = lambda theta: obfn_entropic(theta=theta, x=x, paras=paras)
    elif obfn_name == "dro":
        obfn = lambda theta: obfn_dro(theta=theta, x=x, paras=paras)
    else:
        raise ValueError("Please pass a valid obfn_name.")
    
    ## Compute brackets.
    f_init = obfn(theta=x_init)
    f_low = obfn(theta=x_low)
    f_high = obfn(theta=x_high)
    while f_low < f_init:
        x_low -= np.absolute(x_init) + np.absolute(x_low)
        f_low = obfn(theta=x_low)
        if verbose:
            print("Bracket prep ({}): extending MIN side.".format(obfn_name))
            
    while f_high < f_init:
        x_high += np.absolute(x_init) + np.absolute(x_high)
        f_high = obfn(theta=x_high)
        if verbose:
            print("Bracket prep ({}): extending MAX side.".format(obfn_name))
    
    cond_bracket = (f_low > f_init) and (f_high > f_init)
    if cond_bracket == False:
        print("Warning: bracket condition is", cond_bracket)
        print("Details:", f_low, f_init, f_high)
    
    return (x_low, x_init, x_high)


## Data generating functions using standard parametric distributions.

def gen_data(n, name, rg):
    '''
    Function for generating data.
    '''
    if name == "bernoulli":
        prob = 0.25
        x = rg.uniform(low=0.0, high=1.0, size=(n,1))
        return np.where(x <= prob, 1.0, 0.0)
    elif name == "beta":
        a, b = (1.0, 0.5)
        return rg.beta(a=a, b=b, size=(n,1))
    elif name == "chisquare":
        df = 3.5
        return rg.chisquare(df=df, size=(n,1))
    elif name == "exponential":
        scale = 1.0
        return rg.exponential(scale=scale, size=(n,1))
    elif name == "gamma":
        shape, scale = (4.0, 1.0)
        return rg.gamma(shape=shape, scale=scale, size=(n,1))
    elif name == "gamma-unitvar":
        shape, scale = (1.0, 1.0)
        return rg.gamma(shape=shape, scale=scale, size=(n,1))
    elif name == "lognormal":
        mean, sigma = (0.0, 0.5)
        return rg.lognormal(mean=mean, sigma=sigma, size=(n,1))
    elif name == "normal":
        loc, scale = (0.0, 1.0)
        return rg.normal(loc=loc, scale=scale, size=(n,1))
    elif name == "normal-sharp":
        loc, scale = (0.0, 0.5)
        return rg.normal(loc=loc, scale=scale, size=(n,1))
    elif name == "pareto":
        a = 3.5
        return rg.pareto(a=a, size=(n,1))
    elif name == "uniform":
        low, high = (-0.5, 0.5)
        return rg.uniform(low=low, high=high, size=(n,1))
    elif name == "wald":
        mean, scale = (1.0, 1.0)
        return rg.wald(mean=mean, scale=scale, size=(n,1))
    elif name == "weibull":
        a = 1.2
        return rg.weibull(a=a, size=(n,1))
    elif name == "weibull-unitvar":
        a = 1.2
        rescaler = 1.0 / np.sqrt(gamma_fn(1+2/a)-gamma_fn(1+1/a)**2)
        return rescaler*rg.weibull(a=a, size=(n,1))
    else:
        return None

## Names and key subsets of the data types being used.

data_all = ["bernoulli", "beta", "chisquare", "exponential", "gamma",
            "lognormal", "normal", "pareto", "uniform",
            "wald", "weibull"]
data_bounded = ["bernoulli", "beta", "uniform"]
data_unbounded = ["chisquare", "exponential", "gamma", "lognormal",
                  "normal", "pareto", "wald", "weibull"]
data_heavytails = ["chisquare", "exponential", "lognormal",
                   "pareto", "wald", "weibull"]
data_symmetric = ["normal", "uniform"]


###############################################################################
