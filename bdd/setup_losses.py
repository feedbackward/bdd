'''Setup: loss functions used for training and evaluation.'''

## External modules.
from copy import deepcopy
import numpy as np

## Internal modules.
from mml.losses import Loss
from mml.losses.absolute import Absolute
from mml.losses.classification import Zero_One
from mml.losses.cvar import CVaR
from mml.losses.dro import DRO_CR
from mml.losses.logistic import Logistic
from mml.losses.quadratic import Quadratic
from mml.losses.tilted import Tilted
from mml.utils.mest import est_loc_fixedpt, scale_madmed
from setup_dispersions import get_dispersion


###############################################################################


## Special loss class definitions.

class R_Risk(Loss):
    '''
    A special loss class, which takes a base loss as
    an input, and returns a modified loss which is
    an unbiased estimate of the R-risk (regularized risk).
    '''
    def __init__(self, loss_base, dispersion, dispersion_d1,
                 sigma=None, eta=None, name=None):
        loss_name = "R_Risk x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.dispersion = dispersion
        self.dispersion_d1 = dispersion_d1
        self.sigma = sigma
        self.eta = eta
        return None

    
    def func(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y) # compute losses.
        theta = model.paras["theta"].item() # extract scalar.
        return losses + self.eta * self.dispersion(
            x=losses-theta,
            sigma=self.sigma
        )
    
    
    def grad(self, model, X, y):
        '''
        '''
        
        ## Initial computations.
        losses = self.loss(model=model, X=X, y=y) # compute losses.
        loss_grads = self.loss.grad(model=model, X=X, y=y) # loss gradients.
        theta = model.paras["theta"].item() # extract scalar.
        dispersion_grads = self.dispersion_d1(
            x=losses-theta,
            sigma=self.sigma
        ) # evaluate the derivative of the dispersion term.
        ddim = dispersion_grads.ndim
        tdim = model.paras["theta"].ndim
        
        ## Main gradient computations.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ddim > gdim:
                raise ValueError("Axis dimensions are wrong; ddim > gdim.")
            elif ddim < gdim:
                dispersion_grads_exp = np.expand_dims(
                    a=dispersion_grads,
                    axis=tuple(range(ddim,gdim))
                )
                g *= 1.0 + self.eta * dispersion_grads_exp / self.sigma
            else:
                g *= 1.0 + self.eta * dispersion_grads / self.sigma
        
        ## Compute the derivative with respect to threshold theta.
        ## (be careful to note the minus sign)
        loss_grads["theta"] = -self.eta * np.expand_dims(
            a=dispersion_grads,
            axis=tuple(range(ddim,1+tdim))
        ) / self.sigma
        
        ## Return gradients for all parameters being optimized.
        return loss_grads


class T_Risk(Loss):
    '''
    A special loss class, which takes a base loss as
    an input, and returns a modified loss which is
    an unbiased estimate of the R-risk (regularized risk).
    '''
    def __init__(self, loss_base, dispersion, dispersion_d1,
                 sigma=None, etatilde=None, name=None):
        loss_name = "T_Risk x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.dispersion = dispersion
        self.dispersion_d1 = dispersion_d1
        self.sigma = sigma
        self.etatilde = etatilde
        return None
    
    
    def func(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y) # compute losses.
        theta = model.paras["theta"].item() # extract scalar.
        return self.etatilde * theta + self.dispersion(
            x=losses-theta,
            sigma=self.sigma
        )
    
    
    def grad(self, model, X, y):
        '''
        '''
        
        ## Initial computations.
        losses = self.loss(model=model, X=X, y=y) # compute losses.
        loss_grads = self.loss.grad(model=model, X=X, y=y) # loss gradients.
        theta = model.paras["theta"].item() # extract scalar.
        dispersion_grads = self.dispersion_d1(
            x=losses-theta,
            sigma=self.sigma
        ) # evaluate the derivative of the dispersion term.
        ddim = dispersion_grads.ndim
        tdim = model.paras["theta"].ndim
        
        ## Main gradient computations.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ddim > gdim:
                raise ValueError("Axis dimensions are wrong; ddim > gdim.")
            elif ddim < gdim:
                dispersion_grads_exp = np.expand_dims(
                    a=dispersion_grads,
                    axis=tuple(range(ddim,gdim))
                )
                g *= dispersion_grads_exp
            else:
                g *= dispersion_grads

        ## Compute the derivative with respect to threshold theta.
        ## (be careful to note the minus sign)
        loss_grads["theta"] = self.etatilde - np.expand_dims(
            a=dispersion_grads,
            axis=tuple(range(ddim,1+tdim))
        ) / self.sigma
        
        ## Return gradients for all parameters being optimized.
        return loss_grads


class T_Risk_General(Loss):
    '''
    Generalized form of the T-risk, where the threshold
    theta is specified at construction, and is not a
    parameter to be optimized.
    '''
    def __init__(self, loss_base, dispersion, dispersion_d1,
                 theta, sigma=None, etatilde=None, name=None):
        loss_name = "T_Risk_General x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.dispersion = dispersion
        self.dispersion_d1 = dispersion_d1
        self.theta = theta
        self.sigma = sigma
        self.etatilde = etatilde
        return None
    
    
    def func(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y) # compute losses.
        return self.etatilde * self.theta + self.dispersion(
            x=losses-self.theta,
            sigma=self.sigma
        )
    
    
    def grad(self, model, X, y):
        '''
        '''
        
        ## Initial computations.
        losses = self.loss(model=model, X=X, y=y) # compute losses.
        loss_grads = self.loss.grad(model=model, X=X, y=y) # loss gradients.
        dispersion_grads = self.dispersion_d1(
            x=losses-self.theta,
            sigma=self.sigma
        ) # evaluate the derivative of the dispersion term.
        ddim = dispersion_grads.ndim
        
        ## Main gradient computations.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ddim > gdim:
                raise ValueError("Axis dimensions are wrong; ddim > gdim.")
            elif ddim < gdim:
                dispersion_grads_exp = np.expand_dims(
                    a=dispersion_grads,
                    axis=tuple(range(ddim,gdim))
                )
                g *= dispersion_grads_exp
            else:
                g *= dispersion_grads

        ## Return gradients for all parameters being optimized.
        return loss_grads


class T_Risk_CustomThreshold(Loss):
    '''
    Generalized form of the T-risk, where the threshold
    theta is determined internally based on loss/grad stats
    computed by a function passed at construction time.
    '''
    def __init__(self, loss_base, dispersion, dispersion_d1, set_threshold,
                 sigma=None, etatilde=None, name=None):
        loss_name = "T_Risk_CustomThreshold x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.dispersion = dispersion
        self.dispersion_d1 = dispersion_d1
        self.theta = None
        self.sigma = sigma
        self.etatilde = etatilde
        self.set_threshold = set_threshold
        return None


    def update_threshold(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y) # compute base losses.
        self.theta = self.set_threshold(x=losses) # set new threshold.
        self.sigma = scale_madmed(X=losses) # use MAD about median.
        return None


    def func(self, model, X, y):
        '''
        '''
        ## Set the threshold value if it is not set already.
        if self.theta is None:
            self.update_threshold(model=model, X=X, y=y)
        
        ## Loss computations.
        losses = self.loss(model=model, X=X, y=y) # compute base losses.
        return self.etatilde * self.theta + self.dispersion(
            x=losses-self.theta,
            sigma=self.sigma
        )
    
    
    def grad(self, model, X, y):
        '''
        '''
        
        ## Set the threshold value if it is not set already.
        if self.theta is None:
            self.update_threshold(model=model, X=X, y=y)
        
        ## Initial computations.
        losses = self.loss(model=model, X=X, y=y) # compute base losses.
        loss_grads = self.loss.grad(model=model, X=X, y=y) # loss gradients.
        dispersion_grads = self.dispersion_d1(
            x=losses-self.theta,
            sigma=self.sigma
        ) # evaluate the derivative of the dispersion term.
        ddim = dispersion_grads.ndim
        
        ## Main gradient computations.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ddim > gdim:
                raise ValueError("Axis dimensions are wrong; ddim > gdim.")
            elif ddim < gdim:
                dispersion_grads_exp = np.expand_dims(
                    a=dispersion_grads,
                    axis=tuple(range(ddim,gdim))
                )
                g *= dispersion_grads_exp
            else:
                g *= dispersion_grads

        ## Return gradients for all parameters being optimized.
        return loss_grads


class ConvexPolynomial(Loss):
    '''
    '''
    
    def __init__(self, exponent, name=None):
        super().__init__(name=name)
        self.exponent = exponent
        if self.exponent < 1.0:
            raise ValueError("This class only takes exponent >= 1.0.")
        return None

    
    def func(self, model, X, y):
        '''
        '''
        abdiffs = np.absolute(model(X=X)-y)
        if self.exponent == 1.0:
            return abdiffs
        else:
            return abdiffs**self.exponent / self.exponent
    
    
    def grad(self, model, X, y):
        '''
        '''
        
        loss_grads = deepcopy(model.grad(X=X)) # start with model grads.
        diffs = model(X=X)-y
        
        if self.exponent == 1.0:
            factors = np.sign(diffs)
        else:
            factors = np.absolute(diffs)**(self.exponent-1.0) * np.sign(diffs)
        
        ## Shape check to be safe.
        if factors.ndim != 2:
            raise ValueError("Require model(X)-y to have shape (n,1).")
        elif factors.shape[1] != 1:
            raise ValueError("Only implemented for single-output models.")
        else:
            for pn, g in loss_grads.items():
                g *= np.expand_dims(a=factors,
                                    axis=tuple(range(2,g.ndim)))
        return loss_grads


## Parser function for setting the DRO_CR parameters.
def parse_dro(atilde):
    shape = 2.0
    bound = ((1.0/(1.0-atilde))-1.0)**2.0 / 2.0
    return (bound, shape)


## Parser function for threshold setters.
def parse_threshold_setter(name, dispersion_d1=None, sigma=None):
    if name == "mean":
        return lambda x: np.mean(x)
    elif name == "median":
        return lambda x: np.median(x)
    elif name == "mest":
        if dispersion_d1 is None or sigma is None:
            s_error = "Missing dispersion_d1 or sigma for M-estimation."
            raise ValueError(s_error)
        else:
            inf_fn = lambda x: dispersion_d1(x=x, sigma=1.0)
            return lambda x: est_loc_fixedpt(X=x, s=sigma,
                                             inf_fn=inf_fn)
    else:
        s_error = "Did not recognize threshold setter {}".format(name)
        raise ValueError(s_error)


## Grab the desired loss object.

dict_losses = {
    "absolute": Absolute(name="absolute"),
    "logistic": Logistic(name="logistic"),
    "quadratic": Quadratic(name="quadratic"),
    "zeroone": Zero_One(name="zeroone")
}

def get_loss(name, **kwargs):
    '''
    A simple parser that takes a base loss and risk name,
    and returns the loss object that amounts to an unbiased
    estimator of the specified risk.
    '''

    ## First grab the loss and risk name, with a check.
    try:
        loss_base = dict_losses[name]
        risk_name = kwargs["risk_name"]
    except KeyError:
        print("Error: either loss is invalid or risk is missing.")
    
    ## Prepare and return the modified loss object as requested.
    if risk_name == "erm":
        loss = loss_base
    
    elif risk_name in ["rrisk", "trisk", "triskSigS",
                       "triskSigM", "triskSigL", "meanvar"]:
        
        dispersion_kwargs = {"interpolate": kwargs["interpolate"],
                             "alpha": kwargs["alpha"],
                             "beta": kwargs["beta"]}
        dispersion, dispersion_d1 = get_dispersion(
            name=kwargs["dispersion"], **dispersion_kwargs
        )
        
        if risk_name == "rrisk":
            loss = R_Risk(loss_base=loss_base,
                          dispersion=dispersion,
                          dispersion_d1=dispersion_d1,
                          sigma=kwargs["sigma"],
                          eta=kwargs["eta"])
        else:
            loss = T_Risk(loss_base=loss_base,
                          dispersion=dispersion,
                          dispersion_d1=dispersion_d1,
                          sigma=kwargs["sigma"],
                          etatilde=kwargs["etatilde"])

    elif risk_name in ["triskCustom"]:
        dispersion_kwargs = {"interpolate": kwargs["interpolate"],
                             "alpha": kwargs["alpha"],
                             "beta": kwargs["beta"]}
        dispersion, dispersion_d1 = get_dispersion(
            name=kwargs["dispersion"], **dispersion_kwargs
        )
        set_threshold = parse_threshold_setter(name=kwargs["set_threshold"],
                                               dispersion_d1=dispersion_d1,
                                               sigma=kwargs["sigma"])
        loss = T_Risk_CustomThreshold(loss_base=loss_base,
                                      dispersion=dispersion,
                                      dispersion_d1=dispersion_d1,
                                      set_threshold=set_threshold,
                                      sigma=kwargs["sigma"],
                                      etatilde=kwargs["etatilde"])
    elif risk_name == "cvar":
        loss = CVaR(loss_base=loss_base,
                    alpha=1.0-kwargs["prob"])
    
    elif risk_name == "entropic":
        loss = Tilted(loss_base=loss_base,
                      tilt=kwargs["gamma"])
    
    elif risk_name == "dro":
        bound, shape = parse_dro(atilde=kwargs["atilde"])
        loss = DRO_CR(loss_base=loss_base, bound=bound, shape=shape)
    
    else:
        raise ValueError("Invalid risk name.")

    ## Finally, return both the base loss and the modified loss.
    return (loss_base, loss)


###############################################################################
