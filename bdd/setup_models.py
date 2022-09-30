'''Setup: models.'''

## External modules.

## Internal modules.
from mml.models import init_range
from mml.models.linreg import LinearRegression, LinearRegression_Multi


###############################################################################


## The main parser function, returning model instances.

def get_model(name, paras_init=None, rg=None, **kwargs):
    
    ## Parse the model name and instantiate the desired model.
    if name == "linreg_multi":
        model_out = LinearRegression_Multi(num_features=kwargs["num_features"],
                                           num_outputs=kwargs["num_classes"],
                                           paras_init=paras_init,
                                           rg=rg)
    elif name == "linreg":
        model_out = LinearRegression(num_features=kwargs["num_features"],
                                     paras_init=paras_init, rg=rg)
    else:
        raise ValueError("Please pass a valid model name.")
    
    ## As needed, initialize any extra parameters.
    no_theta = "theta" not in model_out.paras
    needs_theta = ["rrisk", "trisk", "dro", "meanvar",
                   "triskSigS", "triskSigM", "triskSigL"]
    no_v = "v" not in model_out.paras
    if kwargs["risk_name"] in needs_theta and no_theta:
        model_out.paras["theta"] = rg.uniform(low=0.0,
                                              high=init_range,
                                              size=(1,1))
    elif kwargs["risk_name"] == "cvar" and no_v:
        model_out.paras["v"] = rg.uniform(low=0.0,
                                          high=init_range,
                                          size=(1,1))
    else:
        pass # don't need to do anything in this case.
    
    ## Finally, return the completely initialized model.
    return model_out
    

###############################################################################
