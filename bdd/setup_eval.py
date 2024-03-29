'''Setup: post-training evaluation of performance.'''

## External modules.
import numpy as np
from sklearn.metrics import confusion_matrix

## Internal modules.
from mml.losses.classification import Zero_One
from mml.utils.linalg import onehot


###############################################################################


## Evaluation metric parser.

def get_eval(loss_base=None, loss=None, **kwargs):
    '''
    '''
    
    evaluators = {}
    
    if loss_base is None:
        return evaluators
    else:
        ## Evaluation function using base loss.
        ## Note: "dist" is for distribution computations.
        eval_loss_base = lambda model, X, y: loss_base(model=model,
                                                       X=X, y=y)
        evaluators.update({"base": eval_loss_base,
                           "dist": eval_loss_base})
    
    if loss is not None:

        ## Evaluation function using the loss defining the objective.
        if kwargs["risk_name"] in ["dro", "entropic"]:
            ## Here we use the handy orig() method; returns a scalar.
            eval_loss = lambda model, X, y: loss.orig(model=model, X=X, y=y)
        else:
            ## Here we just run the vectorized loss fn; returns an array.
            eval_loss = lambda model, X, y: loss(model=model, X=X, y=y)
        evaluators.update({"obj": eval_loss})
    
    ## If learning task is classification, also use 0-1 loss.
    if kwargs["type"] == "classification":
        loss_01 = Zero_One()
        eval_loss_01 = lambda model, X, y: loss_01(model=model, X=X, y=y)
        eval_confusion = lambda model, X, y: confusion(model=model, X=X, y=y)
        evaluators.update({"zeroone": eval_loss_01,
                           "confuse": eval_confusion})
    
    ## Finally, for reference we will record two norms.
    eval_l1 = lambda model, X, y: np.linalg.norm(model.paras["w"].reshape(-1),
                                                 ord=1)
    eval_l2 = lambda model, X, y: np.linalg.norm(model.paras["w"].reshape(-1),
                                                 ord=2)
    evaluators.update({"l1": eval_l1, "l2": eval_l2})
    
    return evaluators


## Evaluation procedures.

def eval_model(epoch, model, storage, data,
               evaluators, risk_name, save_dist):
    '''
    '''
    for key_data in data.keys():
        store = storage[key_data]
        X, y = data[key_data]
        for key_eval, evaluator in evaluators.items():
            if key_eval == "dist":
                ## Store base test distribution if desired.
                if key_data == "test" and save_dist:
                    store[key_eval] = evaluator(model=model,
                                                X=X, y=y).reshape(-1)
            elif key_eval == "confuse":
                ## For confusion matrix, shape is special.
                store[key_eval][epoch,:] = evaluator(model=model,
                                                     X=X, y=y)
            else:
                ## For all other evaluators, just save summary statistics.
                if key_eval == "obj" and risk_name in ["dro", "entropic"]:
                    store[key_eval][epoch,0] = evaluator(model=model,
                                                         X=X, y=y)
                    store[key_eval][epoch,1] = np.nan
                    store[key_eval][epoch,2] = np.nan
                else:
                    evaluations = evaluator(model=model, X=X, y=y)
                    store[key_eval][epoch,0] = evaluations.mean()
                    store[key_eval][epoch,1] = np.median(evaluations)
                    store[key_eval][epoch,2] = evaluations.std()
    return None


def eval_models(epoch, models, storage, data,
                evaluators, risk_name, save_dist):
    '''
    Loops over the model list, assuming enumerated index
    matches the performance array index.
    '''
    for j, model in enumerate(models):
        eval_model(epoch=epoch, model=model, model_idx=j,
                   storage=storage, data=data,
                   evaluators=evaluators,
                   risk_name=risk_name,
                   save_dist=save_dist)
    return None


## Sub-routine for writing to disk.

def eval_write(fname, storage):
    '''
    Write the model evaluations to disk as desired.
    '''
    ## Write to disk as desired.
    for key_data, store in storage.items():
        for key_eval in store.keys():
            if store[key_eval] is not None:
                if key_eval == "confuse":
                    ## For confusion matrix, store only FINAL epoch.
                    to_write = store[key_eval][-1,:]
                else:
                    to_write = store[key_eval]
                np.savetxt(fname=".".join([fname, key_eval+"_"+key_data]),
                           X=to_write, fmt="%.7e", delimiter=",")
    return None


## Helper function for confusion matrix computation.

def confusion(model, X, y):
    return confusion_matrix(
        y_true=y.argmax(axis=1),
        y_pred=model(X=X).argmax(axis=1),
        labels=np.arange(y.shape[1]),
        normalize="all"
    )


###############################################################################
