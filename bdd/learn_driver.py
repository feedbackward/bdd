'''Driver script for tests of learning algorithms under diverse risks.'''

## External modules.
from argparse import ArgumentParser
from copy import deepcopy
from json import dump as json_dump
import numpy as np
from os import path

## Internal modules.
from mml.utils import makedir_safe
from setup_algos import get_algo
from setup_data import get_data
from setup_eval import get_eval, eval_model, eval_write
from setup_losses import get_loss
from setup_models import get_model
from setup_results import results_dir
from setup_train import train_epoch


###############################################################################


## Basic setup.

parser = ArgumentParser(description="Arguments for driver script.")

parser.add_argument("--algo-ancillary",
                    help="Ancillary algorithm class (default: SGD).",
                    type=str, default="SGD", metavar="S")
parser.add_argument("--algo-main",
                    help="Main algorithm class to test (default: '').",
                    type=str, default="", metavar="S")
parser.add_argument("--alpha",
                    help="Set Barron alpha parameter (default: 2.0).",
                    type=float, default=2.0, metavar="F")
parser.add_argument("--atilde",
                    help="For setting DRO_CR bound (default: 0.5).",
                    type=float, default=0.5, metavar="F")
parser.add_argument("--batch-size",
                    help="Mini-batch size for algorithms (default: 1).",
                    type=int, default=1, metavar="N")
parser.add_argument("--beta",
                    help="Set Huber beta parameter (default: 0.5).",
                    type=float, default=0.5, metavar="F")
parser.add_argument("--data",
                    help="Specify data set to be used (default: None).",
                    type=str, default=None, metavar="S")
parser.add_argument("--dispersion",
                    help="Dispersion for R-risk or T-risk (default: barron).",
                    type=str, default="barron", metavar="S")
parser.add_argument("--entropy",
                    help="For data-gen seed sequence (default is random).",
                    type=int,
                    default=np.random.SeedSequence().entropy,
                    metavar="N")
parser.add_argument("--eta",
                    help="Weight for R-risk (default: 1.0).",
                    type=float, default=1.0, metavar="F")
parser.add_argument("--etatilde",
                    help="Weight for T-risk (default: 1.0).",
                    type=float, default=1.0, metavar="F")
parser.add_argument("--gamma",
                    help="Shape of entropic risk (default: 1.0).",
                    type=float, default=1.0, metavar="F")
parser.add_argument("--interpolate",
                    help="Flag for automated interpolator (default: False).",
                    action="store_true", default=False)
parser.add_argument("--loss-base",
                    help="Loss name (default: quadratic).",
                    type=str, default="quadratic", metavar="S")
parser.add_argument("--model",
                    help="Model class (default: linreg).",
                    type=str, default="linreg", metavar="S")
parser.add_argument("--noise-rate",
                    help="Noisy label rate (default: 0.0).",
                    type=float, default=0.0, metavar="F")
parser.add_argument("--num-epochs",
                    help="Number of epochs to run (default: 3)",
                    type=int, default=3, metavar="N")
parser.add_argument("--num-trials",
                    help="Number of independent random trials (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--prob",
                    help="Set CVaR quantile level (default: 0.9).",
                    type=float, default=0.9, metavar="F")
parser.add_argument("--risk-name",
                    help="Risk function name. Default is 'mean'.",
                    type=str, default="mean", metavar="S")
parser.add_argument("--set-threshold",
                    help="Name of threshold setter (default: 'none').",
                    type=str, default="none", metavar="S")
parser.add_argument("--save-dist",
                    help="Save loss distribution or not. (default: 'no')",
                    type=str, default="no", metavar="S")
parser.add_argument("--sigma",
                    help="Scale sigma for R-risk and T-risk (default: 1.0).",
                    type=float, default=1.0, metavar="F")
parser.add_argument("--step-size",
                    help="Step size parameter (default: 0.01).",
                    type=float, default=0.01, metavar="F")
parser.add_argument("--task-name",
                    help="A task name. Default is the word 'default'.",
                    type=str, default="default", metavar="S")

## Parse the arguments passed via command line.
args = parser.parse_args()
if args.data is None:
    raise TypeError("No dataset has been specified; use '--data=[.]'.")

## Name to be used identifying the results etc. of this experiment.
towrite_name = args.task_name+"-"+"_".join([args.risk_name,
                                            args.loss_base,
                                            args.model,
                                            args.algo_ancillary])
if len(args.algo_main) > 0:
    towrite_name += "_{}".format(args.algo_main)

## Prepare a directory to save results.
if args.noise_rate > 0.0:
    towrite_dir = path.join(results_dir, args.data+"-noisy")
else:
    towrite_dir = path.join(results_dir, args.data)
makedir_safe(towrite_dir)

## Setup of manually-specified seed sequence for data generation.
ss_parent = np.random.SeedSequence(args.entropy)
ss_data_children = ss_parent.spawn(args.num_trials)
ss_mth_children = ss_parent.spawn(args.num_trials)
rg_data_children = [np.random.default_rng(seed=ss) for ss in ss_data_children]
rg_mth_children = [np.random.default_rng(seed=ss) for ss in ss_mth_children]


## Write a JSON file to disk that summarizes key experiment parameters.
dict_to_json = vars(args)
towrite_json = path.join(towrite_dir, towrite_name+".json")
with open(towrite_json, "w", encoding="utf-8") as f:
    json_dump(obj=dict_to_json, fp=f,
              ensure_ascii=False,
              sort_keys=True, indent=4)

if __name__ == "__main__":

    ## Arguments for constructing loss object.
    loss_kwargs = {
        "alpha": args.alpha,
        "atilde": args.atilde,
        "beta": args.beta,
        "dispersion": args.dispersion,
        "eta": args.eta,
        "etatilde": args.etatilde,
        "gamma": args.gamma,
        "interpolate": args.interpolate,
        "prob": args.prob,
        "set_threshold": args.set_threshold,
        "sigma": args.sigma,
        "risk_name": args.risk_name
    }
    
    ## Arguments for algorithms.
    algo_kwargs = {}
    
    ## Arguments for models.
    model_kwargs = {}
    
    ## Prepare the loss used for training.
    loss_base, loss = get_loss(name=args.loss_base, **loss_kwargs)
    
    ## Start the loop over independent trials.
    for trial in range(args.num_trials):
        
        ## Get trial-specific random generator.
        rg_data = rg_data_children[trial]
        rg_mth = rg_mth_children[trial]
        
        ## Load in data.
        print("Doing data prep.")
        (X_train, y_train, X_val, y_val,
         X_test, y_test, ds_paras) = get_data(
             dataset=args.data, rg=rg_data, noisy_label_rate=args.noise_rate
         )
        
        ## Data index.
        data_idx = np.arange(len(X_train))
        
        ## Prepare evaluation metric(s).
        evaluators = get_eval(loss_base=loss_base, loss=loss,
                              **loss_kwargs, **ds_paras)
        
        ## Model setup.
        model_ancillary = get_model(
            name=args.model,
            paras_init=None,
            rg=rg_mth,
            **loss_kwargs, **model_kwargs, **ds_paras
        )
        if args.algo_main is not None and len(args.algo_main) > 0:
            model_main = get_model(
                name=args.model,
                paras_init=deepcopy(model_ancillary.paras),
                rg=rg_mth,
                **loss_kwargs, **model_kwargs, **ds_paras
            )
        else:
            model_main = None
        
        ## Prepare algorithms.
        model_dim = np.array(
            [p.size for pn, p in model_ancillary.paras.items()]
        ).sum()
        algo_kwargs.update(
            {"num_data": len(X_train),
             "step_size": args.step_size/np.sqrt(model_dim)}
        )
        algo_ancillary, algo_main = get_algo(
            name=args.algo_ancillary,
            model=model_ancillary,
            loss=loss,
            name_main=args.algo_main,
            model_main=model_main,
            **ds_paras, **algo_kwargs, **loss_kwargs
        )
        
        ## Prepare storage for performance evaluation this trial.
        store_train, store_val, store_test = {}, {}, {}
        num_records = args.num_epochs + 1
        for key_eval in evaluators.keys():

            if key_eval == "dist":
                store_train[key_eval] = None
                store_val[key_eval] = None
                store_test[key_eval] = None
            elif key_eval == "confuse":
                ## In this case, store full confusion matrix.
                num_classes = y_train.shape[1]
                store_train[key_eval] = np.zeros(shape=(num_records,
                                                        num_classes,
                                                        num_classes),
                                                 dtype=np.float32)
                store_val[key_eval] = np.zeros(shape=(num_records,
                                                      num_classes,
                                                      num_classes),
                                               dtype=np.float32)
                store_test[key_eval] = np.zeros(shape=(num_records,
                                                       num_classes,
                                                       num_classes),
                                                dtype=np.float32)
            else:
                ## Just store three stats: mean, median, and stdev. 
                store_train[key_eval] = np.zeros(shape=(num_records,3),
                                                 dtype=np.float32)
                store_val[key_eval] = np.zeros(shape=(num_records,3),
                                               dtype=np.float32)
                store_test[key_eval] = np.zeros(shape=(num_records,3),
                                                dtype=np.float32)
        
        storage = {"train": store_train,
                   "val": store_val,
                   "test": store_test}
        
        ## Loop over epochs.
        for epoch in range(args.num_epochs):
            
            print("(Tr {}) Ep {} starting.".format(trial, epoch))

            #### START: Evaluation block (during training) ####
            ## Ensure the correct model is evaluated.
            if model_main is not None:
                model_to_eval = model_main
            else:
                model_to_eval = model_ancillary
                
            ## Evaluate performance.
            eval_model(epoch=epoch,
                       model=model_to_eval,
                       storage=storage,
                       data={"train": (X_train, y_train),
                             "val": (X_val, y_val),
                             "test": (X_test, y_test)},
                       evaluators=evaluators,
                       risk_name=args.risk_name,
                       save_dist=False)
            #### END: Evaluation block (during training) ####
            
            ## Shuffle data.
            rg_data.shuffle(data_idx)
            X_train = X_train[data_idx,:]
            y_train = y_train[data_idx,:]

            ## Carry out one epoch's worth of training.
            train_epoch(algo=algo_ancillary,
                        loss=loss,
                        X=X_train,
                        y=y_train,
                        batch_size=args.batch_size,
                        algo_main=algo_main)
            
            print("(Tr {}) Ep {} finished.".format(trial, epoch), "\n")

        ## Having finished the final step, check whether to save dist.
        save_dist = all([args.save_dist == "yes",
                         trial == 0])
        
        #### START: Evaluation block (final step) ####
        ## Ensure the correct model is evaluated.
        if model_main is not None:
            model_to_eval = model_main
        else:
            model_to_eval = model_ancillary
        ## Evaluate performance.
        eval_model(epoch=args.num_epochs, # saves as final record.
                   model=model_to_eval,
                   storage=storage,
                   data={"train": (X_train, y_train),
                         "val": (X_val, y_val),
                         "test": (X_test, y_test)},
                   evaluators=evaluators,
                   risk_name=args.risk_name,
                   save_dist=save_dist)
        #### END: Evaluation block (final step) ####
        
        ## Write performance for this trial to disk.
        perf_fname = path.join(towrite_dir,
                               towrite_name+"-"+str(trial))
        eval_write(fname=perf_fname, storage=storage)


###############################################################################
