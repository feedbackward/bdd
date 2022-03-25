'''Setup: preparation of data sets.'''

## External modules.
import os
from pathlib import Path

## Internal modules.
from mml.data import dataset_dict, dataset_list, get_data_general


###############################################################################


## If benchmark data is to be used, specify the directory here.
dir_data_toread = os.path.join(str(Path.home()),
                               "mml", "mml", "data")

## First set dataset parameter dictionary with standard values
## for all the benchmark datasets in mml.
dataset_paras = dataset_dict

## Data generation procedure.
def get_data(dataset, rg, do_normalize=True, do_shuffle=True, do_onehot=True):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    if dataset in dataset_paras:
        paras = dataset_paras[dataset]
        if dataset in dataset_list:
            ## Benchmark dataset case.
            return get_data_general(dataset=dataset,
                                    paras=paras, rg=rg,
                                    directory=dir_data_toread,
                                    do_normalize=do_normalize,
                                    do_shuffle=do_shuffle,
                                    do_onehot=do_onehot)
        else:
            ## Local simulation case.
            return get_data_simulated(paras=paras, rg=rg)
    else:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )


###############################################################################
