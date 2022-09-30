'''Setup: preparation of data sets.'''

## External modules.
import numpy as np
import os
from pathlib import Path

## Internal modules.
from mml.data import dataset_dict, dataset_list, get_data_general
from mml.utils.linalg import onehot


###############################################################################


## If benchmark data is to be used, specify the directory here.
dir_data_toread = os.path.join(str(Path.home()), "datasets")

## First set dataset parameter dictionary with standard values
## for all the benchmark datasets in mml.
dataset_paras = dataset_dict

## Data generation procedure.
def get_data(dataset, rg, noisy_label_rate=0.0, do_normalize=True,
             do_shuffle=True, do_onehot=True):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    if dataset in dataset_paras:
        paras = dataset_paras[dataset]
        if noisy_label_rate > 0.0:
            ## When requested, flip labels uniformly at random.
            (X_train, y_train, X_val, y_val,
             X_test, y_test, ds_paras) = get_data_general(
                 dataset=dataset,
                 paras=paras, rg=rg,
                 directory=dir_data_toread,
                 do_normalize=do_normalize,
                 do_shuffle=do_shuffle,
                 do_onehot=do_onehot
             )
            if ds_paras["num_labels"] > 1:
                raise ValueError(
                    "We only flip labels for single-label setting."
                )
            num_noisy_train = round(noisy_label_rate*len(X_train))
            num_noisy_val = round(noisy_label_rate*len(X_val))
            idx_train = np.arange(len(X_train))
            idx_val = np.arange(len(X_val))
            rg.shuffle(idx_train)
            rg.shuffle(idx_val)
            idx_train_toflip = idx_train[0:num_noisy_train]
            idx_val_toflip = idx_val[0:num_noisy_val]
            y_train[idx_train_toflip,:] = onehot(
                y=rg.integers(low=np.min(y_train),
                              high=np.max(y_train),
                              size=num_noisy_train),
                num_classes=ds_paras["num_classes"]
            )
            y_val[idx_val_toflip,:] = onehot(
                y=rg.integers(low=np.min(y_val),
                              high=np.max(y_val),
                              size=num_noisy_val),
                num_classes=ds_paras["num_classes"]
            )
            return (X_train, y_train, X_val, y_val, X_test, y_test, ds_paras)
        else:
            return get_data_general(dataset=dataset,
                                    paras=paras, rg=rg,
                                    directory=dir_data_toread,
                                    do_normalize=do_normalize,
                                    do_shuffle=do_shuffle,
                                    do_onehot=do_onehot)
    else:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )


###############################################################################
