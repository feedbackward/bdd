# bdd: bidirectional dispersion and risk function design

In this repository, we provide software and demonstrations related to the following paper:

- <a href="https://proceedings.mlr.press/v206/holland23a.html">Flexible risk design using bi-directional dispersion</a>. Matthew J. Holland, AISTATS 2023.

This repository contains code which can be used to faithfully reproduce all the experimental results given in the above paper, and it can be easily applied to more general machine learning tasks outside the examples considered here.


A table of contents for this README file:

- <a href="#setup_init">Setup: initial software preparation</a>
- <a href="#setup_data">Setup: preparing the benchmark data sets</a>
- <a href="#start">Getting started</a>
- <a href="#demos">Demos and visualization</a>
- <a href="#safehash">Safe hash value</a>


<a id="setup_init"></a>
## Setup: initial software preparation

To begin, please ensure you have the <a href="https://github.com/feedbackward/mml#prerequisites">prerequisite software</a> used in the setup of our `mml` repository.

Next, make a local copy of the repository and create a virtual environment for working in as follows:

```
$ git clone https://github.com/feedbackward/mml.git
$ git clone https://github.com/feedbackward/bdd.git
$ conda create -n bdd python=3.9 jupyter matplotlib pip pytables scikit-learn scipy unzip
$ conda activate bdd
```

Having made (and activated) this new environment, we would like to use `pip` to install the supporting libraries for convenient access. This is done easily, by simply running

```
(bdd) $ cd [mml path]/mml
(bdd) $ pip install -e ./
```

with the `[mml path]` placeholder replaced with the path to wherever you placed the repositories. If you desire a safe, tested version of `mml`, just run

```
(bdd) $ git checkout [safe hash mml]
```

and then do the `pip install -e ./` command mentioned above. The `[safe hash mml]` placeholder is to be replaced using the safe hash value given at the end of this document.


<a id="setup_data"></a>
## Setup: preparing the benchmark data sets

Please follow the instructions under <a href="https://github.com/feedbackward/mml#data">"Acquiring benchmark datasets"</a> using our `mml` repository. The rest of this README assumes that the user has prepared any desired benchmark datasets, stored in a local data storage directory (default path is `[path to mml]/mml/mml/data` as specified by the variable `dir_data_towrite` in `mml/mml/config.py`.

One __important__ step is to ensure that once you've acquired the benchmark data using `mml`, you must ensure that `bdd` knows where that data is. To do this, set `dir_data_toread` in `setup_data.py` to the directory housing the HDF5 format data sub-directories (default setting: your home directory).


<a id="start"></a>
## Getting started

We have basically three types of files:

- __Setup files:__ these take the form `setup_*.py`.
  - Configuration for all elements of the learning process, with one setup file for each of the following major categories: learning algorithms, data preparation, learned model evaluation, loss functions, models, result processing, and general-purpose training functions.

- __Driver scripts:__ just one at present, called `learn_driver.py`.
  - This script controls the flow of the learning procedure and handle all the clerical tasks such as organizing, naming, and writing numerical results to disk. No direct modification to this file is needed to run the experiments in the above paper.

- __Execution scripts:__ all the files of the form `run_*.sh`.
  - The choice of algorithm, model, data generation protocol, among other key parameters is made within these simple shell scripts. Basically, parameters are specified explicitly, and these are then passed to the driver script as options.

The experiments using real-world datasets require the user to run the driver scripts themselves; all the other experiments are self-contained within Jupyter notebooks.


### A quick example

Here are two simple examples. The first one uses the pre-prepared script just for the "iris" dataset.

```
(bdd) bash run_for_iris.sh
```

The next example uses `run.sh` and `run_common.sh` to execute tests with pre-fixed settings for multiple risk classes and multiple datasets.

```
(bdd) bash run.sh cifar10 emnist_balanced protein
```

Of course, the above examples assume the user has (via `mml` or some other route) already obtained the datasets (`iris`, `cifar10`, `emnist_balanced`, `protein`) being specified, and that `setup_data.py` has been modified such that the program knows where to find the data.


<a id="demos"></a>
## List of demos

This repository includes detailed demonstrations to walk the user through re-creating the results in the paper cited at the top of this document. Below is a list of demo links which give our demos (constructed in Jupyter notebook form) rendered using the useful <a href="https://github.com/jupyter/nbviewer">nbviewer</a> service.

- <a href="https://nbviewer.jupyter.org/github/feedbackward/bdd/blob/main/bdd/barron.ipynb">Barron dispersion function analysis</a> (section 3 in paper)
- <a href="https://nbviewer.jupyter.org/github/feedbackward/bdd/blob/main/bdd/risk_comparison-mloc.ipynb">Risk comparison</a> (section 3)
- <a href="https://nbviewer.jupyter.org/github/feedbackward/bdd/blob/main/bdd/2D_classification.ipynb">2-dim classification tests using simulated data</a> (section 5.1)
- <a href="https://nbviewer.jupyter.org/github/feedbackward/bdd/blob/main/bdd/outliers_1D_reg.ipynb">Impact of outliers (1D regression example)</a> (section 5.2)
- <a href="https://nbviewer.jupyter.org/github/feedbackward/bdd/blob/main/bdd/real_data.ipynb">Tests using real data benchmarks</a> (section 5.3)


<a id="safehash"></a>
## Safe hash value

- Replace `[safe hash mml]` with `30b0f2be3f4b755c4ac9b1170883d983dc93a5fd`.

__Date of safe hash test:__ 2022/09/30.
