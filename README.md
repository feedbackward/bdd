# bdd: bidirectional dispersion and risk function design

In this repository, we provide software and demonstrations related to the following paper:

- Risk regularization through bidirectional dispersion. Matthew J. Holland. Preprint.

This repository contains code which can be used to faithfully reproduce all the experimental results given in the above paper, and it can be easily applied to more general machine learning tasks outside the examples considered here.

## Software setup

```
$ git clone https://github.com/feedbackward/bdd.git
$ git clone https://github.com/feedbackward/mml.git
$ conda update conda
$ conda create -n bdd python=3.9 jupyter matplotlib pip pytables scipy unzip
$ conda activate bdd
(bdd) $ cd mml
(bdd) $ pip install -e ./
```
