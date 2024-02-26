# Introduction
This repository contains experiments on uncertainty quantification UQ for regression tasks.

# Content

First, we reproduce the findings of Lakshminarayanan et al. (Deep ensembles) on common regression tasks. The chosen UCl datasets, are used by several works in the field of UQ and allow for better comparison with existing work on UQ.

We then compare UQ from ensembles of mean,variance neural networks with ensembles of mean-only neural networks, applying post-hoc calibration on a hold out validation set.


# Install

This repository contains no data. To load the 10 UCI datasets, run the loadscript to download the datasets.

```
python ./UCI_experiments/data/load.py
```

The individual experiments can then be run from within the ` ./experiment/xx/ ` directory via the bash script

```
bash run.sh
```



# Results

We replicate the regression results of Lakshminarayanan et al. and train a MVE Neural network (predicting mean and variance with a neural network). Exactly as in the original paper we chose a one hidden layer neural network with 50 hidden neurons and ReLU activation functions and a fixed batch size of 100 samples. We use the adam optimizer and always iterate for 40 epochs over the datasets. We minimze the gaussian negative log likelihood.
5 ensemble members per committee

We notice that the absolute accuracies vary depending on minor implementation details (weight initializations, strength of the minium variance added and "gain" of the softplus function used to always have positive predicted variances.) 

Finally, the only parameter that we had to change compared to the original paper, was a significantly lower learning rate (0.01 (ours) vs 0.1 (theirs reported), for all exept the years dataset (ours 1e-3 vs 0.1 theirs)).

We also apply the same number of train test splits as in the original publication (5 for protein, 1 for Years and 20 for all other datasets.) Train and test split sizes are 90:10 % except for the years dataset, which requires a special train test split.



