# Lag Selection for Forecasting using Deep Learning: An Empirical Study

This repository contains a set of experiments for studying the number of lags in global forecasting models based on deep learning.


## Reproducing experiments

Run the following scripts on your console:
- scripts/experiments/cv_inner.py - inner cross-validation cycle for lag selection
- scripts/experiments/cv_outer.py - outer cross-validation cycle for evaluation
- scripts/experiments/lag_selection.py - selecting lags based on validation data and inner cv results

Then, you can analyse the results with the scripts in scripts/analysis