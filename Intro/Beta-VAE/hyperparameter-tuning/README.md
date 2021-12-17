# Latent Space Encoding for B-VAE Hyperparameter Selection

Scripts here are used to select hyperparameters, number of latent units (n) and B. Three hyperparameter tuning algorithms that are compared in our work.

(1) Bayesian Optimization, (2) Random Search, and (3) Grid Search

To run the Bayesian optimization search using MIG as the optimization criteria, run the following. 

```
python3 bayes_mig.py              ---Bayesian optimization
```

To run other search algorithms using MIG as the optimization criteria, run the following.

```
python3 random_mig.py            ---random search

python3 grid_mig.py              ---grid search
```

To run other search algorithms using Elbo as the optimization criteria, run the following.

```
python3 bayes_elbo.py          ---bayesian optimization

python3 random_elbo.py        ---random search

python3 grid_elbo.py          ---grid search

```
