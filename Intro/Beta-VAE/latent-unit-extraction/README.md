# Latent Variable Mapping

Scripts to generate latent variables, and perform latent variable mapping. Figure illustrates the latent variable mapping heuristic

<p align="center">
   <img src="https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/figures/latent-mapping.png" align="center" >
</p>

LP - Partition latent variables with high variance in average KL-divergence\
Ld - Detector latent variables\
Lf - Reasoner Latent variables

```

python3 latent-csv-generator.py    --use the trained B-VAE weights to generate csv with latent variable parameters (mean, logvar, samples).

python3 latent-unit-comparison.py  --generate csv with average kl-divergence of each latent variables for different scenes in a partition.

python3 latent-unit-selection.py   --uses Welford's variance calculator to return latent variables Ld and Lf.

python3 latent-plotter.py      -- script to scatter plot induvidual latent variables. 

```
