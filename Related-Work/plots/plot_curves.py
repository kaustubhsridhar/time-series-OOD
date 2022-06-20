from matplotlib import pyplot as plt
import numpy as np
import utils

plotter1 = utils.Plotter(fontsize=22, figsize=(9,6))

plotter1.drift_plot('./npz_saved')
plotter1.drift_cl_ablation('./npz_saved')

# plotter2 = utils.DoublePlotter(fontsize=22, figsize=(30,5))
plotter2 = utils.Plotter(fontsize=22)
plotter2.replay_plots('./npz_saved')
