from matplotlib import pyplot as plt
import numpy as np
import utils

plotter1 = utils.Plotter(fontsize=20, figsize=(9,6))

plotter1.drift_plot('./npz_saved')

plotter2 = utils.Plotter(fontsize=20, figsize=(8,6))
plotter2.replay_plots('./npz_saved')
