#!/usr/bin/env python3

"""
This script is a minimum working example for how to unmix the downstream observations for a specific element seeking a smooth continuous solution.
It solves for the smoothest continusous upstream concentration map of the element that fits the observations downstream.

It loads a sample network graph and observations from data files, sets constants such as the element and regularization strength, and visualizes the network.
Next, it builds the continuous optimization problem by creating a sample network, specifying area labels, and setting other parameters.
The script performs a sweep of different regularization strengths for the problem and visualizes the results.
Then, it solves the problem using a specified solver and regularization strength, obtaining predicted downstream concentrations and an upstream concentration map.
The script visualizes the upstream concentration map and the predicted downstream concentrations for the specified element.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geochem_inverse_optimize as gio

# Constants
element = "Mg"  # Set element
regularizer_strength = 10 ** (-0.8)

# Load sample network
sample_network, _ = gio.get_sample_graphs(
    flowdirs_filename="data/d8.asc",
    sample_data_filename="data/sample_data.dat",
)

# Load in observations
obs_data = pd.read_csv("data/sample_data.dat", delimiter=" ")
obs_data = obs_data.drop(columns=["Bi", "S"])

area_map = plt.imread("labels.tif")[:, :, 0]

print("Building problem...")
problem = gio.SampleNetwork(
    sample_network=sample_network, ny=60, nx=60, area_labels=area_map, continuous=True
)
element_data = gio.get_element_obs(
    element, obs_data
)  # Return dictionary of {sample_name:concentration}

gio.plot_sweep_of_regularizer_strength(problem, element_data, -2, 2, 11)

print("Solving problem...")
down_dict, upstream_map = problem.solve(
    element_data, regularization_strength=regularizer_strength, solver="ecos"
)

print("Visualising output...")
plt.imshow(upstream_map)
plt.colorbar()
plt.show()

gio.visualise_downstream(pred_dict=down_dict, obs_dict=element_data, element=element)
plt.show()
