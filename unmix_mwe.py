#!/usr/bin/env python3

"""
This script is a minimum working example for how to unmix the downstream observations for a specific element.
It treats each sub-basin defined by a sample as a discrete variable, and solves for the upstream concentrations of the element in each sub-basin.

It loads a sample network graph and observations from data files, visualizes the network, and builds the optimization problem.
Then, it performs a sweep of different regularization strengths for the problem and visualizes the results.
Next, it solves the problem using a specified solver and regularization strength, obtaining predicted downstream and upstream concentrations.
The script also calculates unique upstream areas for each basin in the network and generates an upstream concentration map based on the predictions.
Finally, it visualizes the predicted downstream concentrations and the upstream concentration map for the specified element (default: Mg).
"""


import matplotlib.pyplot as plt
import pandas as pd

import geochem_inverse_optimize as gio

# Constants
element = "Mg"  # Set element
regularizer_strength = 10 ** (-3.0)

# Load sample network
sample_network, _ = gio.get_sample_graphs(
    flowdirs_filename="data/d8.asc",
    sample_data_filename="data/sample_data.dat",
)

# Load in observations
obs_data = pd.read_csv("data/sample_data.dat", delimiter=" ")
obs_data = obs_data.drop(columns=["Bi", "S"])

plt.figure(figsize=(15, 10))  # Visualise network
gio.plot_network(sample_network)
plt.show()
print("Building problem...")
problem = gio.SampleNetworkUnmixer(sample_network=sample_network)


element_data = gio.get_element_obs(
    element, obs_data
)  # Return dictionary of {sample_name:concentration}

gio.plot_sweep_of_regularizer_strength(problem, element_data, -5, -1, 11)

print("Solving problem...")
element_pred_down, element_pred_upstream = problem.solve(
    element_data, solver="ecos", regularization_strength=10 ** (-3)
)

area_dict = gio.get_unique_upstream_areas(sample_network)  # Extract areas for each basin
upstream_map = gio.get_upstream_concentration_map(
    area_dict, element_pred_upstream
)  # Assign to upstream preds

# Visualise outputs downstream
gio.visualise_downstream(pred_dict=element_pred_down, obs_dict=element_data, element=element)
plt.show()
# Visualise outputs upstream
plt.imshow(upstream_map)
cb = plt.colorbar()
cb.set_label(element + "concentration mg/kg")
plt.title("Upstream Concentration Map")
plt.show()
