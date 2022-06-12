#!/usr/bin/env python3

import pyfastunmix
import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict

def add_concentrations(a: Dict[str, cp.Variable], b: Dict[str, cp.Variable]) -> Dict[str, cp.Variable]:
  """Add two dictionaries together"""
  temp: Dict[str, cp.Variable] = {}
  for e in a.keys():
    assert e in b.keys(), f"Key {e} from dictionary a was not found in dictionary b!"
    temp[e] = a[e] + b[e]
  return temp

# Get the graph representations of the data
sample_network, sample_adjacency = pyfastunmix.fastunmix("data/")

# Convert it into a networkx graph for easy use in Python
G = nx.DiGraph()
for i, node_data in enumerate(sample_network):
  if i==0:
    continue
  G.add_node(i, data=node_data)
  if node_data.downstream_node != 0:
    G.add_edge(i, node_data.downstream_node)

elements = ['Li', 'Be', 'Na', 'Mg', 'Al', 'P', 'K', 'Ca', 'Ti', 'V',
       'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'As', 'Se', 'Rb', 'Sr',
       'Y', 'Zr', 'Nb', 'Mo', 'Ag', 'Cd', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce',
       'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
       'Hf', 'Ta', 'W', 'Tl', 'Pb', 'Th', 'U']

# Code from the notebook
geochem_raw = np.loadtxt('data/geochem.dat',dtype=str) # Read in data
geochem_raw = np.delete(geochem_raw,[7,53],1) # Delete columns for S and Bi (too many NAs)
elems = geochem_raw[0,1:54] # List of element strings
obs_data = pd.DataFrame(geochem_raw[1:,],columns=geochem_raw[0,:]) # Cast to DataFrame for quick access
obs_data[elems]=obs_data[elems].astype(float) # Cast numeric data to float

# Build the main objective
# Use a topological sort to ensure upstream-to-downstream motion.
primary_terms = []
for x in nx.topological_sort(G):
  my_data = G.nodes[x]['data']
  print(f"Processing {my_data.data.name}...")
  my_data.my_values = {e : cp.Variable(pos=True) for e in elements}
  my_data.my_concentrations = {e : my_data.area * v for e, v in my_data.my_values.items()}
  if hasattr(my_data, "total_concentrations"):
    my_data.total_concentrations = add_concentrations(my_data.my_concentrations, my_data.total_concentrations)
  else:
    my_data.total_concentrations = my_data.my_concentrations

  for element, concentration in my_data.total_concentrations.items():
    observed = float(obs_data[obs_data["Sample.Code"]==my_data.data.name][element])
    primary_terms.append(concentration - observed)

  if my_data.downstream_node!=0:
    downstream_data = G.nodes[my_data.downstream_node]['data']
    if hasattr(downstream_data, "total_concentrations"):
      downstream_data.total_concentrations = add_concentrations(my_data.total_concentrations, downstream_data.total_concentrations)
    else:
      downstream_data.total_concentrations = my_data.total_concentrations

# Build the regularizer
regularizer_terms = []
for adjacent_nodes, border_length in sample_adjacency.items():
  node_a, node_b = adjacent_nodes
  a_data = G.nodes[node_a]['data']
  b_data = G.nodes[node_b]['data']
  for e in a_data.my_concentrations.keys():
    assert e in b_data.my_concentrations.keys()
    a_concen = a_data.my_concentrations[e]
    b_concen = b_data.my_concentrations[e]
    regularizer_terms.append(border_length * (a_concen-b_concen))

if not regularizer_terms:
  print("WARNING: No regularizer terms found!")

# Build the objective and constraints
regularizer_strength = 1e-3
objective = cp.norm(cp.vstack(primary_terms))
if regularizer_terms:
  objective += regularizer_strength * cp.norm(cp.vstack(regularizer_terms))
constraints = []

# Create and solve the problem
print("Compiling and solving problem...")
problem = cp.Problem(cp.Minimize(objective), constraints)
objective_value = problem.solve(verbose=True)

# Print the solution we found
print(f"Problem status = {problem.status}")
print(f"Objective value = {objective_value}")
for x in G.nodes:
  data = G.nodes[x]['data']
  if not hasattr(data, "my_values"):
    continue
  for element, concentration in data.my_values.items():
    print(f"\t{element}: {concentration.value}")