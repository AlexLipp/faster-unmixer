#!/usr/bin/env python3

import pyfastunmix
import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict

# Get the graph representation of the data
data = pyfastunmix.fastunmix("data/")

# Convert it into a networkx graph for easy use in Python
G = nx.DiGraph()
for i, x in enumerate(data):
  if i==0:
    continue
  G.add_node(i, data=x)
  if x.parent!=0:
    G.add_edge(i, x.parent)

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

def add_concentrations(a: Dict[str, cp.Variable], b: Dict[str, cp.Variable]) -> Dict[str, cp.Variable]:
  return {element: value + b[element] for element, value in a.items()}

# Use a topological sort to ensure upstream-to-downstream motion
l2_terms = []
for x in nx.topological_sort(G):
  my_data = G.nodes[x]['data']
  print(f"Processing {my_data.data.name}...")
  my_data.my_values = {e : cp.Variable(pos=True) for e in elements}
  my_data.my_concentrations = {e : my_data.area * v for e,v in my_data.my_values.items()}
  if hasattr(my_data, "total_concentrations"):
    my_data.total_concentrations = add_concentrations(my_data.my_concentrations, my_data.total_concentrations)
  else:
    my_data.total_concentrations = my_data.my_concentrations

  for element, concentration in my_data.total_concentrations.items():
    observed = float(obs_data[obs_data["Sample.Code"]==my_data.data.name][element])
    l2_terms.append(concentration - observed)

  if my_data.parent!=0:
    parent_data = G.nodes[my_data.parent]['data']
    if hasattr(parent_data, "total_concentrations"):
      parent_data.total_concentrations = add_concentrations(my_data.total_concentrations, parent_data.total_concentrations)
    else:
      parent_data.total_concentrations = my_data.total_concentrations

print("Building problem...")
objective = cp.Minimize(cp.norm(cp.vstack(l2_terms)))
constraints = []
problem = cp.Problem(objective, constraints)
objective_value = problem.solve()

for x in G.nodes:
  data = G.nodes[x]['data']
  if not hasattr(data, "my_values"):
    continue
  for element, concentration in data.my_values.items():
    print(f"\t{element}: {concentration.value}")