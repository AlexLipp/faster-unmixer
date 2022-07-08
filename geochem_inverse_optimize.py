#!/usr/bin/env python3

import os
import tempfile
from typing import Any, Dict, Final, List, Tuple

import cvxpy as cp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyfastunmix

NO_DOWNSTREAM: Final[int] = 0


def add_concentrations(a: Dict[str, cp.Variable], b: Dict[str, cp.Variable]) -> Dict[str, cp.Variable]:
  """Add two dictionaries together"""
  temp: Dict[str, cp.Variable] = {}
  for e in a.keys():
    assert e in b.keys(), f"Key {e} from dictionary a was not found in dictionary b!"
    temp[e] = a[e] + b[e]
  return temp


def cp_log_ratio_norm(a, b):
  return cp.maximum(a/b, b * cp.inv_pos(a))


def nx_topological_sort_with_data(G: nx.DiGraph):
  return ((x, G.nodes[x]['data']) for x in nx.topological_sort(G))


def nx_get_downstream(G: nx.DiGraph, x: str) -> str:
  """Gets the downstream child from a node with only one child"""
  s = list(G.successors(x))
  if len(s) == 0:
    return None
  elif len(s) == 1:
    return s[0]
  else:
    raise Exception("More than one downstream neighbour!")


def plot_network(G: nx.DiGraph):
  ag = nx.nx_agraph.to_agraph(G)
  ag.layout(prog="dot")
  temp = tempfile.NamedTemporaryFile(delete=False)
  tempname = temp.name + ".png"
  ag.draw(tempname)
  img = mpimg.imread(tempname)
  plt.imshow(img)
  plt.show()
  os.remove(tempname)


def get_sample_graphs(data_dir: str) -> Tuple[nx.DiGraph, Any]:
  # Get the graph representations of the data
  sample_network_raw, sample_adjacency = pyfastunmix.fastunmix(data_dir)

  ids_to_names: Dict[int, str] = {i: data.data.name for i, data in enumerate(sample_network_raw)}

  # Convert it into a networkx graph for easy use in Python
  sample_network = nx.DiGraph()
  for x in sample_network_raw[1:]: #Skip the first node into which it all flows
    sample_network.add_node(x.data.name, data=x)
    if x.downstream_node != NO_DOWNSTREAM:
      sample_network.add_edge(x.data.name, ids_to_names[x.downstream_node])

  # Calculate the total contributing area for each sample
  for x, my_data in nx_topological_sort_with_data(sample_network):
    my_data.total_area += my_data.area
    if (ds := nx_get_downstream(sample_network, x)):
        downstream_data = sample_network.nodes[ds]['data']
        downstream_data.total_area += my_data.total_area

  return sample_network, sample_adjacency


def get_sample_data(data_filename: str) -> Tuple[pd.DataFrame, List[str]]:
  geochem_raw = pd.read_csv(data_filename, delimiter=" ")
  # Delete columns for S and Bi (too many NAs)
  geochem_raw = geochem_raw.drop(columns=['Bi', 'S'])
  # Assume that elements are columns whose names contain 1-2 characters
  element_list = [x for x in geochem_raw.columns if len(x)<=2]
  return geochem_raw, element_list


def get_primary_terms(sample_network: nx.DiGraph, obs_data: pd.DataFrame, element_list: List[str]) -> List[Any]:
  # Build the main objective
  # Use a topological sort to ensure an upstream-to-downstream traversal
  primary_terms = []
  for sample_name, my_data in nx_topological_sort_with_data(sample_network):
  #     print(f"Processing {my_data.data.name}...")

      # Set up a CVXPY parameter for each element for each node
      my_data.my_values = {e : cp.Variable(pos=True) for e in element_list}

      # area weighted contribution from this node
      my_data.my_flux = {e : my_data.area * v for e, v in my_data.my_values.items()}

      # If node already receives flux, add this node's contribution to that
      if hasattr(my_data, "total_flux"):
          my_data.total_flux = add_concentrations(my_data.my_flux, my_data.total_flux)
      # If leaf node, set the total flux leaving node as the contribution from this node only
      else:
          my_data.total_flux = my_data.my_flux

      for element, concentration in my_data.total_flux.items():
          observed = (obs_data[obs_data["Sample.Code"]==my_data.data.name][element]).values[0]
          if isinstance(observed, str):
            continue # TODO: This is here because there's a bad data point at the moment
          # Convert from a single-value pandas dataframe to a float\
          normalised_concentration = concentration/my_data.total_area
          primary_terms.append(cp_log_ratio_norm(normalised_concentration, observed))

      if (ds := nx_get_downstream(sample_network, sample_name)):
          downstream_data = sample_network.nodes[ds]['data']
          if hasattr(downstream_data, "total_flux"):
              # If downstream node already receives flux, add this nodes flux to it
              downstream_data.total_flux = add_concentrations(my_data.total_flux, downstream_data.total_flux)
          else:
              # If not, set the flux received downstream just as the contribution from this node
              downstream_data.total_flux = my_data.total_flux

  return primary_terms


def get_regularizer_terms(sample_network: nx.DiGraph, adjacency_graph) -> List[Any]:
  # Build the regularizer
  regularizer_terms = []
  # for adjacent_nodes, border_length in sample_adjacency.items():
  #   node_a, node_b = adjacent_nodes
  #   a_data = sample_network.nodes[node_a]['data']
  #   b_data = sample_network.nodes[node_b]['data']
  #   for e in a_data.my_concentrations.keys():
  #     assert e in b_data.my_concentrations.keys()
  #     a_concen = a_data.my_concentrations[e]
  #     b_concen = b_data.my_concentrations[e]
  #     regularizer_terms.append(border_length * (a_concen-b_concen))
  return regularizer_terms


def get_solution_dataframe(sample_network: nx.DiGraph, obs_data: pd.DataFrame) -> pd.DataFrame:
  # Print the solution we found
  rows = []
  for sample_name, data in sample_network.nodes(data=True):
    data = data['data']
    row = {}
    row["sample_name"] = sample_name
    for element, flux in data.total_flux.items():
      row[f"{element}_obs"] = (obs_data[obs_data["Sample.Code"]==sample_name][element]).values[0]
      if not flux.value:
        print(f"WARNING: No flux.value for {sample_name}-{element}!")
      row[f"{element}_pred"] = flux.value / data.total_area if flux.value else None
    rows.append(row)

  return pd.DataFrame(rows)


def main():
  sample_network, sample_adjacency = get_sample_graphs("data/")

  plot_network(sample_network)

  obs_data, element_list = get_sample_data("data/geochem_no_dupes.dat")

  primary_terms = get_primary_terms(sample_network=sample_network, obs_data=obs_data, element_list=element_list)

  regularizer_terms = get_regularizer_terms(sample_network=sample_network, adjacency_graph=sample_adjacency)

  if not regularizer_terms:
    print("WARNING: No regularizer terms found!")

  # Build the objective and constraints
  regularizer_strength = 1e-3
  objective = cp.sum(primary_terms)
  if regularizer_terms:
    objective += regularizer_strength * cp.norm(cp.vstack(regularizer_terms))
  constraints = []

  # Create and solve the problem
  print("Compiling and solving problem...")
  problem = cp.Problem(cp.Minimize(objective), constraints)
  # Solvers that can handle this problem type include:
  # ECOS, SCS
  # See: https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
  solvers = {
    "scip": {"solver": cp.SCIP, "verbose": True}, # VERY SLOW, probably don't use
    "ecos": {"solver": cp.ECOS, "verbose": True, "max_iters": 10000},
    "scs": {"solver": cp.SCS, "verbose": True, "max_iters": 10000},
  }
  objective_value = problem.solve(**solvers["ecos"])
  print(f"Status = {problem.status}")
  print(f"Objective value = {objective_value}")

  soldf = get_solution_dataframe(sample_network=sample_network, obs_data=obs_data)

  print(soldf)


if __name__ == "__main__":
  main()