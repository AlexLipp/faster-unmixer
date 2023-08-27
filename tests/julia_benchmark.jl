using Gurobi
using Convex
using Random

MINIMUM_CONC = 1.
MAXIMUM_CONC = 1e2

# Set the range to explore sub-basin area values over
MINIMUM_AREA = 1.
MAXIMUM_AREA = 1e2

# Tree properties
BRANCHING_FACTOR = 3
HEIGHT = 10
TOTAL_NODES = BRANCHING_FACTOR ^ HEIGHT

TEST_REPEATS = 10

function log_uniform(n::Int, min::Float64, max::Float64)
  @assert min > 0 && max > 0 && min < max
  unif = exp.(log(min) .+ (log(max)-log(min)) .* rand(n))
end

function traverse_tree(index::Int, total_nodes::Int, branching_factor::Int, end_func, reduction_func)
  # Out-of-bounds check; end recursion
  if index > total_nodes
      return nothing
  end

  # Check if leaf node
  is_leaf = true
  children_values = []
  for childi in 1:branching_factor
      child_index = branching_factor * (index - 1) + childi + 1  # Assuming 1-based indexing for root
      child_value = traverse_tree(child_index, total_nodes, branching_factor, end_func, reduction_func)
      if child_value !== nothing
          is_leaf = false
          push!(children_values, child_value)
      end
  end

  # If it's a leaf, call end_func
  if is_leaf
      return end_func(index)
  else
      # Otherwise, call reduction_func on the children's values
      return reduction_func(index, children_values)
  end
end

areas = log_uniform(TOTAL_NODES, MINIMUM_AREA, MAXIMUM_AREA)
concentrations = log_uniform(TOTAL_NODES, MINIMUM_CONC, MAXIMUM_CONC)
my_total_flux = zeros(TOTAL_NODES)
my_total_tracer_flux = zeros(TOTAL_NODES)
predictions = zeros(TOTAL_NODES)

# Get summed areas
traverse_tree(
  1,
  TOTAL_NODES,
  BRANCHING_FACTOR,
  (index) -> my_total_flux[index] = areas[index],
  (index, children_values) -> my_total_flux[index] = sum(children_values) + areas[index]
)

# Get tracer fluxes
traverse_tree(
  1,
  TOTAL_NODES,
  BRANCHING_FACTOR,
  (index) -> my_total_tracer_flux[index] = my_total_flux[index] * concentrations[index],
  (index, children_values) -> my_total_tracer_flux[index] = sum(children_values) + my_total_flux[index] * concentrations[index]
)

objective = traverse_tree(
  1,
  TOTAL_NODES,
  BRANCHING_FACTOR,
  (index) -> begin my_modeled_tflux = Variable(Positive());my_modeled_tflux end,
  (index, children_values) -> begin my_modeled_tflux = Variable(Positive()); sum(children_values) + my_modeled_tflux end
)

problem = minimize(objective)

for i in 1:TEST_REPEATS
  @time solve!(problem, Gurobi.Optimizer)
end