using Parameters, Plots
using BifurcationKit
using DifferentialEquations
using ConstructionBase, LinearAlgebra
using ModelingToolkit, CellMLToolkit
const BK = BifurcationKit

ml = CellModel("tentusscher_noble_noble_panfilov_2004_a.cellml")
# Use list_states(ml) to find membrane voltage index
list_states(ml)
idx_V = 12
tspan = (0, 5000.0)
prob = ODEProblem(ml, tspan)
sol = solve(prob, Rodas5(), dtmax=1.)
plot(sol, idxs=idx_V)
display(title!("Membrane voltage for ten Tusscher 2004"))

# Plot concentrations
idx_concs = [1, 4, 16, 17]
concs = (sol[idx_concs,:]./sol[idx_concs,1])
plot(sol.t, concs')
display(title!("Normalised concentrations for ten Tusscher 2004"))

n = map(x -> norm(x), eachcol(concs))
plot(sol.t, n)
display(title!("Norm of normalised concentrations for ten Tusscher 2004"))
# Concentrations slowly moving towards a steady state (limit cycle) somewhere

# Run long simulation
prob = remake(prob; tspan=(0, 100000.0))
sol = solve(prob, Rodas5(), dtmax=1.)
concs = (sol[idx_concs,:]./sol[idx_concs,1])
n = map(x -> norm(x), eachcol(concs))
plot(sol.t, n)
display(title!("Norm of conc for long run time"))

# Check converged by getting 1 pulse
prob = remake(prob; tspan=(0, 10000.0), u0=sol[end])
sol = solve(prob, Rodas5(), dtmax=1.)
concs = (sol[idx_concs,:]./sol[idx_concs,1])
plot(sol.t, concs')
display(title!("Normalised concentrations at limit cycle"))

