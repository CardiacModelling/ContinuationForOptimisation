# ContinuationForOptimisation
Using limit cycle continuation methods to speed up evaluation of ion channel cell models

# Using the code
The code should be cloned from GitHub, or downloaded from Zenodo.

# Setting up the Julia project
For up to date versions of the dependent packages, run the *packages.jl* file.

To use the same package versions as used in the paper, refer to the *Project.toml* and *Manifest.toml* and use *Pkg.instantiate()*.

# Running the analysis
The analysis from the paper can be run by running the scripts *simulationTimings.jl* and *mhmcmc.jl*.

Note that running the scripts will override the previously recorded results.

When running *mhmcmc.jl*, there are a handful of changes you may wish to make. 
The number of samples to run the MCMC algorithm for can be changed from the default of 50,000.
The "converger" can be chosen by setting the variables *use_continuation* and *use_fast_ODE*. 
If `use_continuation=true`, then the continuation converger will be used.
Otherwise, the ODE solver will be used to converge, running for 10,000ms if `use_fast_ODE=true` or 50,000ms otherwise.

# Reproducing the figures from the pre-generated data
The figures for the benchmark results can be loaded built from the *.json* files.
The files can be loaded into a Julia workspace using:
```
using BenchmarkTools, Plots, BenchmarkPlots, StatsPlots
t = BenchmarkTools.load("simulation_timings.json")[1]
```
Then, the plotting code from the end of the *simulationTimings.jl* can be run.
