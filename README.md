# ContinuationForOptimisation
Using limit cycle continuation methods to speed up evaluation of ion channel cell models

# Using the code
The code can be cloned from GitHub (most recent version), or downloaded from Zenodo (version of record).

# Setting up the Julia project
For up to date versions of the dependent packages, run the *packages.jl* file.

To use the same package versions as used in the paper, refer to the *Project.toml* and *Manifest.toml* and use *Pkg.instantiate()*.

Note the figures were generated under a different environment to the one given in the *Project.toml* and *Manifest.toml* files, which were only used to generated the results/data.

# Running the analysis
The analysis from the paper can be run by running the scripts *simulationTimings.jl*, *mcmcSetup.jl* and *mhmcmc.jl*.

Note that running the scripts will override the previously recorded results.

When running *mhmcmc.jl*, there are a handful of changes you may wish to make. 
The number of samples to run the MCMC algorithm for can be changed from the default of 40,000.
The "converger" can be chosen by setting the variables *use_continuation* and *use_tracking_ODE*. 
If `use_continuation=true`, then the continuation converger will be used.
Otherwise, the ODE solver will be used to converge, running from a previous limit cycle if `use_tracking_ODE=true` or from standard initial conditions otherwise.

# Reproducing the figures from the pre-generated data
The figures can be regenerated using *figures.jl* (for results figures) and *diagrams.jl* (for non-results figures such as the phase plane diagrams).
