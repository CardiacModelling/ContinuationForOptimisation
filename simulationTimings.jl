using Parameters, Plots
using BifurcationKit
using DifferentialEquations
using ConstructionBase
using Revise
using Setfield
using BenchmarkTools
const BK = BifurcationKit
include("./model.jl")
using .Model

# ODE Benchmark
prob = ODEProblem(Model.ode!, Model.ic, (0.0, 50000.0), Model.params, abstol=1e-10, reltol=1e-8)
sol = solve(prob, Tsit5(), maxiters=1e7)
plot(sol, idxs=Model.slow_idx)
@benchmark solve(prob, Tsit5(), maxiters=1e7, save_everystep = false)
    # BenchmarkTools.Trial: 4 samples with 1 evaluation.
    # Range (min … max):  1.389 s …   1.445 s  ┊ GC (min … max): 0.00% … 0.00%
    # Time  (median):     1.423 s              ┊ GC (median):    0.00%
    # Time  (mean ± σ):   1.420 s ± 24.393 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

    # █                       █                  █            █
    # █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
    # 1.39 s         Histogram: frequency by time        1.45 s <

    # Memory estimate: 7.56 KiB, allocs estimate: 65.



# Continuation Benchmark
# Look at times to do a +- 10% change in each parameter and average results across a few parameters (only conductances)
# Bifurcation Problem
lens = Model.cont_params[1]
pVal = Setfield.get(Model.params, lens)
bp = BifurcationProblem(Model.ode!, Model.ic_conv, Model.params, lens;
	record_from_solution = (x, p) -> (V = x[plot_idx]),)

argspo = (record_from_solution = (x, p) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[plot_idx,:]),
				min = minimum(xtt[plot_idx,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[plot_idx,:]; label = "V", k...)
	end)

# 1 pulse solution
prob = remake(prob, u0 = Model.ic_conv, tspan=(0.0, 20.0))
sol_pulse = solve(prob, Rodas5())

bptrap, ci = BK.generate_ci_problem(PeriodicOrbitTrapProblem(M = 150),
bp, sol_pulse, 20.0)

opts_br = ContinuationPar(p_min = pVal*0.9, p_max = pVal*1.1, max_steps = 50, tol_stability = 1e-8, ds=0.1*pVal, dsmax=0.1*pVal, 
detect_bifurcation=0, detect_fold=false, newton_options=NewtonPar(verbose=true))
brpo_fold = continuation(bptrap, ci, PALC(), opts_br;
	verbosity = 3, plot = true,
	argspo...
)

# Verify that s is converged - mostly, but could probably get closer
ic = brpo_fold.sol[2].x[1:5]
p = Model.params
p = @set p.gna = 132.0
prob = remake(prob, u0 = ic, tspan=(0.0, 50000.0), p=p)
sol = solve(prob, Rodas5())
plot(sol, idxs=slow_idx)
display(title!("Plot of slow variable from continuation"))

reducedOpts = ContinuationPar(p_min = pVal*0.9, p_max = pVal*1.1, max_steps = 50, tol_stability = 1e-8, ds=0.1*pVal, dsmax=0.1*pVal, 
detect_bifurcation=0, detect_fold=false,)
@benchmark continuation($bptrap, $ci, $PALC(), $reducedOpts)
    # BenchmarkTools.Trial: 22 samples with 1 evaluation.
    # Range (min … max):  190.612 ms … 443.830 ms  ┊ GC (min … max):  0.00% … 25.17%
    # Time  (median):     221.913 ms               ┊ GC (median):    10.89%
    # Time  (mean ± σ):   232.739 ms ±  51.737 ms  ┊ GC (mean ± σ):   9.46% ±  7.75%

    # ▁  ▁▁▁ █     ▁
    # █▁▆███▆█▁▆▆▆▁█▆▁▁▁▁▁▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆ ▁
    # 191 ms           Histogram: frequency by time          444 ms <

    # Memory estimate: 230.38 MiB, allocs estimate: 376687.




# How long to it take to converge for ODE small step?
prob = remake(prob, u0 = Model.ic_conv, tspan=(0.0, 50000.0))
sol = solve(prob, Tsit5(), maxiters=1e7)
plot(sol, idxs=slow_idx)
xlabel!("Time (ms)")
ylabel!("Slow variable")
display(title!("ODE from previous converged IC (gna: 120 -> 132)"))
# Converged for the small change after t=30,000ms
prob = remake(prob, tspan=(0.0, 30000.0))
@benchmark solve($prob, $Tsit5(), maxiters=1e7, save_everystep = false)
    # BenchmarkTools.Trial: 7 samples with 1 evaluation.
    # Range (min … max):  601.395 ms …    1.155 s  ┊ GC (min … max): 0.00% … 0.00%
    # Time  (median):     650.410 ms               ┊ GC (median):    0.00%
    # Time  (mean ± σ):   754.617 ms ± 215.762 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

    # ▁ █  ▁▁                                ▁                    ▁
    # █▁█▁▁██▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
    # 601 ms           Histogram: frequency by time          1.15 s <

    # Memory estimate: 7.56 KiB, allocs estimate: 65.



