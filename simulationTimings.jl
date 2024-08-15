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
b = @benchmarkable solve($prob, $Tsit5(), maxiters=1e7, save_everystep = false)
run(b, samples=10, seconds=300)
    # BenchmarkTools.Trial: 10 samples with 1 evaluation.
    # Range (min … max):  944.588 ms …   1.025 s  ┊ GC (min … max): 0.00% … 0.00%
    # Time  (median):     985.782 ms              ┊ GC (median):    0.00%
    # Time  (mean ± σ):   983.557 ms ± 27.565 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

    # █             ▁     ▁      ▁      ▁   █                ▁   ▁
    # █▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁█ ▁
    # 945 ms          Histogram: frequency by time          1.03 s <

    # Memory estimate: 7.56 KiB, allocs estimate: 65.



# Continuation Benchmark
# TODO: try other continuation methods (shooting and oColl)
# Look at times to do a +- 10% change in each parameter and average results across a few parameters (only conductances)
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

# Verify that s is converged
ic = brpo_fold.sol[2].x[1:5]
p = Model.params
p = @set p.gna = 132.0
prob = remake(prob, u0 = ic, tspan=(0.0, 50000.0), p=p)
sol = solve(prob, Rodas5())
plot(sol, idxs=slow_idx)
display(title!("Plot of slow variable from continuation"))

reducedOpts = ContinuationPar(p_min = pVal*0.9, p_max = pVal*1.1, max_steps = 50, tol_stability = 1e-8, ds=0.1*pVal, dsmax=0.1*pVal, 
detect_bifurcation=0, detect_fold=false,)
b = @benchmarkable continuation($bptrap, $ci, $PALC(), $reducedOpts)
run(b, samples=50, seconds=300)
    # BenchmarkTools.Trial: 50 samples with 1 evaluation.
    # Range (min … max):  184.012 ms …    1.468 s  ┊ GC (min … max):  0.00% … 83.09%
    # Time  (median):     212.174 ms               ┊ GC (median):     0.00%
    # Time  (mean ± σ):   258.983 ms ± 180.944 ms  ┊ GC (mean ± σ):  17.66% ± 14.63%

    # █    
    # █▄▄▅▆▅▃▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
    # 184 ms           Histogram: frequency by time          1.47 s <

    # Memory estimate: 230.37 MiB, allocs estimate: 376537.
# Can't seem to get rid of the outlier



# How long to it take to converge for ODE small step?
prob = remake(prob, u0 = Model.ic_conv, tspan=(0.0, 50000.0))
sol = solve(prob, Tsit5(), maxiters=1e7)
plot(sol, idxs=slow_idx)
xlabel!("Time (ms)")
ylabel!("Slow variable")
display(title!("ODE from previous converged IC (gna: 120 -> 132)"))
# Converged for the small change after t=30,000ms
prob = remake(prob, tspan=(0.0, 30000.0))
b = @benchmarkable solve($prob, $Tsit5(), maxiters=1e7, save_everystep = false)
run(b, samples=50, seconds=300)
    # BenchmarkTools.Trial: 50 samples with 1 evaluation.
    # Range (min … max):  564.056 ms …    1.584 s  ┊ GC (min … max): 0.00% … 0.00%
    # Time  (median):     732.226 ms               ┊ GC (median):    0.00%
    # Time  (mean ± σ):   852.892 ms ± 271.111 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

    #     ▆ ▁█    ▁      
    # ▄▇▄█▇██▇▇▄▄█▁▄▁▇▄▁▁▄▁▁▁▄▇▄▁▁▁▇▁▄▁▁▁▄▁▁▁▄▁▄▁▄▄▄▁▄▁▁▁▁▁▁▁▁▁▁▁▄▄ ▁
    # 564 ms           Histogram: frequency by time          1.58 s <

    # Memory estimate: 7.56 KiB, allocs estimate: 65.



