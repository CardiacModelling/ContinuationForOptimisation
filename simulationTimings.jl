using Plots, BenchmarkPlots, StatsPlots
using BifurcationKit, DifferentialEquations
using BenchmarkTools
const BK = BifurcationKit
include("./model.jl")
using .Model
include("./tools.jl")
using .Tools

# Define BenchmarkGroup
bg = BenchmarkGroup()
bg["ODE"] = BenchmarkGroup()
bg["Cont"] = BenchmarkGroup()

# Plot parameters
plot_params = (linewidth=2., dpi=300, size=(450,300), legend=false)

# ODE Benchmark
prob = ODEProblem(Model.ode!, Model.ic, (0.0, 50000.0), Model.params, abstol=1e-10, reltol=1e-8)
sol = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e7)
plot(sol, idxs=Model.slow_idx, xlabel = "Time (ms)", ylabel = "Slow Variable", title="ODE Convergence"; plot_params...)
savefig("ode_converge.pdf")
b = @benchmarkable DifferentialEquations.solve($prob, $Tsit5(), maxiters=1e7, save_everystep = false)
bg["ODE"]["ODE - Full"] = b

# How long to it take to converge for ODE small step?
prob = remake(prob, u0 = Model.ic_conv, tspan=(0.0, 50000.0))
sol = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e7)
plot(sol, idxs=Model.slow_idx, xlabel = "Time (ms)", ylabel = "Slow Variable", title = "ODE (gna: 120 -> 132)"; plot_params...)
savefig("ode_param_change.pdf")
converged = Tools.auto_converge_check(prob, sol(30000), Model.params)
println("ODE Small Step Converged: ", converged)
# Converged for the small change after t=30,000ms
prob = remake(prob, tspan=(0.0, 30000.0))
b = @benchmarkable DifferentialEquations.solve($prob, $Tsit5(), maxiters=1e7, save_everystep = false)
bg["ODE"]["ODE - Small Step"] = b

# Continuation Benchmark
# Look at times to do a +- 10% change in each parameter and average results across a few parameters (only conductances)
lens = @optic _.gna
pVal = lens(Model.params_cont)
bp = BifurcationProblem(Model.ode_cont!, Model.ic_conv, Model.params_cont, lens;
	record_from_solution = (x, p) -> (V = x[Model.plot_idx]),)

argspo = (record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[Model.plot_idx,:]),
				min = minimum(xtt[Model.plot_idx,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[Model.plot_idx,:]; label = "V", k...)
	end)

# 1 pulse solution
prob_cont = ODEProblem(Model.ode_cont!, Model.ic_conv, (0.0, 20.0), Model.params_cont, abstol=1e-10, reltol=1e-8)
sol_pulse = DifferentialEquations.solve(prob_cont, Tsit5())

# Trapezoidal method
bptrap, ci = BK.generate_ci_problem(PeriodicOrbitTrapProblem(M = 150),
bp, sol_pulse, 20.0)

opts_br = ContinuationPar(p_min = pVal*0.9, p_max = pVal*1.1, max_steps = 50, tol_stability = 1e-8, ds=0.1*pVal, dsmax=0.1*pVal, 
detect_bifurcation=0, detect_fold=false, newton_options=NewtonPar(verbose=true))
brpo_trap = continuation(bptrap, ci, PALC(), opts_br;
	verbosity = 3, plot = true,
	argspo...
)

# Orthogonal collocation method
bpoc, cioc = BK.generate_ci_problem(PeriodicOrbitOCollProblem(30, 4),
bp, sol_pulse, 20.0)

brpo_oc = continuation(bpoc, cioc, PALC(), opts_br;
	verbosity = 3, plot = true,
	argspo...
)

# Shooting method
bpsh, cish = BK.generate_ci_problem(ShootingProblem(M=1),
bp, prob_cont, sol_pulse, 20.0; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
	verbosity = 3, plot = true,
	argspo...
)

# Verify that s is converged
p = Model.params
p = @set p.gna = 132.0

function check_converged(prob, ic, p, slow_idx, name="")
    prob = remake(prob, u0 = ic, tspan=(0.0, 50000.0), p=p)
    sol = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e7)
    plot(sol, idxs=slow_idx, title = "Continuation: "*name, xlabel = "Time (ms)", ylabel = "Slow Variable"; plot_params...)
	savefig("continuation_convergence_"*name*".pdf")
	check_converged = Tools.auto_converge_check(prob, ic, p)
	println(name, " Converged: ", check_converged)
end

check_converged(prob, brpo_trap.sol[2].x[1:5], p, Model.slow_idx, "Trap")
check_converged(prob, brpo_oc.sol[2].x[1:5], p, Model.slow_idx, "OColl")
check_converged(prob, brpo_sh.sol[2].x[1:5], p, Model.slow_idx, "Shooting")

reducedOpts = ContinuationPar(p_min = pVal*0.9, p_max = pVal*1.1, max_steps = 50, tol_stability = 1e-8, ds=0.1*pVal, dsmax=0.1*pVal, 
detect_bifurcation=0, detect_fold=false,)
b = @benchmarkable continuation($bptrap, $ci, $PALC(), $reducedOpts)
bg["Cont"]["Cont - Trap"] = b

b = @benchmarkable continuation($bpoc, $cioc, $PALC(), $reducedOpts)
bg["Cont"]["Cont - OColl"] = b

b = @benchmarkable continuation($bpsh, $cish, $PALC(), $reducedOpts)
bg["Cont"]["Cont - Shooting"] = b

t = run(bg, seconds=120)
plot(t["ODE"], yaxis=:log10, dpi=300, size=(450,300), title="ODE vs Continuation Timings")
plot!(t["Cont"], yaxis=:log10, linestyle=:dot, legend=:bottomleft, xaxis=nothing, ylabel="Time (ms)", yformatter=x->x/1e6)
savefig("simulation_timings.pdf")
BenchmarkTools.save("simulation_timings.json", t)
