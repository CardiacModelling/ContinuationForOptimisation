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
bg["Small"] = BenchmarkGroup()
bg["Large"] = BenchmarkGroup()
bg["Small"]["ODE"] = BenchmarkGroup()
bg["Small"]["Cont"] = BenchmarkGroup()
bg["Large"]["ODE"] = BenchmarkGroup()
bg["Large"]["Cont"] = BenchmarkGroup()

# Plot parameters
plot_params = (linewidth=2., dpi=300, size=(450,300), legend=false)

function convergence_plot(ic, prob, title, filename)
	sol = DifferentialEquations.solve(prob, Tsit5(), tspan=(0.0, 200.0), maxiters=1e9, u0 = ic)
	plot(sol, idxs=Model.slow_idx[1], xlabel = "Time (ms)", ylabel = "Nai", title=title; plot_params...)
	savefig(filename*"-nai.pdf")
	plot(sol, idxs=Model.slow_idx[2], xlabel = "Time (ms)", ylabel = "Ki", title=title; plot_params...)
	savefig(filename*"-ki.pdf")
	println(title, "    ", filename, "    ", Tools.auto_converge_check(prob, ic, prob.p))
end

# ODE Convergence
tmp = Model.params
pSmall = @set tmp.g_Na_sf = 1.1
pLarge = @set tmp.g_Na_sf = 1.5
pLarge = @set pLarge.g_K_sf = 1.2
pLarge = @set pLarge.g_L_sf = 0.8

prob = ODEProblem(Model.ode!, Model.ic, (0.0, 250.0), tmp, abstol=1e-10, reltol=1e-8)

# ODE Convergence - Full
params = [pSmall, pLarge]
tspans = [(0.0,300.0), (0.0, 250.0)]
for i in eachindex(params)
	p = params[i]
	prob_de = remake(prob, p=p)
	sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan = tspans[i], maxiters=1e9, save_everystep=false)
	convergence_plot(sol[end], prob_de, "From Converged State: Full ODE", 
	"results/simTimings/"*(i==1 ? "small" : "large")*"Step/convergence/ode-full")
	b = @benchmarkable DifferentialEquations.solve($prob_de, $Tsit5(), maxiters=1e9, save_everystep = false)
	bg[i==1 ? "Small" : "Large"]["ODE"]["ODE - Full"] = b
end

# ODE Convergence - 1000s
params = [pSmall, pLarge]
for i in eachindex(params)
	p = params[i]
	prob_de = remake(prob, p=p)
	sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan = (0.0, 1000.0), maxiters=1e9, save_everystep=false)
	convergence_plot(sol[end], prob_de, "From Converged State: Full ODE 1000sec", 
	"results/simTimings/"*(i==1 ? "small" : "large")*"Step/convergence/ode-1000")
	b = @benchmarkable DifferentialEquations.solve($prob_de, $Tsit5(), maxiters=1e9, save_everystep = false)
	bg[i==1 ? "Small" : "Large"]["ODE"]["ODE - 1000sec"] = b
end

# ODE Convergence - Short
tspans = [(0.0,200.0), (0.0, 150.0)]
for i in eachindex(params)
	p = params[i]
	prob_de = remake(prob, p=p)
	sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e9, u0 = Model.ic_conv, tspan=tspans[i], save_everystep=false)
	convergence_plot(sol[end], prob_de, "From Converged State: Short ODE", "results/simTimings/"*(i==1 ? "small" : "large")*"Step/convergence/ode-short")
	b = @benchmarkable DifferentialEquations.solve($prob_de, $Tsit5(), maxiters=1e9, save_everystep = false)
	bg[i==1 ? "Small" : "Large"]["ODE"]["ODE - Short"] = b
end

# Continuation Convergence
function early_abort((x, f, J, res, iteration, itlinear, options); kwargs...)
	if res < 5e2
		return true
	else
		return false
	end
end

lens = @optic _.step

tmp = Model.params_cont
pSmall = @set tmp.na_step = 0.1
pLarge = @set tmp.na_step = 0.5
pLarge = @set pLarge.k_step = 0.2
pLarge = @set pLarge.l_step = -0.2

params = [pSmall, pLarge]
ds = [1.0, 0.4]
for i in eachindex(params)
	if i>1
	p = params[i]
	bp = BifurcationProblem(Model.ode_cont!, Model.ic_conv, p, lens;
		record_from_solution = (x, p) -> (V = x[Model.plot_idx]),)

	# 1 pulse solution
	prob_cont = ODEProblem(Model.ode_cont!, Model.ic_conv, (0.0, 0.5216), p, abstol=1e-10, reltol=1e-8)
	sol_pulse = DifferentialEquations.solve(prob_cont, Tsit5())

	opts_br = ContinuationPar(p_min = 0.0, p_max = 1.0, max_steps = 50, tol_stability = 1e-8, ds=ds[i], dsmax=1.0, 
	detect_bifurcation=0, detect_fold=false, newton_options=NewtonPar(verbose=true, tol=1e-10))

	# Shooting method
	bpsh, cish = BK.generate_ci_problem(ShootingProblem(M=1, update_section_every_step=0), #update_section_every_step=0 avoids bpsh being perturbed between benchmark runs
	bp, prob_cont, sol_pulse, 0.5216; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

	brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
		verbosity = 3, callback_newton = early_abort
	)

	# Check converged
	convergence_plot(brpo_sh.sol[end].x[1:end-1], remake(prob_cont, p=@set p.step = 1.0), "From Converged State: Shooting", "results/simTimings/"*(i==1 ? "small" : "large")*"Step/convergence/cont-shooting")

	reducedOpts = ContinuationPar(p_min = 0.0, p_max = 1.0, max_steps = 50, tol_stability = 1e-8, 
	ds=1.0, dsmax=1.0, detect_bifurcation=0, detect_fold=false, newton_options=NewtonPar(tol=1e-10))

	b = @benchmarkable continuation($bpsh, $cish, $PALC(), $reducedOpts; callback_newton = early_abort)
	bg[i==1 ? "Small" : "Large"]["Cont"]["Cont - Shooting"] = b
	end
end

println("Reached the end of the script. Just running benchmark now.")
t = run(bg, seconds=20)
plot(t["Small"]["ODE"], yaxis=:log10, dpi=300, size=(450,300), title="Small Perturbation")
plot!(t["Small"]["Cont"], yaxis=:log10, linestyle=:dot, legend=:bottomleft, xaxis=nothing, ylabel="Time (ms)", yformatter=x->x/1e6)
savefig("results/simTimings/smallStep/smallTimings.pdf")
plot(t["Large"]["ODE"], yaxis=:log10, dpi=300, size=(450,300), title="Large Perturbation")
plot!(t["Large"]["Cont"], yaxis=:log10, linestyle=:dot, legend=:topleft, xaxis=nothing, ylabel="Time (ms)", yformatter=x->x/1e6)
savefig("results/simTimings/largeStep/largeTimings.pdf")

BenchmarkTools.save("results/simTimings/data.json", t)
