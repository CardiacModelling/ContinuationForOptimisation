using Revise, ModelingToolkit, LinearAlgebra
using DifferentialEquations, Plots
using BifurcationKit
const BK = BifurcationKit

indexof(sym, syms) = findfirst(isequal(sym),syms)

@variables t E(t) x(t) u(t) SS0(t) SS1(t) 	# independent and dependent variables
@parameters U0 τ J E0 τD U0 τF τS α    		# parameters
D = Differential(t) 				# define an operator for the differentiation w.r.t. time

# define the model
@named NMmodel = ODESystem([SS0 ~ J * u * x * E + E0,
	SS1 ~ α * log(1 + exp(SS0 / α)),
	D(E) ~ (-E + SS1) / τ,
	D(x) ~ (1.0 - x) / τD - u * x * E,
	D(u) ~ (U0 - u) / τF +  U0 * (1 - u) * E],
	defaults = Dict(E => 0.238616, x => 0.982747, u => 0.367876,
	α => 1.5, τ => 0.013, J => 3.07, E0 => -2.0, τD => 0.200, U0 => 0.3, τF => 1.5, τS => 0.007))

# get the vector field and jacobian
odeprob = ODEProblem(structural_simplify(NMmodel), [], (0.,10.), [], jac = true)
odefun = odeprob.f
F = (u,p) -> odefun(u,p,0)
J = (u,p) -> odefun.jac(u,p,0)

id_E0 = indexof(E0, parameters(NMmodel))
par_tm = odeprob.p

# we collect the differentials together in a problem
prob = BifurcationProblem(F, odeprob.u0, par_tm, (@lens _[id_E0]); J = J,
    record_from_solution = (x, p) -> (E = x[1], x = x[2], u = x[3]))

# continuation options
opts_br = ContinuationPar(p_min = -10.0, p_max = -0.9,
	# parameters to have a smooth result
	ds = 0.04, dsmax = 0.05,
	# this is to detect bifurcation points precisely with bisection
	detect_bifurcation = 3,
	# Optional: bisection options for locating bifurcations
	n_inversion = 8, max_bisection_steps = 25, nev = 3)

# continuation of equilibria
br = continuation(prob, PALC(tangent = Bordered()), opts_br; normC = norminf)

scene = plot(br, plotfold=false, markersize=3, legend=:topleft)